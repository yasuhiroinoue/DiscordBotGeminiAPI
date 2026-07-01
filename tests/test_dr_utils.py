"""Unit tests for dr_utils.extract_dr_result (Deep Research result extraction).

Run directly (no pytest dependency needed) — use the venv that has google-genai:
    /home/yasuhiro_inoue/disco/bin/python tests/test_dr_utils.py
or under pytest:
    /home/yasuhiro_inoue/disco/bin/python -m pytest tests/test_dr_utils.py

Two layers:
- Primary: duck-typed fakes (SimpleNamespace). Deterministic, no SDK dependency.
  extract_dr_result uses only getattr, so fakes faithfully model its contract.
- Secondary: a guarded smoke test built from the REAL SDK models. Those live in
  the PRIVATE path google.genai._gaos.types.interactions (they are NOT in public
  google.genai.types), so the imports are wrapped and the test SKIPS if the path
  is unavailable — degrading to skip rather than error on a future SDK reshuffle.
"""

import base64
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dr_utils import extract_dr_result  # noqa: E402

_PYTEST = "pytest" in sys.modules


class _Skip(Exception):
    """Raised to skip a test under the dependency-free direct runner."""


def _skip(msg):
    # Under pytest, use its native skip; under the direct __main__ runner, raise
    # _Skip (never call pytest.skip when pytest isn't driving — it would error).
    if _PYTEST:
        import pytest
        pytest.skip(msg)
    raise _Skip(msg)


# --- fixtures / fakes ---------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n-fake-image-bytes-"
PNG_B64 = base64.b64encode(PNG).decode()
PNG2 = b"second-image-bytes"
PNG2_B64 = base64.b64encode(PNG2).decode()


def _text(t):
    return NS(type="text", text=t)


def _image(data):
    return NS(type="image", data=data)


def _model_output(*content):
    return NS(type="model_output", content=list(content))


def _user_input():
    return NS(type="user_input", content=[])


def _thought(t):
    return NS(type="thought", content=[_text(t)])


def _interaction(output_text=None, steps=None):
    return NS(output_text=output_text, steps=steps)


# --- Case 1: output_text preferred; images pulled from steps -----------------

def test_case1_output_text_preferred():
    inter = _interaction(
        output_text="Report body",
        steps=[_model_output(_text("fallback-not-used"), _image(PNG_B64))],
    )
    text, images = extract_dr_result(inter)
    assert text == "Report body"
    assert images == [PNG]


# --- Case 2: output_text absent => concat current-turn model_output text ------

def test_case2_steps_text_fallback():
    inter = _interaction(
        output_text=None,
        steps=[_model_output(_text("A"), _text("B"))],
    )
    text, images = extract_dr_result(inter)
    assert text == "A\n\nB"
    assert images == []
    # whitespace-only output_text also triggers the fallback
    inter2 = _interaction(output_text="   ", steps=[_model_output(_text("real"))])
    assert extract_dr_result(inter2)[0] == "real"


# --- Case 3: multiple images (str + bytes data) ------------------------------

def test_case3_multiple_images():
    inter = _interaction(
        output_text="x",
        steps=[_model_output(_image(PNG_B64), _image(PNG2))],  # base64 str + raw bytes
    )
    _, images = extract_dr_result(inter)
    assert images == [PNG, PNG2]


# --- Case 4: no images -------------------------------------------------------

def test_case4_no_images():
    inter = _interaction(output_text="only text", steps=[_model_output(_text("hi"))])
    text, images = extract_dr_result(inter)
    assert text == "only text"
    assert images == []


# --- Case 5: thought / tool steps ignored ------------------------------------

def test_case5_non_model_output_ignored():
    inter = _interaction(
        output_text=None,
        steps=[
            _thought("private reasoning that must not leak"),
            NS(type="google_search_call", content=[_text("tool noise")]),
            _model_output(_text("real answer")),
        ],
    )
    text, images = extract_dr_result(inter)
    assert text == "real answer"
    assert images == []


# --- Case 6: chained interaction — only the latest turn counts (P2 scoping) ---

def test_case6_current_turn_scoping():
    inter = _interaction(
        output_text=None,
        steps=[
            _model_output(_text("OLD turn"), _image(PNG_B64)),   # before user_input
            _user_input(),
            _model_output(_text("NEW turn"), _image(PNG2_B64)),  # current turn
        ],
    )
    text, images = extract_dr_result(inter)
    assert text == "NEW turn"
    assert images == [PNG2]


# --- Case 7: one undecodable image skipped; text + other images survive ------

def test_case7_bad_image_isolated():
    inter = _interaction(
        output_text="report",
        steps=[_model_output(_image("a"), _image(PNG_B64))],  # "a": invalid base64
    )
    text, images = extract_dr_result(inter)
    assert text == "report"          # text unaffected
    assert images == [PNG]           # only the decodable image survives


# --- Case 8: empty completed => ("", []) -------------------------------------

def test_case8_empty():
    assert extract_dr_result(_interaction(output_text=None, steps=None)) == ("", [])
    assert extract_dr_result(_interaction(output_text="", steps=[])) == ("", [])


# --- Case 9: guarded real-SDK smoke test -------------------------------------

def _real_sdk_types():
    try:
        from google.genai._gaos.types.interactions.interaction import Interaction
        from google.genai._gaos.types.interactions.modeloutputstep import ModelOutputStep
        from google.genai._gaos.types.interactions.textcontent import TextContent
        from google.genai._gaos.types.interactions.imagecontent import ImageContent
        return Interaction, ModelOutputStep, TextContent, ImageContent
    except ImportError:
        return None


def test_case9_real_sdk_smoke():
    types_ = _real_sdk_types()
    if types_ is None:
        _skip("private google.genai._gaos.types.interactions not importable")
    Interaction, ModelOutputStep, TextContent, ImageContent = types_
    inter = Interaction(
        status="completed",
        output_text="Real report",
        steps=[ModelOutputStep(content=[TextContent(text="Real report"),
                                        ImageContent(data=PNG_B64)])],
    )
    text, images = extract_dr_result(inter)
    assert text == "Real report"
    assert images == [PNG]


_CASES = [
    ("Case1 output_text preferred", test_case1_output_text_preferred),
    ("Case2 steps text fallback", test_case2_steps_text_fallback),
    ("Case3 multiple images", test_case3_multiple_images),
    ("Case4 no images", test_case4_no_images),
    ("Case5 thought/tool ignored", test_case5_non_model_output_ignored),
    ("Case6 current-turn scoping", test_case6_current_turn_scoping),
    ("Case7 bad image isolated", test_case7_bad_image_isolated),
    ("Case8 empty completed", test_case8_empty),
    ("Case9 real-SDK smoke", test_case9_real_sdk_smoke),
]


def _run_all():
    passed = skipped = 0
    for name, fn in _CASES:
        try:
            fn()
        except _Skip as e:
            print(f"{name}: SKIPPED ({e})")
            skipped += 1
            continue
        print(f"{name}: OK")
        passed += 1
    print(f"ALL TESTS PASSED ({passed} passed, {skipped} skipped)")


if __name__ == "__main__":
    _run_all()
