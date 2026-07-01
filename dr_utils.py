"""Pure helpers for extracting results from a Gemini Interactions API response.

Kept import-side-effect-free (no Discord/Vertex initialization) so it can be
unit-tested without importing GeminiDiscordBot.py, mirroring markdown_utils.py.

Targets the google-genai 2.10.0 Interaction schema: a completed interaction
exposes its result via `interaction.output_text` and
`interaction.steps[].content[]` (model_output steps → text/image content, with
images carrying base64 in `ImageContent.data`). The pre-2.10 `interaction.outputs`
field was removed. Only duck-typed attribute access (`getattr`) is used here, so
this works against the real SDK models without importing SDK internals.
"""

import base64
import logging


def _current_turn_steps(steps):
    """Return only the steps of the latest turn: everything after the last
    `user_input` step.

    Deep Research chains interactions via `previous_interaction_id`, so `steps`
    can contain earlier turns. Restricting to the segment after the final
    `user_input` mirrors the SDK's `output_text` semantics and prevents an older
    turn's report/visualizations from leaking into the current result.
    """
    if not steps:
        return []
    start = 0
    for idx in range(len(steps) - 1, -1, -1):
        if getattr(steps[idx], "type", None) == "user_input":
            start = idx + 1
            break
    return steps[start:]


def _decode_image(data):
    """Decode base64 str / bytes to raw bytes, or return None if undecodable."""
    if not data:
        return None
    try:
        return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    except Exception as exc:
        logging.warning("Skipping undecodable Deep Research image: %s", exc)
        return None


def extract_dr_result(interaction):
    """Extract ``(report_text, image_bytes_list)`` from a completed Interaction.

    - text: prefer ``interaction.output_text``; if empty, concatenate text
      content from the latest turn's ``model_output`` steps.
    - images: raw bytes decoded from image content in the latest turn's
      ``model_output`` steps. A single undecodable image is skipped and never
      affects the text or the other images (text and image extraction are
      independent).

    Returns ``("", [])`` when nothing usable is present. `thought` and tool
    steps are ignored (only `model_output` steps are considered).
    """
    steps = _current_turn_steps(getattr(interaction, "steps", None))

    # --- text ---
    text = (getattr(interaction, "output_text", None) or "").strip()
    if not text:
        parts = []
        for step in steps:
            if getattr(step, "type", None) != "model_output":
                continue
            for content in (getattr(step, "content", None) or []):
                if getattr(content, "type", None) == "text":
                    t = getattr(content, "text", None)
                    if t:
                        parts.append(t)
        text = "\n\n".join(parts).strip()

    # --- images (independent of text; per-image failures are skipped) ---
    images = []
    for step in steps:
        if getattr(step, "type", None) != "model_output":
            continue
        for content in (getattr(step, "content", None) or []):
            if getattr(content, "type", None) == "image":
                decoded = _decode_image(getattr(content, "data", None))
                if decoded is not None:
                    images.append(decoded)

    return text, images
