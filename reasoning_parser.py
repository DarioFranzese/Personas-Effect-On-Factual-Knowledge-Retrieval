from typing import Optional, Tuple

THINK_CLOSE_TAG = "</think>"
ALT_CLOSE_TAG = "assistantfinal"


def parse_reasoning_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, text

    close_tag = THINK_CLOSE_TAG if THINK_CLOSE_TAG in text else ALT_CLOSE_TAG
    if close_tag not in text:
        return None, text

    before, after = text.split(close_tag, 1)
    reasoning = before.strip() or None
    output = after.strip() or None

    return reasoning, output
