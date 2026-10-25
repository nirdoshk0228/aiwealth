"""Display strings for tool outputs. The model is told to quote these verbatim, so it never
converts units itself (0.0138 -> "1.38%"), and the grounding check can find every number of an
answer in the tool outputs as text."""

import math


def pct(x: float | None, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def signed_pct(x: float | None, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    s = f"{abs(x) * 100:.{digits}f}%"
    return s if float(s[:-1]) == 0 else ("+" if x > 0 else "-") + s


def inr(x: float | None) -> str:
    """Rupees with Indian digit grouping: 1234567.8 -> "Rs 12,34,568"."""
    if x is None or not math.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    digits = f"{round(abs(x)):d}"
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{sign}Rs {digits}"


def ratio(x: float | None, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"
