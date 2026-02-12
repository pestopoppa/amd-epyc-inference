"""Sanitize Unicode characters in model-generated Python code.

Models frequently copy special Unicode characters from question text into their
generated code (e.g., ° from "contact angle of 47°"). These cause SyntaxError
since Python doesn't accept them as operators or in identifiers.

This module replaces common Unicode lookalikes with their ASCII equivalents,
applied BEFORE ast.parse() and exec().
"""

from __future__ import annotations

import re

# Unicode → ASCII replacement map for code contexts
# Only includes characters that (a) models commonly copy from prompts and
# (b) have an obvious ASCII equivalent in Python code
_UNICODE_REPLACEMENTS: dict[str, str] = {
    # Mathematical operators
    "\u00d7": "*",   # × multiplication sign
    "\u00f7": "/",   # ÷ division sign
    "\u2212": "-",   # − minus sign (different from hyphen-minus)
    "\u2013": "-",   # – en dash (often used as minus)
    "\u2014": "-",   # — em dash
    "\u00b1": "+-",  # ± plus-minus (approximate)
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
    # Degree and unit symbols — replace with numeric value in string context
    "\u00b0": "",    # ° degree sign — strip (models write 47° meaning 47)
    # Quote lookalikes
    "\u2018": "'",   # ' left single curly quote
    "\u2019": "'",   # ' right single curly quote
    "\u201c": '"',   # " left double curly quote
    "\u201d": '"',   # " right double curly quote
    "\u00b4": "'",   # ´ acute accent
    "\u0060": "'",   # ` grave accent (backtick — rare in Python code)
    # Arrow-like
    "\u2192": "->",  # → rightwards arrow
    "\u2190": "<-",  # ← leftwards arrow (not valid Python but safer)
    # Whitespace lookalikes
    "\u00a0": " ",   # non-breaking space
    "\u2003": " ",   # em space
    "\u2002": " ",   # en space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM / zero-width no-break space
    # Superscripts/subscripts that break identifiers
    "\u00b2": "**2",  # ² superscript 2
    "\u00b3": "**3",  # ³ superscript 3
    # Subscript digits (models copy from physics notation like ω₁, E₂)
    "\u2080": "0",  # ₀
    "\u2081": "1",  # ₁
    "\u2082": "2",  # ₂
    "\u2083": "3",  # ₃
    "\u2084": "4",  # ₄
    "\u2085": "5",  # ₅
    "\u2086": "6",  # ₆
    "\u2087": "7",  # ₇
    "\u2088": "8",  # ₈
    "\u2089": "9",  # ₉
    # Superscript digits beyond ² and ³
    "\u2070": "**0",  # ⁰
    "\u00b9": "**1",  # ¹
    "\u2074": "**4",  # ⁴
    "\u2075": "**5",  # ⁵
    "\u2076": "**6",  # ⁶
    "\u2077": "**7",  # ⁷
    "\u2078": "**8",  # ⁸
    "\u2079": "**9",  # ⁹
}

# Pre-compile regex for all replaceable chars
_UNICODE_PATTERN = re.compile(
    "[" + re.escape("".join(_UNICODE_REPLACEMENTS.keys())) + "]"
)


def sanitize_code_unicode(code: str) -> str:
    """Replace common Unicode characters with ASCII equivalents in code.

    Fast path: if no non-ASCII chars, returns immediately (O(n) scan).
    Replacement path: single regex pass over the string.

    Args:
        code: Python source code potentially containing Unicode chars.

    Returns:
        Sanitized code with Unicode replacements applied.
    """
    if not code or code.isascii():
        return code

    return _UNICODE_PATTERN.sub(
        lambda m: _UNICODE_REPLACEMENTS.get(m.group(), m.group()),
        code,
    )
