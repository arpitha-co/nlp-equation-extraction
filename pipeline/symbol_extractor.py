"""
Simple rule-based symbol extraction from LaTeX equations.

This finds symbol keys from equations. No definition/meaning extraction
is performed — symbols are returned as plain name lists.
"""

import json
import re
import sys
from typing import Any, Dict, List, Mapping, Optional


WRAPPER_COMMANDS = {
    "bar",
    "boldsymbol",
    "bra",
    "hat",
    "ket",
    "mathbb",
    "mathbf",
    "mathcal",
    "overline",
    "tilde",
    "vec",
    "widehat",
    "widetilde",
}

KEEP_CONTENT_COMMANDS = WRAPPER_COMMANDS | {"cal", "mathrm", "sqrt"}

DROP_CONTENT_COMMANDS = {
    "operatorname",
    "text",
    "textit",
    "textnormal",
    "textrm",
    "textstyle",
    "vphantom",
}

OPERATOR_COMMANDS = {
    "begin",
    "big",
    "bigg",
    "biggl",
    "biggr",
    "bigl",
    "bigr",
    "binom",
    "cap",
    "cdot",
    "coloneqq",
    "cos",
    "dagger",
    "dfrac",
    "displaystyle",
    "dot",
    "dots",
    "dotsb",
    "dotsc",
    "dotsi",
    "dotso",
    "end",
    "equiv",
    "exp",
    "forall",
    "frac",
    "ge",
    "geq",
    "in",
    "int",
    "le",
    "left",
    "leq",
    "llangle",
    "ln",
    "log",
    "langle",
    "ldots",
    "kern",
    "hskip",
    "mid",
    "nabla",
    "neq",
    "not",
    "oplus",
    "otimes",
    "partial",
    "Pr",
    "quad",
    "qquad",
    "rangle",
    "right",
    "rightarrow",
    "rrangle",
    "rm",
    "sin",
    "sinc",
    "sup",
    "prod",
    "sum",
    "uparrow",
    "downarrow",
    "tfrac",
    "times",
    "to",
    "notin",
    "det",
    "min",
    "max",
    "wedge",
    "cosh",
    "sinh"
    "cos",
    "sin",
    "boldsymbol"
}

PLAIN_WORDS = {
    "Maximize",
    "Minimize",
    "Subject",
    "arg",
    "bra",
    "d",
    "for",
    "if",
    "ket",
    "langle",
    "left",
    "otherwise",
    "rangle",
    "right",
    "rrangle",
    "llangle",
    "kern",
    "hskip",
    "vphantom",
    "where",
    "differential",
}

DUMMY_INDEX_SUBSCRIPTS = {
    "i",
    "j",
    "k",
    "l",
    "r",
    "s",
    "u",
    "v",
    "w",
    "ij",
    "ji",
    "kl",
    "lk",
    "mn",
    "nm",
    "uv",
    "st",
}

DIRAC_DIRECTION_COMMANDS = {
    "downarrow",
    "leftarrow",
    "nearrow",
    "nwarrow",
    "rightarrow",
    "searrow",
    "swarrow",
    "uparrow",
}

GREEK_TO_NAME = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "η": "eta",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "ϕ": "phi",
    "ψ": "psi",
    "ω": "omega",
    "ℏ": "hbar",
}

NAME_TO_GREEK = {value: key for key, value in GREEK_TO_NAME.items()}


def _replace_command_group(text: str, command: str, keep_content: bool) -> str:
    pattern = re.compile(rf"\\{command}\s*\{{([^{{}}]*)\}}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(lambda m: f" {m.group(1)} " if keep_content else " ", text)
    return text


def _strip_outer_braces(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text[1:-1].strip()
    return text


def _normalize_symbol_name(text: str) -> str:
    text = _strip_outer_braces(text)
    text = text.replace("%", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = _strip_wrappers(text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = re.sub(r"_\{([^{}]+)\}", lambda m: "_" + _normalize_subscript(m.group(1)), text)
    text = re.sub(r"\s+", "", text)
    return text


def _strip_wrappers(text: str) -> str:
    wrapper_names = "|".join(sorted(WRAPPER_COMMANDS, key=len, reverse=True))
    pattern = re.compile(rf"\\(?:{wrapper_names})\s*\{{([^{{}}]+)\}}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1", text)
    return text


def _format_wrapped_content(content: str) -> str:
    content = _strip_outer_braces(content)
    content = content.replace("%", "")
    content = content.replace("\\left", "").replace("\\right", "")
    wrapper_names = "|".join(sorted(WRAPPER_COMMANDS, key=len, reverse=True))
    pattern = re.compile(rf"^\\({wrapper_names})\s*\{{(.+)\}}$")
    match = pattern.match(content)
    if match:
        nested_command = match.group(1)
        nested_content = _format_wrapped_content(match.group(2))
        if nested_command in ("ket", "bra", "bar"):
            return nested_content
        return f"{nested_command}{{{nested_content}}}"
    return _normalize_symbol_name(content)


def _clean_dirac_content(content: str) -> str:
    direction_names = "|".join(sorted(DIRAC_DIRECTION_COMMANDS, key=len, reverse=True))
    content = re.sub(rf"\s*,?\s*\\(?:{direction_names})\b", "", content)
    content = re.sub(r"\s*,\s*$", "", content)
    return content.strip()


def _wrapped_symbol_key(command: str, content: str, subscript: str = "") -> str:
    if command in ("ket", "bra"):
        content = _clean_dirac_content(content)
    content = _format_wrapped_content(content)
    subscript = _normalize_subscript(subscript or "")
    if command in ("ket", "bra", "bar"):
        symbol = content
    else:
        symbol = f"{command}{{{content}}}"
    if subscript:
        symbol = f"{symbol}_{subscript}"
    return symbol


def _dirac_symbol_key(command: str, content: str, subscript: str = "") -> str:
    """Canonical key for bra/ket notation from either macro or vertical-bar form."""
    return _wrapped_symbol_key(command, content, subscript)


def _normalize_superscript(superscript: str) -> str:
    superscript = _strip_outer_braces(superscript)
    superscript = _strip_wrappers(superscript)
    superscript = re.sub(r"\\([A-Za-z]+)", r"\1", superscript)
    superscript = re.sub(r"[^A-Za-z0-9^]+", "", superscript)
    return superscript


def _is_meaningful_superscript(superscript: str) -> bool:
    normalized = _normalize_superscript(superscript)
    return bool(re.search(r"(dagger|star|ast|prime)", normalized))


def _normalize_formula(equation: str) -> str:
    text = re.sub(r"%\s*", "", equation)
    text = re.sub(
        r"\\(?:kern|hskip)\s*-?\s*\d+(?:\.\d+)?\s*(?:pt|em|ex|mu|mm|cm|in)",
        " ",
        text,
    )
    text = re.sub(r"\\mathrm\s*\{\s*d\s*\}", " ", text)
    text = re.sub(r"\\rm\s+", "", text)

    for command in DROP_CONTENT_COMMANDS:
        text = _replace_command_group(text, command, keep_content=False)

    for command in KEEP_CONTENT_COMMANDS:
        text = _replace_command_group(text, command, keep_content=True)

    text = re.sub(r"(?<![_^])\{([A-Za-z]+)\}", r" \1 ", text)

    return text


def _is_delta_modifier(text: str, end_index: int) -> bool:
    """Return True when delta/Delta prefixes another variable, not a named symbol."""
    suffix = text[end_index:]
    next_symbol = re.match(r"\s*(?:\\([A-Za-z]+)|([A-Za-z]))", suffix)
    if next_symbol and (next_symbol.group(1) or next_symbol.group(2)) == "t":
        return False
    if next_symbol:
        return True
    return False


def _normalize_subscript(subscript: str) -> str:
    subscript = _strip_outer_braces(subscript)
    subscript = _strip_wrappers(subscript)
    subscript = re.sub(r"\\([A-Za-z]+)", r"\1", subscript)
    subscript = subscript.replace("'", "prime")
    subscript = re.sub(r"[^A-Za-z0-9+-]+", "", subscript)
    return subscript


def _replace_unicode_symbols(text: str) -> str:
    for char, name in GREEK_TO_NAME.items():
        text = text.replace(char, name)
    text = text.replace("′", "prime").replace("`", "prime")
    text = text.replace("\u2018", "prime").replace("\u2019", "prime")
    text = text.replace("†", "dagger")
    return text


def _strip_function_arguments(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol.endswith(")"):
        return symbol

    depth = 0
    for index in range(len(symbol) - 1, -1, -1):
        char = symbol[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                prefix = symbol[:index].strip()
                if prefix:
                    return prefix
    return symbol


def _split_base_subscript(symbol: str) -> tuple[str, str]:
    match = re.match(r"^(.+?)_([A-Za-z0-9+\-]+)(?:\^\{[^{}]+\})?$", symbol)
    if not match:
        return symbol, ""
    return match.group(1), match.group(2)


def _is_wrapper_symbol(symbol: str) -> bool:
    return bool(re.match(r"^[A-Za-z]+\{", symbol))


def normalize_symbol(raw_symbol: str, context: Optional[Mapping[str, Any]] = None) -> str:
    """Normalize notation for matching while preserving meaningful decorations."""
    context = context or {}
    symbol = str(raw_symbol or "").strip()
    if not symbol:
        return ""

    symbol = _replace_unicode_symbols(symbol)
    symbol = re.sub(r"\s+", "", symbol)
    symbol = symbol.replace("\\left", "").replace("\\right", "")
    symbol = symbol.replace("\\prime", "prime")

    dirac_match = re.match(
        r"^(?:\\)?(ket|bra)\{(.+)\}"
        r"(?:_(?:\{([^{}]+)\}|([A-Za-z0-9\\,+\-]+)))?"
        r"(?:\^\{([^{}]+)\})?$",
        symbol,
    )
    if dirac_match:
        command = dirac_match.group(1)
        content = dirac_match.group(2)
        subscript = dirac_match.group(3) or dirac_match.group(4) or ""
        normalized = _dirac_symbol_key(command, content, subscript)
        superscript = dirac_match.group(5) or ""
        if superscript:
            normalized = f"{normalized}^{{{_normalize_superscript(superscript)}}}"
        return normalized

    symbol = _strip_function_arguments(symbol)

    # Normalize subscripts first so they can retain their backslashes for _strip_wrappers!
    idx = 0
    while True:
        idx = symbol.find("_{", idx)
        if idx == -1:
            break
        close_idx = _find_matching_brace(symbol, idx + 1)
        if close_idx != -1:
            sub = symbol[idx+2:close_idx]
            norm_sub = _normalize_subscript(sub)
            symbol = symbol[:idx] + "_{" + norm_sub + "}" + symbol[close_idx+1:]
            idx = idx + 2 + len(norm_sub) + 1
        else:
            idx += 2

    # Convert common LaTeX spellings into the project's symbol-key spelling.
    symbol = re.sub(r"\\([A-Za-z]+)", r"\1", symbol)
    symbol = re.sub(r"\^\{?\\?dagger(?:\^?\d+)?\}?", "^{dagger}", symbol)
    symbol = re.sub(r"\^\{?(?:\\?prime|prime)\}?", "", symbol)
    symbol = re.sub(r"prime(?=_[A-Za-z0-9+\-]+$)", "", symbol)
    symbol = re.sub(r"prime$", "", symbol)

    # Normalize repeated dagger artifacts such as hat{a}_x^{dagger^2}.
    symbol = re.sub(r"\^\{dagger\^\d+\}", "^{dagger}", symbol)

    base, subscript = _split_base_subscript(symbol)
    allow_family = context.get("allow_family_normalization", False)
    if allow_family and subscript and subscript in DUMMY_INDEX_SUBSCRIPTS:
        return base

    return symbol


def _add_symbol(symbols: List[str], symbol: str) -> None:
    symbol = normalize_symbol(symbol)
    if symbol and symbol not in symbols:
        symbols.append(symbol)


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _parse_wrapper_at(text: str, start: int):
    wrapper_names = sorted(WRAPPER_COMMANDS, key=len, reverse=True)
    for command in wrapper_names:
        prefix = "\\" + command
        if not text.startswith(prefix, start):
            continue

        index = start + len(prefix)
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "{":
            return None

        close_index = _find_matching_brace(text, index)
        if close_index == -1:
            return None

        content = text[index + 1:close_index]
        end_index = close_index + 1
        subscript = ""
        superscript = ""

        if end_index < len(text) and text[end_index] == "'":
            prime_count = 0
            while end_index < len(text) and text[end_index] == "'":
                prime_count += 1
                end_index += 1
            superscript = "prime" * prime_count

        if end_index < len(text) and text[end_index] == "_":
            sub_start = end_index + 1
            if sub_start < len(text) and text[sub_start] == "{":
                sub_close = _find_matching_brace(text, sub_start)
                if sub_close != -1:
                    subscript = text[sub_start + 1:sub_close]
                    end_index = sub_close + 1
            else:
                match = re.match(r"[A-Za-z0-9\\]+", text[sub_start:])
                if match:
                    subscript = match.group(0)
                    end_index = sub_start + len(subscript)

        if end_index < len(text) and text[end_index] == "^":
            super_start = end_index + 1
            if super_start < len(text) and text[super_start] == "{":
                super_close = _find_matching_brace(text, super_start)
                if super_close != -1:
                    candidate = text[super_start + 1:super_close]
                    if _is_meaningful_superscript(candidate):
                        superscript = candidate
                        end_index = super_close + 1
            else:
                match = re.match(r"[A-Za-z0-9\\]+", text[super_start:])
                if match and _is_meaningful_superscript(match.group(0)):
                    superscript = match.group(0)
                    end_index = super_start + len(superscript)

        return command, content, subscript, superscript, end_index

    return None


def _extract_wrapped_symbols(equation: str) -> tuple[List[str], str]:
    symbols: List[str] = []
    masked_parts = []
    index = 0

    while index < len(equation):
        parsed = _parse_wrapper_at(equation, index)
        if not parsed:
            masked_parts.append(equation[index])
            index += 1
            continue

        command, content, subscript, superscript, end_index = parsed
        if command in {"ket", "bra"}:
            key = _dirac_symbol_key(command, content, subscript)
        else:
            key = _wrapped_symbol_key(command, content, subscript)
        if superscript:
            key = f"{key}^{{{_normalize_superscript(superscript)}}}"

        _add_symbol(symbols, key)
        masked_parts.append(" ")
        index = end_index

    masked_equation = "".join(masked_parts)
    return symbols, masked_equation


def _flatten_angle_bracket_expressions(equation: str) -> str:
    """
    Keep the contents of expectation values/inner products while removing delimiters.

    This prevents expressions like ``\\langle a|b\\rangle`` from becoming one
    malformed ``bra{...ket{...}}`` symbol, while still allowing the normal
    symbol scanner to see ``a`` and ``b``.
    """
    angle_pattern = re.compile(
        r"\\langle(?P<content>[\s\S]{0,400}?)\\rangle"
        r"(?P<sub>_(?:\{[^{}]+\}|[A-Za-z0-9\\,+\-]+))?"
    )

    def replace(match: re.Match) -> str:
        content = match.group("content").replace("|", " ")
        subscript = match.group("sub") or ""
        return f" {content} {subscript} "

    return angle_pattern.sub(replace, equation)


def _mask_double_angle_products(equation: str) -> str:
    """Remove custom double-angle inner products before plain symbol scanning."""
    equation = re.sub(
        r"\\left\s*\\llangle[\s\S]{0,600}?\\right\s*\\rrangle",
        " ",
        equation,
    )
    return re.sub(r"\\llangle[\s\S]{0,600}?\\rrangle", " ", equation)


def _extract_vertical_dirac_symbols(equation: str) -> tuple[List[str], str]:
    """Extract ``|x\\rangle`` and bare ``\\langle x|`` forms consistently."""
    symbols: List[str] = []
    masked_parts: List[str] = []
    index = 0

    def read_subscript(start: int) -> tuple[str, int]:
        if start >= len(equation) or equation[start] != "_":
            return "", start

        sub_start = start + 1
        if sub_start < len(equation) and equation[sub_start] == "{":
            sub_close = _find_matching_brace(equation, sub_start)
            if sub_close != -1:
                return equation[sub_start + 1:sub_close], sub_close + 1
            return "", start

        match = re.match(r"[A-Za-z0-9\\,+\-]+", equation[sub_start:])
        if match:
            return match.group(0), sub_start + len(match.group(0))
        return "", start

    while index < len(equation):
        if equation[index] == "|":
            close_index = equation.find(r"\rangle", index + 1)
            next_bar = equation.find("|", index + 1)
            if close_index != -1 and (next_bar == -1 or close_index < next_bar):
                content = equation[index + 1:close_index].strip()
                if 0 < len(content) <= 180:
                    end_index = close_index + len(r"\rangle")
                    subscript, end_index = read_subscript(end_index)
                    _add_symbol(symbols, _dirac_symbol_key("ket", content, subscript))
                    masked_parts.append(" ")
                    index = end_index
                    continue

        if equation.startswith(r"\langle", index):
            content_start = index + len(r"\langle")
            close_index = equation.find("|", content_start)
            close_rangle = equation.find(r"\rangle", content_start)
            if close_index != -1 and (close_rangle == -1 or close_index < close_rangle):
                content = equation[content_start:close_index].strip()
                if 0 < len(content) <= 180:
                    end_index = close_index + 1
                    subscript, end_index = read_subscript(end_index)
                    _add_symbol(symbols, _dirac_symbol_key("bra", content, subscript))
                    masked_parts.append(" ")
                    index = end_index
                    continue

        masked_parts.append(equation[index])
        index += 1

    return symbols, "".join(masked_parts)


def extract_symbol_list(equation: str) -> List[str]:
    """
    Extract normalized symbol names from a LaTeX equation.
    Uses regex patterns to parse variables and Greek letters,
    while removing LaTeX formatting commands (like \frac, \sum).

    Examples:
    - \\\\psi -> psi
    - \\\\vec{x} -> vec{x}
    - x_{ij} -> x_ij
    
    Args:
        equation (str): The raw LaTeX equation.
        
    Returns:
        List[str]: A list of unique, normalized symbol names found in the equation.
    """
    equation = _mask_double_angle_products(equation)
    equation = _flatten_angle_bracket_expressions(equation)
    symbols, equation = _extract_vertical_dirac_symbols(equation)
    wrapped_symbols, text = _extract_wrapped_symbols(equation)
    for symbol in wrapped_symbols:
        _add_symbol(symbols, symbol)
    text = _normalize_formula(text)

    # LaTeX command symbols such as \\psi, \\hbar, \\lambda.
    latex_subscript = r"(?:_(?:\{([A-Za-z0-9\\+\-']+)\}|([A-Za-z0-9\\']+)))?"
    for match in re.finditer(r"\\([A-Za-z]+)" + latex_subscript, text):
        command = match.group(1)
        if command in OPERATOR_COMMANDS or command in KEEP_CONTENT_COMMANDS:
            continue
        if command in {"delta", "Delta"} and not (match.group(2) or match.group(3)):
            if _is_delta_modifier(text, match.end()):
                continue
        braced_subscript = match.group(2)
        plain_subscript = match.group(3)
        if braced_subscript:
            symbol = f"{command}_{{{_normalize_subscript(braced_subscript)}}}"
        elif plain_subscript:
            symbol = f"{command}_{_normalize_subscript(plain_subscript)}"
        else:
            symbol = command
        _add_symbol(symbols, symbol)

    # Remove remaining commands before finding plain Latin symbols.
    plain = re.sub(r"\\[A-Za-z]+" + latex_subscript, " ", text)

    plain_subscript = r"(?:_(?:\{([A-Za-z0-9+\-']+)\}|([A-Za-z0-9']+)))?"
    for match in re.finditer(r"(?<![A-Za-z])([A-Za-z]+)" + plain_subscript, plain):
        base = match.group(1)
        braced_subscript = match.group(2)
        plain_subscript = match.group(3)

        if base in PLAIN_WORDS:
            continue

        if len(base) > 1 and not braced_subscript and not plain_subscript and not base.isupper():
            for char in base:
                _add_symbol(symbols, char)
            continue

        if braced_subscript:
            symbol = f"{base}_{{{_normalize_subscript(braced_subscript)}}}"
        elif plain_subscript:
            symbol = f"{base}_{_normalize_subscript(plain_subscript)}"
        else:
            symbol = base
        _add_symbol(symbols, symbol)

    return symbols


def extract_symbols_from_equation(equation: str) -> List[str]:
    """
    Public wrapper for equation-symbol extraction.
    
    Args:
        equation (str): Raw LaTeX equation string.
        
    Returns:
        List[str]: A list of extracted symbol names.
    """
    return extract_symbol_list(equation)


def extract_symbols(equation: str) -> List[str]:
    """
    Return the list of symbol names extracted from an equation.
    Alias for extract_symbol_list.
    
    Args:
        equation (str): Raw LaTeX equation string.
        
    Returns:
        List[str]: A list of extracted symbol names.
    """
    return extract_symbol_list(equation)


def add_symbols_to_dataset(dataset_path: str) -> None:
    """
    Iterates through a JSON dataset of equations and enriches each entry 
    with the list of extracted mathematical symbols.
    
    Args:
        dataset_path (str): The file path to the dataset JSON.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for paper_id, paper_data in dataset.items():
        for equation_id, eq_info in paper_data.items():
            equation = eq_info.get("equation", "")
            symbols = extract_symbol_list(equation)
            eq_info["symbols"] = symbols
            audit_trail = eq_info.setdefault("audit-trail", {})
            audit_trail["find_symbols"] = f"Found symbols: {', '.join(symbols) or 'none'}"

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python symbol_extractor.py <dataset.json>")
        sys.exit(1)

    add_symbols_to_dataset(sys.argv[1])
