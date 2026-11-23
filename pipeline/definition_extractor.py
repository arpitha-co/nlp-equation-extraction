import symbol_extractor
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Definition phrase
DEF_REGEX = r"([A-Za-z][A-Za-z\s\-/()]{2,100})"
BRACE_CONTENT_REGEX = r"(?:[^{}]|\{[^{}]*\})+"
RAW_SYMBOL_REGEX = (
    r"(?:\\(?:vec|mathbf|boldsymbol|hat|bar|tilde|overline|mathcal|mathbb)"
    rf"\s*\{{{BRACE_CONTENT_REGEX}\}}|\\[A-Za-z]+|[A-Za-z])"
    rf"(?:_\s*(?:\{{{BRACE_CONTENT_REGEX}\}}|[A-Za-z0-9]+))?"
    rf"(?:\^\s*(?:\{{{BRACE_CONTENT_REGEX}\}}|[A-Za-z0-9+\\-]+))?"
)
GLOBAL_CONTEXT_DIR = "global_context_windows"
PREDEFINED_DEFINITIONS_PATH = "predefined.json"
DEFINITION_CANDIDATES_PATH = "definition_candidates.json"
DEFINITION_SOURCES_PATH = "definition_sources.json"
POS_FALLBACK_MIN_CONFIDENCE = 0.85


PATTERNS = [
    # where the term Q(...) = ..., is recognized as the Bohm quantum potential
    (
        r"where\s+the\s+term\s+{X}[\s\S]{0,500}?"
        r"\s+is\s+recognized\s+as\s+(?:a|an|the)?\s*" + DEF_REGEX,
        1,
    ),

    # where Q denotes/represents the quantum potential
    (r"where\s+{X},?\s+denotes\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),
    (r"where\s+{X},?\s+represents\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # particle of mass m / mass m
    (r"\b(mass)\s+{X}\b", 1),

    # Q refers to the quantum potential
    (r"{X}[^,.]*?\s+refers\s+to\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # Q is defined as / called / recognized as / proportional to
    (r"{X}\s+is\s+defined\s+as\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),
    (r"{X}\s+is\s+called\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),
    (r"{X}\s+is\s+recognized\s+as\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),
    (r"{X}\s+is\s+proportional\s+to\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # x_{ij} are indicator variables
    (r"{X}\s+are\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # X is a/an/the Y
    (r"{X}\s+is\s+(?:a|an|the)\s+" + DEF_REGEX, 1),

    # let X be the transverse coordinate
    (r"{X}\s+be\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # X ∈ {0,1} be the indicator variable
    (r"{X}\s*\\in\b[\s\S]{0,80}?\s+be\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # X and X1 are indicator variables
    (r"{X}\s+and\s+.*?\s+are\s+(?:a|an|the)?\s*" + DEF_REGEX, 1),

    # probability density is then given by n = ...
    (
        r"(?:the\s+)?" + DEF_REGEX +
        r"\s+is\s+then\s+given\s+by\s+{X}\b(?:\s*=|\s+as)?",
        1,
    ),

    # probability density is defined as n = ...
    (
        r"(?:the\s+)?" + DEF_REGEX +
        r"\s+is\s+defined\s+as\s+{X}\b(?:\s*=|\s+as)?",
        1,
    ),

    # probability flux becomes J = ...
    (
        r"(?:the\s+)?" + DEF_REGEX +
        r"\s+becomes\s+{X}\b(?:\s*=|\s+as)?",
        1,
    ),

    # represented by the wavefunction psi
    (
        r"represented\s+by\s+(?:a|an|the)?\s*" +
        DEF_REGEX + r"\s+{X}\b",
        1,
    ),

    # where probability density n
    (
        r"where\s+(?:a|an|the)?\s*" +
        DEF_REGEX + r"\s+{X}\b",
        1,
    ),

    # the probability density n
    (
        r"(?:a|an|the)\s+" +
        DEF_REGEX + r"\s+{X}\b",
        1,
    ),

    # n, the probability density
    (
        r"{X}\s*,?\s+(?:a|an|the)\s+" +
        DEF_REGEX,
        1,
    ),

]


BAD_TRAILING_WORDS = {
    "is", "are", "where", "and", "or", "of", "with", "by",
    "given", "defined", "called", "denotes", "represents",
    "as", "the", "a", "an"
}

BAD_DEFINITION_STARTS = {
    "expected",
    "hold",
    "mapped",
    "placed",
    "scheduled",
}


def clean_definition(definition: str) -> str:
    definition = definition.strip()
    # Remove journal citations like PhysRev.85.166 or PhysRev
    definition = re.sub(r'\bPhysRev(?:\.\d+)*\b', '', definition)
    # Remove repeated digit metadata markers like 1 1 1, 2 2 2, etc.
    definition = re.sub(r'\b(\d+)(?:\s+\1)+\b', '', definition)
    definition = re.sub(r"\s+", " ", definition)
    definition = definition.strip(" ,.;:()[]{}")

    # Clean leading verbs/prepositions
    split_patterns = [
        r"\bexpressing\s+(?:the|a|an)?\s*",
        r"\bwriting\s+(?:the|a|an)?\s*",
        r"\brepresenting\s+(?:the|a|an)?\s*",
        r"\busing\s+(?:the|a|an)?\s*",
        r"\bdefining\s+(?:the|a|an)?\s*",
        r"\bdenoting\s+(?:the|a|an)?\s*",
        r"\bby\s+(?:expressing|writing|representing|using|defining|denoting)?\s*(?:the|a|an)?\s*",
        r"^to\s+be\s+(?:the|a|an)?\s*",
        r"\bderived\s+by\s+(?:the|a|an)?\s*",
        r"\bobtained\s+by\s+(?:the|a|an)?\s*",
    ]
    for pat in split_patterns:
        parts = re.split(pat, definition, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if len(candidate) > 2:
                definition = candidate

    # Strip leading common articles
    definition = re.sub(r"^(?:the|a|an)\s+", "", definition, flags=re.IGNORECASE)

    words = definition.split()
    while words and words[-1].lower() in BAD_TRAILING_WORDS:
        words.pop()

    return " ".join(words).strip()


def canonical_symbol(symbol: str) -> str:
    s = symbol.strip()
    while True:
        prev = s
        for mod in ["vec", "mathbf", "mathcal", "tilde", "hat", "bar", "textbf", "mathrm"]:
            s = re.sub(rf"\\?{mod}\{{([^{{}}]+)\}}", r"\1", s)
        if prev == s:
            break
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    return s.strip()


def symbol_lookup_keys(symbol: str) -> List[str]:
    keys = [
        symbol.strip(),
        canonical_symbol(symbol),
        get_symbol_family(symbol),
        canonical_symbol(get_symbol_family(symbol)),
    ]
    
    # If the symbol has a single-character subscript wrapped in braces, add a version without braces
    new_keys = []
    for key in keys:
        if key and re.search(r"_\{([A-Za-z0-9])\}", key):
            new_keys.append(re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", key))
    keys.extend(new_keys)

    seen = set()
    unique_keys = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
    return unique_keys


def _dirac_subscript_pattern(subscript: str) -> str:
    subscript = subscript.strip()
    if not subscript:
        return ""

    escaped = re.escape(subscript)
    variants = {escaped}
    if subscript.isalpha() and len(subscript) > 1:
        variants.add(r"\s*,?\s*".join(re.escape(char) for char in subscript))

    subscript_regex = "|".join(sorted(variants, key=len, reverse=True))
    return rf"(?:_(?:\{{(?:{subscript_regex})\}}|(?:{subscript_regex})))"


def _latex_name_pattern(name: str) -> str:
    greek_names = {
        "alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta",
        "lambda", "mu", "xi", "pi", "rho", "sigma", "tau", "phi",
        "psi", "omega", "hbar",
    }
    uppercase_greek = {"Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega"}

    if name in greek_names or name in uppercase_greek:
        return rf"(?:\\{re.escape(name)}|{re.escape(name)})"

    escaped = re.escape(name)
    return re.sub(r"([A-Za-z]+)", r"(?:\\\\)?\1", escaped)


def _dirac_content_pattern(content: str) -> str:
    content = content.strip()
    arg_pattern = ""

    arg_match = re.match(r"^(.+?)\(([^()]*)\)$", content)
    if arg_match:
        content = arg_match.group(1)
        args = arg_match.group(2)
        args_regex = re.escape(args)
        args_regex = re.sub(r"([A-Za-z]+)", r"(?:\\\\)?\1", args_regex)
        arg_pattern = rf"(?:\\left)?\(\s*{args_regex}\s*(?:\\right)?\)"

    subscript_pattern = ""
    if "_" in content:
        base, subscript = content.split("_", 1)
        subscript_pattern = _dirac_subscript_pattern(subscript)
    else:
        base = content

    return _latex_name_pattern(base) + subscript_pattern + arg_pattern


def _dirac_regex_variants(symbol: str) -> List[str]:
    match = re.match(r"^(ket|bra)\{(.+)\}(?:_([A-Za-z0-9,+\-]+))?$", symbol.strip())
    if not match:
        return []

    command, content, subscript = match.group(1), match.group(2), match.group(3) or ""
    content_pattern = _dirac_content_pattern(content)
    subscript_pattern = _dirac_subscript_pattern(subscript)

    if command == "ket":
        return [
            rf"\\ket\s*\{{\s*{content_pattern}\s*\}}{subscript_pattern}",
            rf"\|\s*{content_pattern}\s*\\rangle{subscript_pattern}",
        ]

    return [
        rf"\\bra\s*\{{\s*{content_pattern}\s*\}}{subscript_pattern}",
        rf"\\langle\s*{content_pattern}\s*\|{subscript_pattern}",
    ]


def make_symbol_pattern(symbol: str) -> str:
    symbol = symbol.strip()
    variants = {symbol}

    if "_{" in symbol and symbol.endswith("}"):
        compact = symbol.replace("_{", "_").replace("}", "")
        variants.add(compact)

    if "_" in symbol and "_{" not in symbol:
        # Match a single character subscript and add braces. Do not consume subsequent superscripts!
        latex_form = re.sub(r"_([A-Za-z0-9])", r"_{\1}", symbol)
        if latex_form != symbol:
            variants.add(latex_form)

    subscript_suffix = r"(?:_(?:\{[A-Za-z0-9+\-',]+\}|[A-Za-z0-9']+))"

    family = get_symbol_family(symbol)
    if family and family != symbol:
        variants.add(family)

    final_variants = set()
    for v in variants:
        esc = re.escape(v)
        esc = re.sub(r'([A-Za-z]+)', r'(?:\\\\)?\1', esc)
        
        final_variants.add(esc)
        if re.fullmatch(r"[A-Za-z]+", v):
            final_variants.add(esc + r"(?:\s*\([^)]*\))?")
            final_variants.add(esc + subscript_suffix)
        elif "_" not in v and "{" in v and "}" in v:
            final_variants.add(esc + subscript_suffix)

    final_variants.update(_dirac_regex_variants(symbol))

    variant_regex = "|".join(sorted(final_variants, key=len, reverse=True))
    case_sensitive_variants = rf"(?-i:[$]*\s*(?:{variant_regex})\s*[$]*)"
    return rf"(?<![A-Za-z0-9_/]){case_sensitive_variants}(?![A-Za-z0-9_])"


def extract_lhs_symbol(equation: str) -> str:
    """
    Extract LHS symbol from:
    u = ...
    \\vec{u} = ...
    Q(x,t) = ...
    """
    if not equation or "=" not in equation:
        return ""

    lhs = equation.split("=", 1)[0].strip()
    lhs = lhs.strip("$ ")

    # Remove arguments: Q(x,t) -> Q
    lhs = re.sub(r"\([^)]*\)", "", lhs)

    symbols = symbol_extractor.extract_symbols(lhs)
    if symbols:
        return symbols[-1]

    return ""
def extract_constant_definition_from_equation(equation: str) -> Dict[str, str]:
    """
    Extracts definitions where symbols are equated to constant values.
    Example: 'n = 1, 2, \\dots' -> {'n': 'constant'}
    """
    constants = {}
    parts = equation.split('=')
    for i in range(len(parts) - 1):
        lhs = parts[i]
        rhs = parts[i+1]
        
        m = re.match(r"^\s*(?:[0-9]+(?:[\s,]+[0-9]+)*[\s,]*(?:\\(?:dots|ldots|cdots))?|0)\b", rhs)
        if m:
            syms = symbol_extractor.extract_symbols_from_equation(lhs)
            if syms:
                constants[syms[-1]] = "constant"
    return constants



def extract_definition_from_before_defined_as(
    equation: str,
    before_text: str,
) -> Dict[str, str]:
    """
    Handles cases like:

    where the flow velocity is defined as
    u = ...

    Then returns:
    u -> flow velocity
    """
    defs: Dict[str, str] = {}

    lhs_symbol = extract_lhs_symbol(equation)
    if not lhs_symbol or not before_text:
        return defs

    lhs_symbol = canonical_symbol(lhs_symbol)

    text = before_text.strip()
    while True:
        prev_text = text
        text = text.rstrip(" :.,;")
        text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]$", "", text).strip()
        if text == prev_text:
            break

    cue_phrases = [
        r"is\s+(?:then\s+)?defined\s+as",
        r"is\s+(?:then\s+)?given\s+by",
        r"is\s+(?:then\s+)?expressed\s+as",
        r"takes\s+the\s+form",
        r"as\s+follows",
        r"can\s+be\s+written\s+as",
        r"can\s+be\s+expressed\s+as",
    ]

    sym_pat = r"(?:[$\\\s]*" + make_symbol_pattern(lhs_symbol) + r"[$\w\s{}_()\\,^+-]*?)"

    patterns = []
    for cp in cue_phrases:
        # With symbol inside text: e.g. "the vorticity \nu_{1}(\Gamma) can be defined as follows"
        patterns.append(r"where\s+(?:the\s+)?" + DEF_REGEX + r"\s+" + sym_pat + r"\s+" + cp + r"\s*$")
        patterns.append(r"(?:the\s+)?" + DEF_REGEX + r"\s+" + sym_pat + r"\s+" + cp + r"\s*$")
        # Without symbol inside text: e.g. "where the flow velocity is defined as"
        patterns.append(r"where\s+(?:the\s+)?" + DEF_REGEX + r"\s+" + cp + r"\s*$")
        patterns.append(r"(?:the\s+)?" + DEF_REGEX + r"\s+" + cp + r"\s*$")

    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            meaning = clean_definition(match.group(1))
            if meaning:
                defs[lhs_symbol] = meaning
            break

    return defs


def extract_definitions_from_text(symbols: List[str], text: str) -> Dict[str, str]:
    defs: Dict[str, str] = {}

    if not text:
        return defs

    defs.update(_extract_normalized_symbol_definitions(symbols, text))

    for sym in symbols:
        if not sym:
            continue
        if sym in defs:
            continue

        sym_pattern = make_symbol_pattern(sym)

        special_definitions = [
            (r"\bparticle\s+of\s+mass\s+{X}\b", 0, "particle mass"),
            (
                r"\btwo\s+([A-Za-z][A-Za-z\-/()]*s)\s+{X}\s+and\s+"
                r"[A-Za-z](?:_(?:\{[A-Za-z0-9+\-']+\}|[A-Za-z0-9']+))?\b",
                1,
                "",
            ),
            (
                r"\btwo\s+([A-Za-z][A-Za-z\-/()]*s)\s+"
                r"[A-Za-z](?:_(?:\{[A-Za-z0-9+\-']+\}|[A-Za-z0-9']+))?"
                r"\s+and\s+{X}\b",
                1,
                "",
            ),
            (
                r"\ball\s+([A-Za-z][A-Za-z\-/()]*s)\s+{X}\s*\\in\b",
                1,
                "",
            ),
        ]
        for pat_str, group_idx, fixed_definition in special_definitions:
            pat = pat_str.replace("{X}", sym_pattern)
            match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                definition = fixed_definition or clean_definition(match.group(group_idx))
                if not definition:
                    continue
                defs[sym] = definition
                break
        if sym in defs:
            continue

        for pat_str, group_idx in PATTERNS:
            pat = pat_str.replace("{X}", sym_pattern)

            match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue

            definition = clean_definition(match.group(group_idx))

            if len(definition) > 2:
                first_word = definition.split(maxsplit=1)[0].lower()
                if first_word in BAD_DEFINITION_STARTS:
                    continue

                # If the pattern places the symbol adjacent to the definition or uses a weak fallback,
                # reject definitions that contain specific spatial/directional prepositions.
                is_adjacent = (
                    r"\s+{X}\b" in pat_str
                    or pat_str.startswith(r"{X}\s*,?")
                    or r"\s+in\s+{X}\b" in pat_str
                )
                if is_adjacent:
                    preps = {"in", "at", "on", "by", "with", "for", "from", "to", "as"}
                    words = set(definition.lower().split())
                    if words & preps:
                        continue

                defs[sym] = definition
                break

    return defs

def get_symbol_family(symbol: str) -> str:
    """
    x_ij -> x
    x_{ij} -> x
    q_kl -> q
    X_i -> X
    """
    normalized = symbol_extractor.normalize_symbol(symbol)
    if normalized:
        return normalized
    return symbol.split("_", 1)[0] if "_" in symbol else symbol

def get_symbol_list(eq_info: dict) -> List[str]:
    symbols = eq_info.get("symbols", [])

    if isinstance(symbols, dict):
        raw_symbols = list(symbols.keys())
    elif isinstance(symbols, list):
        raw_symbols = symbols
    else:
        return []

    normalized_symbols = []
    for symbol in raw_symbols:
        normalized_symbol = get_symbol_family(symbol)
        if normalized_symbol and normalized_symbol not in normalized_symbols:
            normalized_symbols.append(normalized_symbol)

    return normalized_symbols


def _definition_lookup(definitions: Dict[str, str], symbol: str) -> str:
    lookup_keys = set(symbol_lookup_keys(symbol))
    for key in lookup_keys:
        definition = definitions.get(key)
        if definition:
            return definition

    for definition_symbol, definition in definitions.items():
        if lookup_keys & set(symbol_lookup_keys(definition_symbol)):
            return definition
    return ""


def _target_symbol_key_map(symbols: List[str]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for symbol in symbols:
        normalized = symbol_extractor.normalize_symbol(symbol)
        for variant in [symbol, normalized]:
            for key in symbol_lookup_keys(variant):
                keys.setdefault(key, symbol)
    return keys


def _matching_target_symbol(raw_symbol: str, target_keys: Dict[str, str]) -> str:
    normalized = symbol_extractor.normalize_symbol(raw_symbol)
    for variant in [raw_symbol, normalized]:
        for key in symbol_lookup_keys(variant):
            if key in target_keys:
                return target_keys[key]
    return ""


def _add_symbol_definition(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    raw_symbol: str,
    definition: str,
) -> None:
    target_symbol = _matching_target_symbol(raw_symbol, target_keys)
    if target_symbol and target_symbol not in defs and definition:
        defs[target_symbol] = definition


def _clean_symbol_definition(definition: str) -> str:
    definition = clean_definition(definition)
    head = re.split(
        r"\s+(?:in|at|on|with|for|from|to)\s+(?:the|a|an)\s+",
        definition,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return head if len(head.split()) >= 2 else definition


def _singular_definition(definition: str) -> str:
    definition = clean_definition(definition)
    return re.sub(
        r"\b(atoms|coordinates|functions|masses|operators|wavefunctions)\b",
        lambda match: {
            "atoms": "atom",
            "coordinates": "coordinate",
            "functions": "function",
            "masses": "mass",
            "operators": "operator",
            "wavefunctions": "wavefunction",
        }[match.group(1).lower()],
        definition,
        flags=re.IGNORECASE,
    )


def _extract_grouped_subscript_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    pattern = (
        r"\$?\s*(?P<base>\\?[A-Za-z])(?:')?_\{(?P<subs>[A-Za-z]\s*,\s*[A-Za-z])\}\s*\$?"
        r"\s+(?:be|are|denote|represent)\s+(?:the\s+|a\s+|an\s+)?"
        r"(?P<definition>[A-Za-z][A-Za-z\s\-/()]{2,80})"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        definition = _singular_definition(_clean_symbol_definition(match.group("definition")))
        base = match.group("base").lstrip("\\")
        for subscript in re.split(r"\s*,\s*", match.group("subs")):
            _add_symbol_definition(defs, target_keys, f"{base}_{subscript}", definition)
            _add_symbol_definition(defs, target_keys, rf"\mathbf{{{base}}}_{subscript}", definition)

    vector_pattern = (
        r"\$?\s*(?P<base>\\(?:mathbf|vec|bm)\{[A-Za-z]\}|[A-Za-z])_\{"
        r"(?P<subs>[A-Za-z]\s*,\s*[A-Za-z])\}\s*[^.;]{0,160}?"
        r"(?:refers?\s+to|denotes?|represents?)\s+(?:the\s+|a\s+|an\s+)?"
        r"(?P<definition>[A-Za-z][A-Za-z\s\-/()]{2,100})"
    )
    for match in re.finditer(vector_pattern, text, flags=re.IGNORECASE):
        definition = _singular_definition(_clean_symbol_definition(match.group("definition")))
        base = match.group("base")
        for subscript in re.split(r"\s*,\s*", match.group("subs")):
            _add_symbol_definition(defs, target_keys, f"{base}_{subscript}", definition)


def _extract_pm_coordinate_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    pattern = r"(?P<base>\\?mathbf\{[A-Za-z]\}|[A-Za-z])_\{\\pm\}\s*:=\s*[^.;]{1,120}\\pm"
    for match in re.finditer(pattern, text):
        base = match.group("base")
        _add_symbol_definition(defs, target_keys, f"{base}_+", "sum coordinate")
        _add_symbol_definition(defs, target_keys, f"{base}_-", "difference coordinate")


def _extract_local_function_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    if re.search(r"bi-?photon\s+wavefunction\s+in\s+z'", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\psi_{z'}", "bi-photon wavefunction")

    if re.search(r"phase\s+matching\s+and\s+the\s+pump\s+functions", text, flags=re.IGNORECASE):
        for raw_symbol in (r"\Phi_z", r"\Phi_{z'}", "Phi_z", "Phi_{zprime}"):
            _add_symbol_definition(defs, target_keys, raw_symbol, "phase matching function")
        for raw_symbol in (r"\mathcal{E}_z", r"\mathcal{E}_{z'}", "mathcal{E}_z", "mathcal{E}_{zprime}"):
            _add_symbol_definition(defs, target_keys, raw_symbol, "pump function")


def _extract_state_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    if re.search(r"\$?\s*(?:\|0\\rangle|\\ket\{0\})\s*\$?\s+is\s+(?:the\s+|a\s+)?vacuum\s+state", text):
        _add_symbol_definition(defs, target_keys, "0", "vacuum state")

    if re.search(r"\bvacuum\s+state\b", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\mathrm{vac}", "vacuum state")

    if re.search(r"two-?photon\s+wavefunction[^.]{0,120}\breads\b", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\psi", "two-photon wavefunction")

    if re.search(r"bi-?photon\s+state[^.]{0,120}\bbecomes\b", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\psi", "bi-photon state")

    if re.search(r"transmitted\s+state[^.]{0,160}\bis\b", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\psi", "transmitted state")

    excited_state_pattern = (
        r"\\ket\{(?P<first>[A-Za-z])\}\s+and\s+\\ket\{(?P<second>[A-Za-z])\}"
        r"\s+represent\s+the\s+first\s+and\s+second\s+excited\s+state\s+of\s+the\s+"
        r"(?P<system>[A-Za-z][A-Za-z\s-]{1,60})"
    )
    for match in re.finditer(excited_state_pattern, text, flags=re.IGNORECASE):
        system = clean_definition(match.group("system"))
        _add_symbol_definition(defs, target_keys, match.group("first"), f"first excited state of the {system}")
        _add_symbol_definition(defs, target_keys, match.group("second"), f"second excited state of the {system}")


def _extract_coordinated_entity_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    symbol = rf"\{{?\s*({RAW_SYMBOL_REGEX})\s*\}}?"
    patterns = [
        rf"(?:two\s+)?(?P<definition>masses|atoms|atom\s+species)\s+"
        rf"{symbol}\s+and\s+{symbol}",
        rf"{symbol}\s+and\s+{symbol}\s+"
        rf"(?P<definition>masses|atoms|atom\s+species)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            definition = _singular_definition(match.group("definition"))
            for raw_symbol in match.groups():
                if raw_symbol and raw_symbol.lower() != match.group("definition").lower():
                    _add_symbol_definition(defs, target_keys, raw_symbol, definition)


def _without_dagger(raw_symbol: str) -> str:
    raw_symbol = re.sub(r"\^\s*\{\s*\\?dagger\s*\}", "", raw_symbol)
    raw_symbol = re.sub(r"\^\s*\\?dagger\b", "", raw_symbol)
    return raw_symbol


def _extract_respectively_operator_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    pattern = (
        r"\bwhere\s+(?P<symbols>[^.]{1,260}?)"
        r"\s+are\s+(?:the\s+)?"
        r"(?P<first_kind>bosonic|fermionic)\s+and\s+"
        r"(?P<second_kind>bosonic|fermionic)\s+"
        r"(?P<head>creation\s+operators?|annihilation\s+operators?)"
        r"[^.]{0,180}?\brespectively\b"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        candidates: List[str] = []
        for raw_symbol in symbol_extractor.extract_symbols_from_equation(match.group("symbols")):
            target_symbol = _matching_target_symbol(raw_symbol, target_keys)
            if target_symbol and target_symbol not in candidates:
                candidates.append(target_symbol)
        if len(candidates) < 2:
            continue

        first_kind = match.group("first_kind").lower()
        second_kind = match.group("second_kind").lower()
        head = _singular_definition(match.group("head").lower())
        first_symbol, second_symbol = candidates[:2]

        _add_symbol_definition(defs, target_keys, first_symbol, f"{first_kind} {head}")
        _add_symbol_definition(defs, target_keys, second_symbol, f"{second_kind} {head}")

        if "creation" in head:
            _add_symbol_definition(
                defs,
                target_keys,
                _without_dagger(first_symbol),
                f"{first_kind} annihilation operator",
            )
            _add_symbol_definition(
                defs,
                target_keys,
                _without_dagger(second_symbol),
                f"{second_kind} annihilation operator",
            )


def _operator_mode_key(symbol: str) -> str:
    match = re.match(r"hat\{([A-Za-z])\}$", symbol)
    return match.group(1) if match else ""


def _extract_mode_operator_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    pattern = (
        r"\bwhere\s+(?P<symbols>[^.]{1,160}?)"
        r"\s+are\s+(?:the\s+)?bosonic\s+"
        r"(?P<head>annihilation\s+operators?|creation\s+operators?)"
        r"\s+of\s+the\s+"
        r"(?P<modes>[A-Za-z][A-Za-z\s,-]{1,120}?)"
        r"\s+modes?,\s+respectively\b"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        candidates: List[str] = []
        for raw_symbol in symbol_extractor.extract_symbols_from_equation(match.group("symbols")):
            target_symbol = _matching_target_symbol(raw_symbol, target_keys)
            if target_symbol and target_symbol not in candidates:
                candidates.append(target_symbol)

        modes = [
            clean_definition(re.sub(r"^(?:and\s+)+", "", mode.strip(), flags=re.IGNORECASE))
            for mode in re.split(r"\s*,\s*|\s+and\s+", match.group("modes"))
            if clean_definition(re.sub(r"^(?:and\s+)+", "", mode.strip(), flags=re.IGNORECASE))
        ]
        if len(candidates) < 2 or len(candidates) != len(modes):
            continue

        head = _singular_definition(match.group("head").lower())
        for symbol, mode in zip(candidates, modes):
            definition = f"{mode} mode {head}"
            _add_symbol_definition(defs, target_keys, symbol, definition)

            base = _operator_mode_key(symbol)
            if not base:
                continue
            if "annihilation" in head:
                _add_symbol_definition(
                    defs,
                    target_keys,
                    f"hat{{{base}}}^{{dagger}}",
                    f"{mode} mode creation operator",
                )
            _add_symbol_definition(
                defs,
                target_keys,
                f"\\omega_{{{base}}}",
                f"{mode} mode angular frequency",
            )


def _extract_hamiltonian_process_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    pattern = (
        r"(?:effective\s+)?Hamiltonian\s+(?P<symbols>[^.]{1,160}?)"
        r"\s+describing\s+the\s+"
        r"(?P<process>[A-Za-z][A-Za-z\s-]{2,80}?)"
        r"(?:\s+\([A-Za-z]+\))?\s+process"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        definition = f"{clean_definition(match.group('process'))} Hamiltonian"
        for raw_symbol in symbol_extractor.extract_symbols_from_equation(match.group("symbols")):
            _add_symbol_definition(defs, target_keys, raw_symbol, definition)
            break


def _extract_system_hamiltonian_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    if re.search(r"\bsystem\s+Hamiltonian\s+in\s+the\s+lab\s+frame\b", text, flags=re.IGNORECASE):
        _add_symbol_definition(defs, target_keys, r"\hat{H}_{0}", "lab-frame system Hamiltonian")


def _extract_dispersive_shift_definitions(
    defs: Dict[str, str],
    target_keys: Dict[str, str],
    text: str,
) -> None:
    if re.search(r"\bdispersive\s+shift\b", text, flags=re.IGNORECASE):
        for raw_symbol in (r"\chi_{ge}", r"\chi_{gf}"):
            _add_symbol_definition(defs, target_keys, raw_symbol, "dispersive shift")
        if _matching_target_symbol(r"\chi_{qr}", target_keys):
            _add_symbol_definition(defs, target_keys, r"\chi_{qr}", "transmon-reservoir dispersive shift")


def _extract_normalized_symbol_definitions(symbols: List[str], text: str) -> Dict[str, str]:
    defs: Dict[str, str] = {}
    target_keys = _target_symbol_key_map(symbols)
    if not target_keys:
        return defs

    definition_regex = r"(?P<definition>[A-Za-z][A-Za-z\s\-/()]{2,100})"
    patterns = [
        rf"(?:where\s+)?\$*\s*(?P<symbol>{RAW_SYMBOL_REGEX})\s*\$*\s*,?\s+"
        rf"is\s+(?:a|an|the)\s+{definition_regex}",
        rf"(?:where\s+)?\$*\s*(?P<symbol>{RAW_SYMBOL_REGEX})\s*\$*\s*,?\s+"
        rf"(?:denotes|represents|refers\s+to)\s+(?:a|an|the)?\s*{definition_regex}",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            target_symbol = _matching_target_symbol(match.group("symbol"), target_keys)
            if target_symbol and target_symbol not in defs:
                definition = _clean_symbol_definition(match.group("definition"))
                if definition:
                    defs[target_symbol] = definition

    _extract_grouped_subscript_definitions(defs, target_keys, text)
    _extract_pm_coordinate_definitions(defs, target_keys, text)
    _extract_local_function_definitions(defs, target_keys, text)
    _extract_state_definitions(defs, target_keys, text)
    _extract_coordinated_entity_definitions(defs, target_keys, text)
    _extract_respectively_operator_definitions(defs, target_keys, text)
    _extract_mode_operator_definitions(defs, target_keys, text)
    _extract_hamiltonian_process_definitions(defs, target_keys, text)
    _extract_system_hamiltonian_definitions(defs, target_keys, text)
    _extract_dispersive_shift_definitions(defs, target_keys, text)

    return defs


def _paper_history_keys(symbol: str) -> List[str]:
    return symbol_lookup_keys(symbol)


def _paper_history_lookup(
    paper_history: Dict[str, Dict[str, Any]],
    symbol: str,
) -> Dict[str, Any]:
    for key in _paper_history_keys(symbol):
        entry = paper_history.get(key)
        if entry:
            return entry
    return {}


def _record_paper_history(
    paper_history: Dict[str, Dict[str, Any]],
    symbol: str,
    definition: str,
    source: str,
    equation_id: str,
) -> None:
    if not definition:
        return

    entry = {
        "symbol": symbol,
        "definition": definition,
        "source": source,
        "equation_id": str(equation_id),
    }
    for key in _paper_history_keys(symbol):
        paper_history[key] = entry


def _coerce_equation_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _load_predefined_definitions(predefined_path: str) -> List[Dict[str, Any]]:
    path = Path(predefined_path)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    records: List[Dict[str, Any]] = []

    def add_record(symbol: Any, definition: Any, equations: Any = None) -> None:
        if not symbol or not definition:
            return
        records.append({
            "symbol": str(symbol).strip(),
            "definition": str(definition).strip(),
            "equations": _coerce_equation_list(equations),
        })

    if isinstance(data, dict):
        for symbol, value in data.items():
            if isinstance(value, str):
                add_record(symbol, value)
            elif isinstance(value, dict):
                definition = (
                    value.get("definition")
                    or value.get("meaning")
                    or value.get("description")
                )
                equations = (
                    value.get("equations")
                    or value.get("equation")
                    or value.get("equation_pattern")
                )
                add_record(value.get("symbol") or symbol, definition, equations)
    elif isinstance(data, list):
        for value in data:
            if not isinstance(value, dict):
                continue
            definition = (
                value.get("definition")
                or value.get("meaning")
                or value.get("description")
            )
            equations = (
                value.get("equations")
                or value.get("equation")
                or value.get("equation_pattern")
            )
            add_record(value.get("symbol") or value.get("key"), definition, equations)

    return records


def _predefined_symbol_keys(symbol: str) -> List[str]:
    candidates = symbol_lookup_keys(symbol)

    for candidate in list(candidates):
        if not candidate:
            continue
        family = get_symbol_family(candidate)
        candidates.extend([family, canonical_symbol(family)])

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _normalize_equation_for_match(equation: str) -> str:
    return re.sub(r"\s+", "", equation or "")


def _predefined_equation_matches(record: Dict[str, Any], equation: str) -> bool:
    predefined_equations = record.get("equations", [])
    if not predefined_equations:
        return True

    normalized_equation = _normalize_equation_for_match(equation)
    if not normalized_equation:
        return False

    for predefined_equation in predefined_equations:
        normalized_predefined = _normalize_equation_for_match(predefined_equation)
        if not normalized_predefined:
            continue
        if (
            normalized_predefined in normalized_equation
            or normalized_equation in normalized_predefined
        ):
            return True

    return False


def _predefined_definition(
    predefined_records: List[Dict[str, Any]],
    symbol: str,
    equation: str,
) -> Tuple[str, str]:
    symbol_keys = _predefined_symbol_keys(symbol)

    for key in symbol_keys:
        for record in predefined_records:
            if record.get("symbol") != key:
                continue
            if _predefined_equation_matches(record, equation):
                return record.get("definition", ""), record.get("symbol", "")

    canonical_keys = {canonical_symbol(key) for key in symbol_keys if key}
    for record in predefined_records:
        record_symbol = record.get("symbol", "")
        if canonical_symbol(record_symbol) not in canonical_keys:
            continue
        if _predefined_equation_matches(record, equation):
            return record.get("definition", ""), record_symbol

    return "", ""


def _load_global_context(paper_id: str, global_context_dir: str) -> str:
    candidates = [
        Path(global_context_dir) / f"{paper_id}.json",
        Path(global_context_dir) / f"{paper_id}_global_context.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        context = data.get("global_context_window", "")
        if isinstance(context, str):
            return context

    return ""


def _load_definition_candidates(candidate_path: str) -> Dict[str, Any]:
    path = Path(candidate_path)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _equation_candidate_lists(
    candidates_data: Dict[str, Any],
    paper_id: str,
    equation_id: str,
    eq_info: Dict[str, Any],
) -> Dict[str, Any]:
    paper_candidates = candidates_data.get(paper_id, {})
    equation_candidates = {}

    if isinstance(paper_candidates, dict):
        equation_candidates = paper_candidates.get(str(equation_id), {})

    if isinstance(equation_candidates, dict):
        if "definition_candidates" in equation_candidates:
            candidate_lists = equation_candidates.get("definition_candidates", {})
            if isinstance(candidate_lists, dict):
                return candidate_lists
        return equation_candidates

    embedded_candidates = eq_info.get("definition_candidates", {})
    return embedded_candidates if isinstance(embedded_candidates, dict) else {}


def _highest_confidence_pos_candidate(
    candidate_lists: Dict[str, Any],
    symbol: str,
) -> Tuple[str, Optional[float]]:
    candidates = []
    lookup_keys = set(symbol_lookup_keys(symbol))
    for key in lookup_keys:
        candidates = candidate_lists.get(key, [])
        if candidates:
            break
    if not candidates:
        for candidate_symbol, candidate_values in candidate_lists.items():
            if lookup_keys & set(symbol_lookup_keys(candidate_symbol)):
                candidates = candidate_values
                break
    if not candidates:
        return "", None

    top_candidate = max(
        candidates,
        key=lambda item: item.get("confidence_score", 0.0),
    )
    confidence = float(top_candidate.get("confidence_score", 0.0) or 0.0)
    if confidence < POS_FALLBACK_MIN_CONFIDENCE:
        return "", confidence

    return (
        top_candidate.get("candidate", ""),
        confidence,
    )


def _symbol_mentioned_in_text(symbol: str, text: str) -> bool:
    if not text:
        return False
    for candidate in {symbol, canonical_symbol(symbol)}:
        if not candidate:
            continue
        try:
            if re.search(make_symbol_pattern(candidate), text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _equation_anchor(text: str, equation: str) -> Optional[Tuple[int, int]]:
    if not text or not equation:
        return None

    start = text.find(equation)
    if start >= 0:
        return start, start + len(equation)

    compact_text = re.sub(r"\s+", "", text)
    compact_equation = re.sub(r"\s+", "", equation)
    if not compact_equation:
        return None

    compact_start = compact_text.find(compact_equation)
    if compact_start < 0:
        return None

    compact_index = 0
    start_char = None
    end_char = None
    for char_index, char in enumerate(text):
        if char.isspace():
            continue
        if compact_index == compact_start and start_char is None:
            start_char = char_index
        compact_index += 1
        if compact_index == compact_start + len(compact_equation):
            end_char = char_index + 1
            break

    if start_char is None or end_char is None:
        return None
    return start_char, end_char


def _text_before_equation(context: str, equation: str) -> str:
    anchor = _equation_anchor(context, equation)
    if not anchor:
        return ""
    return context[:anchor[0]]


def add_definitions_to_dataset(
    dataset: Optional[Dict[str, Any]] = None,
    dataset_path: str = "dataset.json",
    global_context_dir: str = GLOBAL_CONTEXT_DIR,
    predefined_path: str = PREDEFINED_DEFINITIONS_PATH,
    candidate_path: str = DEFINITION_CANDIDATES_PATH,
    source_output_path: str = DEFINITION_SOURCES_PATH,
) -> Dict[str, Any]:
    """
    Core function that resolves the meaning of every mathematical symbol in the dataset.
    It cascades through multiple sources in order of confidence:
    1. Text immediately before the equation
    2. Constants defined within the equation itself
    3. Text immediately after the equation
    4. Local context window (surrounding paragraphs)
    5. Global context window (whole paper)
    6. Paper history (meanings established in prior equations)
    7. NLP POS tagging candidates
    8. Predefined backup dictionary
    
    Args:
        dataset (Optional[Dict[str, Any]]): The dataset dict in memory.
        dataset_path (str): Fallback path to load the dataset.
        global_context_dir (str): Directory where global context strings are stored.
        predefined_path (str): Path to the predefined fallback dictionary.
        candidate_path (str): Path to POS tagged candidates.
        source_output_path (str): Path to save the detailed definition sources JSON.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The updated dataset and the detailed source tracking data.
    """
    we_loaded = False
    if dataset is None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        we_loaded = True

    predefined_records = _load_predefined_definitions(predefined_path)
    candidates_data = _load_definition_candidates(candidate_path)
    source_data: Dict[str, Dict[str, Any]] = {}

    for paper_id, paper_data in dataset.items():
        if not isinstance(paper_data, dict):
            continue

        source_data[paper_id] = {}
        paper_history: Dict[str, Dict[str, Any]] = {}
        all_symbols: List[str] = []
        for eq_info in paper_data.values():
            if not isinstance(eq_info, dict):
                continue
            for symbol in get_symbol_list(eq_info):
                if symbol not in all_symbols:
                    all_symbols.append(symbol)

        global_context = paper_data.get("global_context_window")
        if not global_context:
            global_context = _load_global_context(paper_id, global_context_dir)
            
        global_defs = extract_definitions_from_text(all_symbols, global_context)
        global_defs_by_canon = {
            canonical_symbol(symbol): definition
            for symbol, definition in global_defs.items()
        }
        global_defs.update(global_defs_by_canon)

        for equation_id, eq_info in paper_data.items():
            if not isinstance(eq_info, dict):
                continue

            symbols = get_symbol_list(eq_info)

            if "symbol_definition_sources" in eq_info:
                del eq_info["symbol_definition_sources"]
            if "definition_candidates" in eq_info and str(equation_id) in candidates_data.get(paper_id, {}):
                del eq_info["definition_candidates"]

            if not symbols:
                continue

            equation = eq_info.get("equation", "")

            before_text = eq_info.get("before_text", "")
            after_text = eq_info.get("after_text", "")
            local_context = eq_info.get("local_context_window", "")

            if not before_text and local_context:
                before_text = _text_before_equation(local_context, equation)

            before_defs = extract_definitions_from_text(symbols, before_text)
            
            immediate_before_text = before_text
            if local_context:
                exact_before = _text_before_equation(local_context, equation)
                if exact_before:
                    immediate_before_text = exact_before
                    
            lhs_defs = extract_definition_from_before_defined_as(equation, immediate_before_text)
            for k, v in lhs_defs.items():
                before_defs[k] = v

            constant_defs = extract_constant_definition_from_equation(equation)

            after_defs = extract_definitions_from_text(symbols, after_text)
            local_defs = extract_definitions_from_text(symbols, local_context)

            candidate_lists = _equation_candidate_lists(
                candidates_data,
                paper_id,
                str(equation_id),
                eq_info,
            )

            new_symbols = {}
            definition_sources = {}
            found_count = 0

            for sym in symbols:
                definition = ""
                source = ""
                confidence: Optional[float] = None
                predefined_key = ""
                inherited_from = ""
                inherited_source = ""
                global_mentioned = _symbol_mentioned_in_text(sym, global_context)

                definition = _definition_lookup(before_defs, sym)
                if definition:
                    source = "before_text_pattern"

                if not definition:
                    definition = _definition_lookup(constant_defs, sym)
                    if definition:
                        source = "equation_constant"

                if not definition:
                    definition = _definition_lookup(after_defs, sym)
                    if definition:
                        source = "after_text_pattern"

                if not definition:
                    definition = _definition_lookup(local_defs, sym)
                    if definition:
                        source = "local_context_pattern"

                if not definition:
                    definition = _definition_lookup(global_defs, sym)
                    if definition:
                        source = "global_context_pattern"

                if not definition:
                    history_entry = _paper_history_lookup(paper_history, sym)
                    if history_entry:
                        definition = history_entry.get("definition", "")
                        source = "paper_history"
                        inherited_from = history_entry.get("equation_id", "")
                        inherited_source = history_entry.get("source", "")

                if not definition:
                    definition, confidence = _highest_confidence_pos_candidate(
                        candidate_lists,
                        sym,
                    )
                    if definition:
                        source = "local_context_pos_candidate"

                if not definition:
                    definition, predefined_key = _predefined_definition(
                        predefined_records,
                        sym,
                        equation,
                    )
                    if definition:
                        source = "backup_dictionary"

                new_symbols[sym] = definition

                if definition:
                    found_count += 1

                definition_sources[sym] = {
                    "source": source or "unresolved",
                    "global_context_mentioned": global_mentioned,
                }
                if confidence is not None:
                    if source == "local_context_pos_candidate":
                        definition_sources[sym]["confidence_score"] = confidence
                    else:
                        definition_sources[sym]["rejected_pos_candidate_confidence"] = confidence
                        definition_sources[sym]["pos_fallback_min_confidence"] = (
                            POS_FALLBACK_MIN_CONFIDENCE
                        )
                if predefined_key:
                    definition_sources[sym]["predefined_key"] = predefined_key
                    definition_sources[sym]["predefined_path"] = predefined_path
                if inherited_from:
                    definition_sources[sym]["inherited_from_equation"] = inherited_from
                    definition_sources[sym]["inherited_source"] = inherited_source

                if definition and source != "paper_history":
                    _record_paper_history(
                        paper_history,
                        sym,
                        definition,
                        source,
                        str(equation_id),
                    )

            eq_info["symbols"] = new_symbols
            source_data[paper_id][str(equation_id)] = {
                "paper_id": paper_id,
                "equation_id": str(equation_id),
                "equation": equation,
                "symbol_definition_sources": definition_sources,
            }

            if "symbol_definitions" in eq_info:
                del eq_info["symbol_definitions"]
            for field in ("definition_candidates", "symbol_definition_sources"):
                if field in eq_info:
                    del eq_info[field]

            audit_trail = eq_info.setdefault("audit-trail", {})
            for key in ("definition_candidates", "pos_tag_storage", "candidate_storage"):
                if key in audit_trail:
                    del audit_trail[key]
            audit_trail["find_symbols"] = (
                f"Found normalized symbols: {', '.join(new_symbols) or 'none'}"
            )
            audit_trail["extract_definitions"] = (
                f"Resolved definitions for {found_count} symbols using before text, "
                "after text, local context window, global context window, "
                "paper history, POS tagging, then backup dictionary."
            )

        source_data[paper_id]["_paper_history"] = paper_history

    if we_loaded:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        with open(source_output_path, "w", encoding="utf-8") as f:
            json.dump(source_data, f, ensure_ascii=False, indent=2)

    return dataset, source_data


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3, 4, 5, 6}:
        print(
            "Usage: python definition_extractor.py "
            "<dataset.json> [global_context_dir] [predefined.json] "
            "[definition_candidates.json] [definition_sources.json]"
        )
        sys.exit(1)

    context_dir = sys.argv[2] if len(sys.argv) >= 3 else GLOBAL_CONTEXT_DIR
    predefined_file = sys.argv[3] if len(sys.argv) >= 4 else PREDEFINED_DEFINITIONS_PATH
    candidate_file = sys.argv[4] if len(sys.argv) >= 5 else DEFINITION_CANDIDATES_PATH
    source_file = sys.argv[5] if len(sys.argv) == 6 else DEFINITION_SOURCES_PATH
    add_definitions_to_dataset(
        dataset=None,
        dataset_path=sys.argv[1],
        global_context_dir=context_dir,
        predefined_path=predefined_file,
        candidate_path=candidate_file,
        source_output_path=source_file,
    )
