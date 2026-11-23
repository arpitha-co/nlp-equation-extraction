"""
Build equation-level context records for whole-equation meaning extraction.

This does not modify dataset.json. It writes a sidecar JSON file where each
equation is masked as "equation N" and paired with heading/subheading plus the
existing before/after context fields.
"""

import json
import re
from pathlib import Path
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from local_context_window import (
    CACHE_DIRS,
    _clean_context_text,
    _find_html_path,
    _find_markdown_path,
    _html_local_contexts,
    _html_equation_number,
    _html_node_text,
    _markdown_local_contexts,
    _normalize_match_text,
    _sentence_chunks,
    _text_anchor,
)


DATASET_PATH = "dataset.json"
OUTPUT_PATH = "equation_contexts.json"
SENTENCE_SPLITS_OUTPUT_PATH = "equation_sentence_splits.json"
SPACY_MODEL = "en_core_web_sm"
PRIMARY_AFTER_WORDS = 10
LOW_VALUE_CANDIDATES = {"as follows", "respectively", "where", "which", "that", "this", "what", "who", "whom", "whose", "it", "they", "them", "these", "those", "following structure", "following form", "following equation", "following", "namely", "specifically", "thus", "hence", "therefore", "here", "there", "eq", "eqs", "equation", "equations"}
MATH_TOKEN_RE = re.compile(r"(\\[A-Za-z]+|[_^{}$=<>≤≥±∇∂]|[A-Za-z]+_\{)")
FORMULA_CUE_VERB_LEMMAS = {"follow"}
ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth"]
ACTION_ROOT_DEPS = {"dobj", "attr", "nsubj", "nsubjpass"}
GENERIC_BRIDGE_NOUNS = {"part", "portion", "component"}
EQUATION_NAME_TERMS = {
    "equation",
    "equations",
    "hamiltonian",
    "potential",
    "image",
    "condition",
    "constraint",
    "relation",
    "law",
    "identity",
    "formula",
    "expression",
    "approximation",
    "inequality",
    "index",
}
USEFUL_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP"}
DEPENDENCY_BONUS = {
    "nsubj": 14.0,
    "nsubjpass": 14.0,
    "attr": 12.0,
    "dobj": 10.0,
    "pobj": 8.0,
    "appos": 8.0,
    "ROOT": 6.0,
}


def _equation_mask(equation_id: str) -> str:
    return f"EQ_{equation_id}"


def _mask_text(text: str, equation: str, mask: str) -> str:
    if not text or not equation:
        return text or ""

    anchor = _text_anchor(text, equation)
    if anchor:
        return _clean_context_text(f"{text[:anchor[0]]} {mask} {text[anchor[1]:]}")

    return text.replace(equation, mask)


def _mask_all_equations(text: str, paper_data: Dict[str, Any]) -> str:
    masked = text or ""
    equations = []
    for equation_id, eq_info in paper_data.items():
        if not isinstance(eq_info, dict):
            continue
        equation = eq_info.get("equation", "")
        if equation:
            equations.append((str(equation_id), equation))

    for equation_id, equation in sorted(equations, key=lambda item: len(item[1]), reverse=True):
        masked = _mask_text(masked, equation, _equation_mask(equation_id))
    return masked


def _blank_headings() -> Dict[str, str]:
    return {
        "heading": "",
        "subheading": "",
        "subsubheading": "",
    }


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\S+", text or "")


def _counts_as_context_word(token: str) -> bool:
    if re.fullmatch(r"\(?\d+(?:\.\d+)?[a-z]?\)?", token):
        return False
    return bool(re.search(r"[A-Za-z0-9]", token))


def _side_window(tokens: List[str], start: int, step: int, word_limit: int) -> List[str]:
    selected = []
    word_count = 0
    index = start
    while 0 <= index < len(tokens) and word_count < word_limit:
        token = tokens[index]
        selected.append(token)
        if _counts_as_context_word(token):
            word_count += 1
        index += step
    if step < 0:
        selected.reverse()
    return selected


def _split_letter_marker_sentences(sentences: List[str]) -> List[str]:
    split_sentences = []
    for sentence in sentences:
        parts = re.split(r"\s+(?=\([a-z]\)\s+)", sentence)
        split_sentences.extend(part.strip() for part in parts if part.strip())
    return split_sentences


def _strip_numeric_citations(text: str) -> str:
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", " ", text or "")
    text = re.sub(r"\(\s*Fig\.?\s*[^)]*\)", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _move_colon_prefix_to_before(before: str, equation_sentence: str, mask: str) -> tuple[str, str, bool]:
    mask_index = equation_sentence.find(mask)
    if mask_index < 0:
        return before, equation_sentence, False

    colon_index = equation_sentence.rfind(":", 0, mask_index)
    if colon_index < 0:
        return before, equation_sentence, False

    prefix = equation_sentence[:colon_index].strip()
    suffix = equation_sentence[colon_index + 1:].strip()
    if not prefix or not suffix:
        return before, equation_sentence, False

    before = " ".join(part for part in [before, prefix] if part)
    return before, suffix, True


def _split_inline_after_equation(sentence: str, mask: str) -> tuple[str, str]:
    mask_index = sentence.find(mask)
    if mask_index < 0:
        return sentence, ""

    keyword_match = re.search(
        r"\b(where|with|when|if|such|for)\b",
        sentence[mask_index + len(mask):],
        flags=re.IGNORECASE,
    )
    if not keyword_match:
        return sentence, ""

    split_index = mask_index + len(mask) + keyword_match.start()
    return sentence[:split_index].strip(), sentence[split_index:].strip()


def _sentence_plus_after_window(text: str, mask: str) -> Dict[str, Any]:
    sentences = _split_letter_marker_sentences(_sentence_chunks(text))
    sentence_index = next(
        (index for index, sentence in enumerate(sentences) if mask in sentence),
        -1,
    )
    if sentence_index < 0:
        return {
            "primary_before_text": "",
            "primary_equation_sentence": "",
            "primary_after_text": "",
            "primary_context_window": "",
        }

    before = ""
    for idx in range(sentence_index - 1, -1, -1):
        cand = sentences[idx]
        cleaned = re.sub(r"\bEQ_[A-Za-z0-9.]+\b", "", cand)
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = re.sub(r"[0-9.,;:\s\-]", "", cleaned)
        if cleaned:
            before = cand
            break
    equation_sentence = sentences[sentence_index]
    before, equation_sentence, moved_colon_prefix = _move_colon_prefix_to_before(before, equation_sentence, mask)
    inline_after = ""
    if moved_colon_prefix:
        equation_sentence, inline_after = _split_inline_after_equation(equation_sentence, mask)

    before = _strip_numeric_citations(before)
    equation_sentence = _strip_numeric_citations(equation_sentence)
    after_source = " ".join(part for part in [inline_after, *sentences[sentence_index + 1:]] if part)
    after_source = _strip_numeric_citations(after_source)
    after_tokens = _side_window(_word_tokens(after_source), 0, 1, PRIMARY_AFTER_WORDS)
    after = " ".join(after_tokens)
    primary_context = " ".join(part for part in [before, equation_sentence, after] if part)
    return {
        "primary_before_text": before,
        "primary_equation_sentence": equation_sentence,
        "primary_after_text": after,
        "primary_context_window": primary_context,
    }


def _load_dependency_parser():
    import spacy

    return spacy.load(SPACY_MODEL)


def _formula_token_parts(token: str) -> tuple[str, str, str]:
    match = re.match(r"^([(\"']*)(.*?)([.,;:!?)]*)$", token)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def _is_formula_token(token: str) -> bool:
    _, core, _ = _formula_token_parts(token)
    return bool(core and MATH_TOKEN_RE.search(core))


def _normalize_formula_tokens(text: str) -> str:
    normalized = []
    previous_was_math = False

    for token in _word_tokens(text):
        if re.fullmatch(r"(?:\(?\d+(?:\.\d+)?[a-z]?\)?|\([a-z]\))[.,;:]?", token):
            continue

        mask_match = re.fullmatch(r"(EQ_[A-Za-z0-9.]+)([.,;:]?)", token)
        if mask_match:
            normalized.append(mask_match.group(1))
            if mask_match.group(2):
                normalized.append(mask_match.group(2))
            previous_was_math = False
            continue

        if _is_formula_token(token):
            _, _, trailing = _formula_token_parts(token)
            if not previous_was_math:
                normalized.append("MATH")
            if trailing:
                normalized.append(trailing)
            previous_was_math = True
            continue

        normalized.append(token)
        previous_was_math = False

    return " ".join(normalized)


def _clean_for_dependency(sentence: str) -> str:
    sentence = re.sub(r"\$[^$]*\$", " MATH ", sentence or "")
    sentence = _strip_numeric_citations(sentence)
    sentence = _normalize_formula_tokens(sentence)
    sentence = re.sub(r"\bMATH(?:\s+MATH)+\b", "MATH", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def _clean_noun_phrase(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r"\bEQ_[A-Za-z0-9.]+\b", " ", text)
    text = re.sub(r"\bMATH\b", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.;:\"'")
    text = re.sub(
        r"^(?:and|or|where|which|that|this|these|those|the|a|an|by|as|to|from|of|for|with)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(?:is|are|was|were|be|been|being|can|could|would|will|then)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+as follows$", "", text, flags=re.IGNORECASE)
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    prev_text = None
    while text != prev_text:
        prev_text = text
        text = re.sub(
            r"\s+(?:and|or|including|with|of|for|in|on|at|by|to|from|is|are|was|were|be|been|being|can|could|would|will|then|the|a|an)$",
            "",
            text,
            flags=re.IGNORECASE,
        )
    return text.strip(" ,.;:")


def _valid_candidate(text: str) -> bool:
    text_lower = text.lower().strip()
    if len(text_lower) < 3 or text_lower in LOW_VALUE_CANDIDATES:
        return False
    if not re.search(r"[A-Za-z]", text_lower):
        return False
    if re.search(r"[=|{}]", text_lower):
        return False
    words = text_lower.split()
    if all(len(w) == 1 for w in words):
        return False
    return True


def _token_details(doc: Any) -> List[Dict[str, Any]]:
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.text,
            "index": token.i,
        }
        for token in doc
        if not token.is_space
    ]


def _named_entities(doc: Any) -> List[Dict[str, Any]]:
    entities = []
    seen = set()
    for ent in doc.ents:
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in ent):
            continue
        text = _clean_noun_phrase(ent.text)
        if not _valid_candidate(text):
            continue
        key = (text.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "text": text,
            "label": ent.label_,
            "start_token": ent.start,
            "end_token": ent.end,
        })
    return entities


def _equation_token_indices(doc: Any, mask: str) -> List[int]:
    return [token.i for token in doc if token.text == mask]


def _chunk_position(chunk: Any, equation_indices: List[int]) -> str:
    if not equation_indices:
        return "context"
    if chunk.end <= min(equation_indices):
        return "before_equation"
    if chunk.start > max(equation_indices):
        return "after_equation"
    return "contains_equation"


def _token_distance(chunk: Any, equation_indices: List[int]) -> int:
    if not equation_indices:
        return 99
    return min(abs(chunk.root.i - index) for index in equation_indices)


def _verb_info(token: Any) -> Dict[str, Any]:
    return {
        "verb": token.text,
        "verb_lemma": token.lemma_,
        "verb_index": token.i,
    }


def _blank_verb_info() -> Dict[str, Any]:
    return {
        "verb": "",
        "verb_lemma": "",
        "verb_index": -1,
    }


def _governing_verb(token: Any, skip_formula_cues: bool = False) -> Dict[str, Any]:
    current = token
    seen = set()
    while current.i not in seen:
        seen.add(current.i)
        if current.pos_ in {"VERB", "AUX"}:
            if current.pos_ == "AUX":
                if current.head.pos_ == "VERB" and current.head != current:
                    current = current.head
                    continue
                child_verb = next((child for child in current.children if child.pos_ == "VERB"), None)
                if child_verb:
                    current = child_verb
                    continue
            if (
                skip_formula_cues
                and current.lemma_.lower() in FORMULA_CUE_VERB_LEMMAS
                and current.head != current
            ):
                current = current.head
                continue
            return _verb_info(current)
        if current.head == current:
            break
        current = current.head
    return _blank_verb_info()


def _nearest_prior_verb(doc: Any, eq_token: Any) -> Dict[str, Any]:
    for token in reversed(list(doc[:eq_token.i])):
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        if token.lemma_.lower() in FORMULA_CUE_VERB_LEMMAS:
            continue
        return _verb_info(token)
    return _blank_verb_info()


def _path_from_verb_to_chunk(chunk: Any, verb_index: int, text: str) -> List[Dict[str, Any]]:
    if verb_index < 0:
        return []

    nodes = []
    current = chunk.root
    seen = set()
    while current.i not in seen:
        seen.add(current.i)
        nodes.append(current)
        if current.i == verb_index:
            break
        if current.head == current:
            return []
        current = current.head
    else:
        return []

    if nodes[-1].i != verb_index:
        return []

    path = []
    for node in reversed(nodes):
        path.append({
            "text": text if node == chunk.root else node.text,
            "dep": node.dep_,
            "pos": node.pos_,
            "index": node.i,
        })
    return path


def _path_text(path: List[Dict[str, Any]]) -> str:
    return " -> ".join(part["text"] for part in path)


def _expanded_chunk_text(chunk: Any, equation_indices: List[int]) -> str:
    end = chunk.end
    first_equation_index = min(equation_indices) if equation_indices else 10**9
    for child in chunk.root.children:
        if child.dep_ not in {"acl", "relcl", "prep"}:
            continue
        child_end = max(token.i for token in child.subtree) + 1
        if child_end > first_equation_index:
            continue
        end = max(end, child_end)

    return chunk.doc[chunk.start:end].text


def _equation_connection(doc: Any, mask: str) -> Dict[str, Any]:
    eq_token = next((token for token in doc if token.text == mask), None)
    if eq_token is None:
        return {
            "eq_token": mask,
            "eq_index": -1,
            "eq_dep": "",
            "eq_head": "",
            "eq_head_pos": "",
            "verb": "",
            "verb_lemma": "",
            "verb_index": -1,
        }

    verb_info = _governing_verb(eq_token, skip_formula_cues=True)
    if verb_info.get("verb_index", -1) < 0:
        verb_info = _nearest_prior_verb(doc, eq_token)
    return {
        "eq_token": eq_token.text,
        "eq_index": eq_token.i,
        "eq_dep": eq_token.dep_,
        "eq_head": eq_token.head.text,
        "eq_head_pos": eq_token.head.pos_,
        **verb_info,
    }


def _dependency_noun_phrases(doc: Any, mask: str) -> List[Dict[str, Any]]:
    equation_indices = _equation_token_indices(doc, mask)
    candidates = []
    seen = set()

    for chunk in doc.noun_chunks:
        text = _clean_noun_phrase(_expanded_chunk_text(chunk, equation_indices))
        if not _valid_candidate(text):
            continue

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        verb_info = _governing_verb(chunk.root)
        verb_path = _path_from_verb_to_chunk(chunk, verb_info.get("verb_index", -1), text)
        root_text = chunk.root.text
        if root_text == "MATH" or re.match(r"EQ_[A-Za-z0-9.]+", root_text):
            for child in reversed(list(chunk.root.children)):
                if child.pos_ in {"NOUN", "PROPN"} and child.i < chunk.root.i:
                    root_text = child.text
                    break

        candidates.append({
            "text": text,
            "root": root_text,
            "root_pos": chunk.root.pos_,
            "root_dep": chunk.root.dep_,
            "head": chunk.root.head.text,
            "head_pos": chunk.root.head.pos_,
            **verb_info,
            "verb_path": verb_path,
            "verb_path_text": _path_text(verb_path),
            "token_pos": [token.pos_ for token in chunk],
            "start_token": chunk.start,
            "end_token": chunk.end,
            "position": _chunk_position(chunk, equation_indices),
            "token_distance": _token_distance(chunk, equation_indices),
        })
    return candidates


def analyze_sentence(sentence: str, nlp: Any, mask: str) -> Dict[str, Any]:
    dependency_text = _clean_for_dependency(sentence)
    doc = nlp(dependency_text)
    return {
        "sentence": sentence,
        "dependency_text": dependency_text,
        "equation_connection": _equation_connection(doc, mask),
        "dependency_tokens": _token_details(doc),
        "dependency_noun_phrases": _dependency_noun_phrases(doc, mask),
        "named_entities": _named_entities(doc),
    }


def _sentence_items(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    equation_sentence = context.get("primary_equation_sentence", "")
    before = context.get("primary_before_text", "")
    after = context.get("primary_after_text", "")

    if equation_sentence:
        items.append({
            "sentence": equation_sentence,
            "sentence_role": "equation_sentence",
            "sentence_distance": 0,
        })
    if before:
        items.append({
            "sentence": before,
            "sentence_role": "before",
            "sentence_distance": 1,
        })
    if after:
        items.append({
            "sentence": after,
            "sentence_role": "after",
            "sentence_distance": 1,
        })
    return items


def _candidate_type(candidate: Dict[str, Any]) -> str:
    token_pos = candidate.get("token_pos", [])
    words = candidate.get("text", "").split()
    has_name_shape = any(pos == "PROPN" for pos in token_pos)
    has_internal_cap = any(word[:1].isupper() for word in words[1:])
    return "equation_name" if has_name_shape or has_internal_cap else "equation_meaning"


def _mentions_equation(sentence: str, mask: str, equation_id: str) -> bool:
    if mask in sentence:
        return True
    pattern = rf"\b(?:eq\.?|equation)\s*\(?{re.escape(equation_id)}\)?\b"
    return bool(re.search(pattern, sentence, flags=re.IGNORECASE))


def _mentions_other_mask(sentence: str, mask: str) -> bool:
    return any(other != mask for other in re.findall(r"\bEQ_[A-Za-z0-9.]+\b", sentence))


def _candidate_score(candidate: Dict[str, Any], analysis: Dict[str, Any]) -> float:
    sentence_bonus = max(0.0, 30.0 - analysis.get("sentence_distance", 2) * 10.0)
    token_bonus = max(0.0, 30.0 - candidate.get("token_distance", 99) * 4.0)
    dep_bonus = DEPENDENCY_BONUS.get(candidate.get("root_dep", ""), 0.0)
    position_bonus = 8.0 if candidate.get("position") == "before_equation" else 0.0

    word_count = len(candidate.get("text", "").split())
    length_penalty = max(0, word_count - 8) * 2.0
    return round(40.0 + sentence_bonus + token_bonus + dep_bonus + position_bonus - length_penalty, 2)


def _shared_equation_verb(candidate: Dict[str, Any], connection: Dict[str, Any]) -> bool:
    verb_index = connection.get("verb_index", -1)
    return verb_index >= 0 and candidate.get("verb_index") == verb_index


def _relation_type(candidate: Dict[str, Any], connection: Dict[str, Any]) -> str:
    if candidate.get("position") == "contains_equation":
        return "contains_equation"
    if _shared_equation_verb(candidate, connection):
        return "same_governing_verb"
    if candidate.get("root") == connection.get("eq_head"):
        return "equation_head_noun"
    return ""


def _nearest_post_equation_verb_index(
    analysis: Dict[str, Any],
    connection: Dict[str, Any],
) -> int:
    eq_index = connection.get("eq_index", -1)
    if eq_index < 0:
        return -1

    after_candidates = [
        candidate
        for candidate in analysis.get("dependency_noun_phrases", [])
        if candidate.get("verb_index", -1) > eq_index
    ]
    if not after_candidates:
        return -1

    nearest = min(
        after_candidates,
        key=lambda candidate: (
            candidate.get("verb_index", 999) - eq_index,
            candidate.get("token_distance", 99),
        ),
    )
    return nearest.get("verb_index", -1)


def _equation_group_index(sentence_analysis: List[Dict[str, Any]], mask: str) -> int:
    equation_sentence = next(
        (
            analysis.get("sentence", "")
            for analysis in sentence_analysis
            if analysis.get("sentence_role") == "equation_sentence"
        ),
        "",
    )
    masks = re.findall(r"\bEQ_[A-Za-z0-9.]+\b", equation_sentence)
    unique_masks = list(dict.fromkeys(masks))
    if len(unique_masks) < 2 or mask not in unique_masks:
        return -1
    return unique_masks.index(mask)


def _ordered_action_candidate(
    sentence_analysis: List[Dict[str, Any]],
    equation_group_index: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if equation_group_index < 0:
        return {}, {}

    before_analysis = next(
        (
            analysis
            for analysis in sentence_analysis
            if analysis.get("sentence_role") == "before"
        ),
        {},
    )
    if not before_analysis:
        return {}, {}

    all_candidates = sorted(
        [
            candidate
            for candidate in before_analysis.get("dependency_noun_phrases", [])
            if candidate.get("verb_index", -1) >= 0
        ],
        key=lambda item: (item.get("verb_index", 999), item.get("start_token", 999)),
    )
    actions = [
        candidate
        for candidate in all_candidates
        if candidate.get("root_dep") in ACTION_ROOT_DEPS
    ]
    if not all_candidates:
        return {}, {}

    if equation_group_index < len(ORDINAL_WORDS):
        ordinal = ORDINAL_WORDS[equation_group_index]
        ordinal_match = next(
            (
                candidate
                for candidate in all_candidates
                if re.search(rf"\b{ordinal}\b", candidate.get("text", ""), flags=re.IGNORECASE)
            ),
            {},
        )
        if ordinal_match:
            return ordinal_match, before_analysis

    # Only map blindly if there are explicit enumeration cues
    before_sentence = before_analysis.get("sentence", "").lower()
    if "respectively" not in before_sentence and "as follows" not in before_sentence:
        return {}, {}

    if equation_group_index < len(actions):
        return actions[equation_group_index], before_analysis

    return {}, {}


def _relation_evidence(
    mask: str,
    candidate: Dict[str, Any],
    connection: Dict[str, Any],
    relation_type: str,
) -> str:
    text = candidate.get("text", "")
    verb = connection.get("verb", "")
    head = connection.get("eq_head", "")

    if candidate.get("position") == "contains_equation":
        if verb:
            return f"'{text}' contains {mask} and is governed by '{verb}'."
        return f"'{text}' contains {mask}."

    if relation_type == "equation_head_noun":
        return f"{mask} is attached directly to the noun head '{text}'."

    if relation_type == "previous_sentence_context":
        candidate_verb = candidate.get("verb", "")
        if candidate_verb:
            return (
                f"No usable relation was found in the equation sentence; "
                f"the previous sentence connects '{text}' through '{candidate_verb}'."
            )
        return f"No usable relation was found in the equation sentence; the previous sentence contains '{text}'."

    if relation_type == "previous_sentence_ordered_action":
        candidate_verb = candidate.get("verb", "")
        return (
            f"{mask} is part of a grouped equation sentence; "
            f"the previous sentence maps this equation to '{candidate_verb}' -> '{text}' by order."
        )

    if relation_type == "current_sentence_before_context":
        candidate_verb = candidate.get("verb", "")
        if candidate_verb:
            return (
                f"No same-verb or previous-sentence relation was found; "
                f"the current sentence before {mask} connects '{text}' through '{candidate_verb}'."
            )
        return f"No same-verb or previous-sentence relation was found; '{text}' appears before {mask}."

    if verb:
        if relation_type == "nearest_post_equation_verb":
            candidate_verb = candidate.get("verb", "")
            return (
                f"{mask} is governed by '{verb}', but no noun phrase shares that verb; "
                f"the nearest following verb '{candidate_verb}' connects '{text}'."
            )
        return f"{mask} is governed by '{verb}', and '{text}' is connected to the same verb."
    if head:
        return f"{mask} is attached to '{head}', and '{text}' is connected through that dependency context."
    return f"'{text}' is the closest connected noun phrase for {mask}."


def _connected_candidate_score(candidate: Dict[str, Any], analysis: Dict[str, Any]) -> float:
    base = _candidate_score(candidate, analysis)
    if candidate.get("position") == "contains_equation":
        base += 30.0
    elif candidate.get("relation_type") == "equation_head_noun":
        base += 24.0
    elif candidate.get("relation_type") == "same_governing_verb":
        base += 20.0
        if candidate.get("root_dep") in {"dobj", "attr", "nsubj", "nsubjpass"}:
            base += 18.0
        elif candidate.get("root_dep") == "pobj":
            base -= 8.0
    elif candidate.get("relation_type") == "nearest_post_equation_verb":
        base += 10.0
        if candidate.get("root_dep") in {"dobj", "attr", "nsubj", "nsubjpass"}:
            base += 12.0
    elif candidate.get("relation_type") == "previous_sentence_context":
        base += 6.0
        if candidate.get("verb"):
            base += 8.0
        base += min(18.0, candidate.get("start_token", 0) * 0.4)
        if candidate.get("verb_lemma") in {"denote", "call", "name", "refer"}:
            base += 12.0
            if " as -> " in candidate.get("verb_path_text", ""):
                base += 8.0
    elif candidate.get("relation_type") == "previous_sentence_ordered_action":
        base += 36.0
        if any(
            re.search(rf"\b{ordinal}\b", candidate.get("text", ""), flags=re.IGNORECASE)
            for ordinal in ORDINAL_WORDS
        ):
            base += 12.0
    elif candidate.get("relation_type") == "current_sentence_before_context":
        base += 16.0
        if candidate.get("verb"):
            base += 8.0

    verb_path = candidate.get("verb_path", [])
    if verb_path:
        if verb_path[0].get("dep") == "ROOT":
            base += 25.0
        if verb_path[0].get("dep") in {"relcl", "advcl"}:
            base -= 12.0
            
    word_count = len(candidate.get("text", "").split())
    if word_count > 6:
        base -= 2.0 * (word_count - 6)
    
    if len(verb_path) > 2:
        base -= 15.0 * (len(verb_path) - 2)
    elif not verb_path and candidate.get("relation_type") == "same_governing_verb":
        base -= 40.0

    return round(base, 2)


def _equation_verb_noun_relations(mask: str, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relations = []
    for candidate in ranked:
        if candidate.get("relation_type") not in {
            "same_governing_verb",
            "contains_equation",
            "equation_head_noun",
            "nearest_post_equation_verb",
            "previous_sentence_context",
            "previous_sentence_ordered_action",
            "current_sentence_before_context",
            "extended_context_fallback",
        }:
            continue

        path = candidate.get("verb_path", [])
        path_text = candidate.get("verb_path_text", "")
        if not path_text and candidate.get("verb"):
            path_text = f"{candidate['verb']} -> {candidate.get('text', '')}"
        elif not path_text:
            path_text = candidate.get("text", "")

        relations.append({
            "equation": mask,
            "verb": candidate.get("verb", ""),
            "verb_lemma": candidate.get("verb_lemma", ""),
            "noun_phrase": candidate.get("text", ""),
            "connector_path": " -> ".join(part["text"] for part in path[1:]),
            "verb_path": path_text,
            "relation_type": candidate.get("relation_type", ""),
            "root_dep": candidate.get("root_dep", ""),
            "evidence": candidate.get("evidence", ""),
        })
    return relations


def _conjugate_verb(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in {"be", "is", "are", "was", "were"}:
        return "is"
    if lemma in {"have", "has", "had"}:
        return "has"
    if lemma in {"do", "does", "did"}:
        return "does"
    if lemma.endswith(("ch", "sh", "s", "z", "x", "o")):
        return lemma + "es"
    if lemma.endswith("y"):
        if len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
    return lemma + "s"

def _equation_verb_noun_summary(mask: str, relations: List[Dict[str, Any]]) -> str:
    if not relations:
        return ""

    path = relations[0].get("verb_path", "")
    verb = relations[0].get("verb", "")
    verb_lemma = relations[0].get("verb_lemma", "")
    
    if path and verb and verb_lemma and path.startswith(verb):
        conjugated = _conjugate_verb(verb_lemma)
        path = path.replace(verb, conjugated, 1)
        
    return f"{mask} -> {path}" if path else ""


def _prefer_bridge_noun_expansion(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ranked:
        return ranked

    top = ranked[0]
    top_text = top.get("text", "").lower()
    if top_text not in GENERIC_BRIDGE_NOUNS:
        return ranked

    top_path = top.get("verb_path_text", "")
    top_verb_index = top.get("verb_path", [{}])[0].get("index", -1) if top.get("verb_path") else -1
    replacement = next(
        (
            candidate
            for candidate in ranked[1:]
            if candidate.get("verb_path_text", "").startswith(f"{top_path} ->")
            or (
                top_verb_index >= 0
                and candidate.get("verb_path", [{}])[0].get("index", -2) == top_verb_index
                and candidate.get("verb_path_text", "").count("->") > top_path.count("->")
            )
        ),
        {},
    )
    if not replacement:
        return ranked

    return [replacement, *[candidate for candidate in ranked if candidate is not replacement]]


def _has_equation_name_term(text: str) -> bool:
    words = {word.lower().strip(" ,.;:()[]{}\"'") for word in text.split()}
    return bool(words & EQUATION_NAME_TERMS)


def _name_candidate_score(candidate: Dict[str, Any]) -> float:
    role_bonus = {
        "equation_sentence": 16.0,
        "before": 12.0,
        "after": 4.0,
    }.get(candidate.get("sentence_role", ""), 0.0)
    score = 40.0 + role_bonus
    if candidate.get("source") == "equation_name_phrase":
        score += 24.0
    if candidate.get("source") == "ner" and candidate.get("label") in USEFUL_ENTITY_LABELS:
        score += 14.0
    if _has_equation_name_term(candidate.get("text", "")):
        score += 12.0
    score += min(12.0, candidate.get("start_token", 0) * 0.25)
    return round(score, 2)


def _local_context_name_candidates(sentence_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    best_by_key: Dict[str, Dict[str, Any]] = {}

    for analysis in sentence_analysis:
        sentence = analysis.get("sentence", "")
        role = analysis.get("sentence_role", "")
        distance = analysis.get("sentence_distance", 0)
        if role == "after":
            continue

        for entity in analysis.get("named_entities", []):
            text = entity.get("text", "")
            if entity.get("label") not in USEFUL_ENTITY_LABELS:
                continue
            candidates.append({
                "text": text,
                "source": "ner",
                "label": entity.get("label", ""),
                "sentence_role": role,
                "sentence_distance": distance,
                "start_token": entity.get("start_token", 0),
                "end_token": entity.get("end_token", 0),
                "sentence": sentence,
                "evidence": f"spaCy NER tagged '{text}' as {entity.get('label', '')}.",
            })

        for phrase in analysis.get("dependency_noun_phrases", []):
            text = phrase.get("text", "")
            text_lower = text.lower()
            if not _has_equation_name_term(text):
                continue
            if text_lower in EQUATION_NAME_TERMS or text_lower.startswith("above equation"):
                continue
            candidates.append({
                "text": text,
                "source": "equation_name_phrase",
                "label": "",
                "sentence_role": role,
                "sentence_distance": distance,
                "start_token": phrase.get("start_token", 0),
                "end_token": phrase.get("end_token", 0),
                "sentence": sentence,
                "evidence": f"Local noun phrase contains an equation-name term: '{text}'.",
            })

    for candidate in candidates:
        candidate["score"] = _name_candidate_score(candidate)
        key = candidate["text"].lower()
        if key not in best_by_key or candidate["score"] > best_by_key[key]["score"]:
            best_by_key[key] = candidate

    ranked = sorted(best_by_key.values(), key=lambda item: item["score"], reverse=True)
    before_or_current = [
        candidate
        for candidate in ranked
        if candidate.get("sentence_role") in {"equation_sentence", "before"}
    ]
    return before_or_current or ranked


def rank_candidates(
    sentence_analysis: List[Dict[str, Any]],
    mask: str,
    equation_id: str,
) -> Dict[str, Any]:
    ranked = []
    best_by_key: Dict[tuple, Dict[str, Any]] = {}

    for analysis in sentence_analysis:
        sentence = analysis.get("sentence", "")
        connection = analysis.get("equation_connection", {})
        mentions_current = _mentions_equation(sentence, mask, equation_id)
        if not mentions_current and _mentions_other_mask(sentence, mask):
            continue
        if not mentions_current:
            distance = analysis.get("sentence_distance", 2)
            if distance > 2:
                continue

        for candidate in analysis.get("dependency_noun_phrases", []):
            text = candidate.get("text", "")
            relation_type = _relation_type(candidate, connection)
            if not relation_type:
                continue

            candidate_type = (
                "equation_name"
                if candidate.get("position") == "contains_equation"
                else _candidate_type(candidate)
            )
            if candidate_type == "equation_name" and not mentions_current:
                continue

            ranked_candidate = {
                "text": text,
                "candidate_type": candidate_type,
                "score": 0.0,
                "root": candidate.get("root", ""),
                "root_dep": candidate.get("root_dep", ""),
                "head": candidate.get("head", ""),
                "verb": candidate.get("verb", ""),
                "verb_lemma": candidate.get("verb_lemma", ""),
                "verb_path": candidate.get("verb_path", []),
                "verb_path_text": candidate.get("verb_path_text", ""),
                "relation_type": relation_type,
                "evidence": _relation_evidence(mask, candidate, connection, relation_type),
                "position": candidate.get("position", ""),
                "token_distance": candidate.get("token_distance", 99),
                "start_token": candidate.get("start_token", 0),
                "end_token": candidate.get("end_token", 0),
                "sentence_index": analysis.get("sentence_index", 0),
                "sentence_role": analysis.get("sentence_role", ""),
                "sentence_distance": analysis.get("sentence_distance", 0),
                "sentence": sentence,
            }
            ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
            key = (text.lower(), candidate_type)
            if key not in best_by_key or ranked_candidate["score"] > best_by_key[key]["score"]:
                best_by_key[key] = ranked_candidate

    if not best_by_key:
        group_index = _equation_group_index(sentence_analysis, mask)
        candidate, analysis = _ordered_action_candidate(sentence_analysis, group_index)
        if candidate:
            text = candidate.get("text", "")
            relation_type = "previous_sentence_ordered_action"
            candidate_type = _candidate_type(candidate)
            ranked_candidate = {
                "text": text,
                "candidate_type": candidate_type,
                "score": 0.0,
                "root": candidate.get("root", ""),
                "root_dep": candidate.get("root_dep", ""),
                "head": candidate.get("head", ""),
                "verb": candidate.get("verb", ""),
                "verb_lemma": candidate.get("verb_lemma", ""),
                "verb_path": candidate.get("verb_path", []),
                "verb_path_text": candidate.get("verb_path_text", ""),
                "relation_type": relation_type,
                "evidence": _relation_evidence(mask, candidate, {}, relation_type),
                "position": candidate.get("position", ""),
                "token_distance": candidate.get("token_distance", 99),
                "start_token": candidate.get("start_token", 0),
                "end_token": candidate.get("end_token", 0),
                "sentence_index": analysis.get("sentence_index", 0),
                "sentence_role": analysis.get("sentence_role", ""),
                "sentence_distance": analysis.get("sentence_distance", 0),
                "sentence": analysis.get("sentence", ""),
            }
            ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
            best_by_key[(text.lower(), candidate_type)] = ranked_candidate

    if not best_by_key:
        for analysis in sentence_analysis:
            if analysis.get("sentence_role") != "equation_sentence":
                continue

            sentence = analysis.get("sentence", "")
            for candidate in analysis.get("dependency_noun_phrases", []):
                if candidate.get("position") != "before_equation":
                    continue
                if candidate.get("verb_index", -1) < 0:
                    continue

                text = candidate.get("text", "")
                relation_type = "current_sentence_before_context"
                candidate_type = _candidate_type(candidate)
                ranked_candidate = {
                    "text": text,
                    "candidate_type": candidate_type,
                    "score": 0.0,
                    "root": candidate.get("root", ""),
                    "root_dep": candidate.get("root_dep", ""),
                    "head": candidate.get("head", ""),
                    "verb": candidate.get("verb", ""),
                    "verb_lemma": candidate.get("verb_lemma", ""),
                    "verb_path": candidate.get("verb_path", []),
                    "verb_path_text": candidate.get("verb_path_text", ""),
                    "relation_type": relation_type,
                    "evidence": _relation_evidence(mask, candidate, {}, relation_type),
                    "position": candidate.get("position", ""),
                    "token_distance": candidate.get("token_distance", 99),
                    "start_token": candidate.get("start_token", 0),
                    "end_token": candidate.get("end_token", 0),
                    "sentence_index": analysis.get("sentence_index", 0),
                    "sentence_role": analysis.get("sentence_role", ""),
                    "sentence_distance": analysis.get("sentence_distance", 0),
                    "sentence": sentence,
                }
                ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
                key = (text.lower(), candidate_type)
                if key not in best_by_key or ranked_candidate["score"] > best_by_key[key]["score"]:
                    best_by_key[key] = ranked_candidate

    if not best_by_key:
        for analysis in sentence_analysis:
            sentence = analysis.get("sentence", "")
            connection = analysis.get("equation_connection", {})
            mentions_current = _mentions_equation(sentence, mask, equation_id)
            if not mentions_current:
                continue

            fallback_verb_index = _nearest_post_equation_verb_index(analysis, connection)
            if fallback_verb_index < 0:
                continue

            for candidate in analysis.get("dependency_noun_phrases", []):
                if candidate.get("verb_index") != fallback_verb_index:
                    continue

                text = candidate.get("text", "")
                relation_type = "nearest_post_equation_verb"
                candidate_type = _candidate_type(candidate)
                ranked_candidate = {
                    "text": text,
                    "candidate_type": candidate_type,
                    "score": 0.0,
                    "root": candidate.get("root", ""),
                    "root_dep": candidate.get("root_dep", ""),
                    "head": candidate.get("head", ""),
                    "verb": candidate.get("verb", ""),
                    "verb_lemma": candidate.get("verb_lemma", ""),
                    "verb_path": candidate.get("verb_path", []),
                    "verb_path_text": candidate.get("verb_path_text", ""),
                    "relation_type": relation_type,
                    "evidence": _relation_evidence(mask, candidate, connection, relation_type),
                    "position": candidate.get("position", ""),
                    "token_distance": candidate.get("token_distance", 99),
                    "start_token": candidate.get("start_token", 0),
                    "end_token": candidate.get("end_token", 0),
                    "sentence_index": analysis.get("sentence_index", 0),
                    "sentence_role": analysis.get("sentence_role", ""),
                    "sentence_distance": analysis.get("sentence_distance", 0),
                    "sentence": sentence,
                }
                ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
                key = (text.lower(), candidate_type)
                if key not in best_by_key or ranked_candidate["score"] > best_by_key[key]["score"]:
                    best_by_key[key] = ranked_candidate

    if not best_by_key:
        for analysis in sentence_analysis:
            if analysis.get("sentence_role") != "before":
                continue

            sentence = analysis.get("sentence", "")
            for candidate in analysis.get("dependency_noun_phrases", []):
                text = candidate.get("text", "")
                relation_type = "previous_sentence_context"
                candidate_type = _candidate_type(candidate)
                ranked_candidate = {
                    "text": text,
                    "candidate_type": candidate_type,
                    "score": 0.0,
                    "root": candidate.get("root", ""),
                    "root_dep": candidate.get("root_dep", ""),
                    "head": candidate.get("head", ""),
                    "verb": candidate.get("verb", ""),
                    "verb_lemma": candidate.get("verb_lemma", ""),
                    "verb_path": candidate.get("verb_path", []),
                    "verb_path_text": candidate.get("verb_path_text", ""),
                    "relation_type": relation_type,
                    "evidence": _relation_evidence(mask, candidate, {}, relation_type),
                    "position": candidate.get("position", ""),
                    "token_distance": candidate.get("token_distance", 99),
                    "start_token": candidate.get("start_token", 0),
                    "end_token": candidate.get("end_token", 0),
                    "sentence_index": analysis.get("sentence_index", 0),
                    "sentence_role": analysis.get("sentence_role", ""),
                    "sentence_distance": analysis.get("sentence_distance", 0),
                    "sentence": sentence,
                }
                ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
                key = (text.lower(), candidate_type)
                if key not in best_by_key or ranked_candidate["score"] > best_by_key[key]["score"]:
                    best_by_key[key] = ranked_candidate

    best_candidate = max(best_by_key.values(), key=lambda c: c["score"]) if best_by_key else {}
    if not best_by_key or best_candidate.get("root") == "MATH":
        if best_candidate.get("root") == "MATH":
            best_by_key.clear()
        
        for analysis in sentence_analysis:
            sentence = analysis.get("sentence", "")
            for candidate in analysis.get("dependency_noun_phrases", []):
                if candidate.get("verb_index", -1) < 0:
                    continue
                if candidate.get("root") == "MATH":
                    continue
                text = candidate.get("text", "")
                relation_type = "extended_context_fallback"
                candidate_type = _candidate_type(candidate)
                ranked_candidate = {
                    "text": text,
                    "candidate_type": candidate_type,
                    "score": 0.0,
                    "root": candidate.get("root", ""),
                    "root_dep": candidate.get("root_dep", ""),
                    "head": candidate.get("head", ""),
                    "verb": candidate.get("verb", ""),
                    "verb_lemma": candidate.get("verb_lemma", ""),
                    "verb_path": candidate.get("verb_path", []),
                    "verb_path_text": candidate.get("verb_path_text", ""),
                    "relation_type": relation_type,
                    "evidence": _relation_evidence(mask, candidate, {}, relation_type),
                    "position": candidate.get("position", ""),
                    "token_distance": candidate.get("token_distance", 99),
                    "start_token": candidate.get("start_token", 0),
                    "end_token": candidate.get("end_token", 0),
                    "sentence_index": analysis.get("sentence_index", 0),
                    "sentence_role": analysis.get("sentence_role", ""),
                    "sentence_distance": analysis.get("sentence_distance", 0),
                    "sentence": sentence,
                }
                ranked_candidate["score"] = _connected_candidate_score(ranked_candidate, analysis)
                key = (text.lower(), candidate_type)
                if key not in best_by_key or ranked_candidate["score"] > best_by_key[key]["score"]:
                    best_by_key[key] = ranked_candidate

    ranked = sorted(best_by_key.values(), key=lambda item: item["score"], reverse=True)
    ranked = _prefer_bridge_noun_expansion(ranked)
    best_name = next(
        (item for item in ranked if item["candidate_type"] == "equation_name"),
        {},
    )
    best_meaning = next(
        (item for item in ranked if item["candidate_type"] == "equation_meaning"),
        {},
    )
    verb_noun_relations = _equation_verb_noun_relations(mask, ranked)
    local_name_candidates = _local_context_name_candidates(sentence_analysis)
    best_local_name = local_name_candidates[0] if local_name_candidates else {}
    equation_name_with_ner = best_local_name.get("text") or best_name.get("text", "")
    equation_verb_noun_summary = _equation_verb_noun_summary(mask, verb_noun_relations)
    
    is_valid_noun = False
    if best_local_name and best_local_name.get("text") == equation_name_with_ner:
        if best_local_name.get("source") == "equation_name_phrase":
            is_valid_noun = True
    elif best_name.get("text") == equation_name_with_ner:
        is_valid_noun = True

    if is_valid_noun and equation_name_with_ner:
        clean_summary = equation_verb_noun_summary.replace(" -> ", " ")
        if clean_summary:
            # Prevent "Hamiltonian in Hamiltonian" duplication
            if equation_name_with_ner.lower() in clean_summary.lower():
                meaning = clean_summary
            elif clean_summary.lower().replace(f"{mask} is ", "") in equation_name_with_ner.lower():
                meaning = f"{mask} is {equation_name_with_ner}"
            else:
                meaning = f"{clean_summary} in {equation_name_with_ner}".strip()
        else:
            meaning = equation_name_with_ner
    else:
        meaning = equation_verb_noun_summary.replace(" -> ", " ")

    return {
        "ranked_candidates": ranked,
        "local_context_name_candidates": local_name_candidates,
        "best_local_context_equation_name": best_local_name.get("text", ""),
        "best_local_context_equation_name_source": best_local_name.get("source", ""),
        "best_local_context_equation_name_label": best_local_name.get("label", ""),
        "best_local_context_equation_name_evidence": best_local_name.get("evidence", ""),
        "equation_name_with_ner": equation_name_with_ner,
        "equation_verb_noun_summary": equation_verb_noun_summary,
        "meaning": meaning,
        "equation_verb_noun_relations": verb_noun_relations,
        "best_equation_name": best_name.get("text", ""),
        "best_equation_name_verb": best_name.get("verb", ""),
        "best_equation_name_verb_lemma": best_name.get("verb_lemma", ""),
        "best_equation_name_evidence": best_name.get("evidence", ""),
        "best_equation_meaning": best_meaning.get("text", ""),
        "best_equation_meaning_verb": best_meaning.get("verb", ""),
        "best_equation_meaning_verb_lemma": best_meaning.get("verb_lemma", ""),
        "best_equation_meaning_evidence": best_meaning.get("evidence", ""),
    }


def _html_heading_maps(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    html_path = _find_html_path(paper_id, cache_dirs)
    if not html_path:
        return {}

    try:
        soup = BeautifulSoup(
            html_path.read_text(encoding="utf-8", errors="ignore"),
            "html.parser",
        )
    except Exception:
        return {}

    equation_ids = {str(equation_id) for equation_id in paper_data}
    headings = _blank_headings()
    heading_map: Dict[str, Dict[str, str]] = {}

    for node in soup.find_all(["h2", "h3", "h4", "table", "div", "tr"]):
        classes = node.get("class", [])
        text = _clean_context_text(node.get_text(" ", strip=True))

        if "ltx_title_section" in classes:
            headings["heading"] = text
            headings["subheading"] = ""
            headings["subsubheading"] = ""
            continue
        if "ltx_title_subsection" in classes:
            headings["subheading"] = text
            headings["subsubheading"] = ""
            continue
        if "ltx_title_subsubsection" in classes:
            headings["subsubheading"] = text
            continue

        if "ltx_equation" not in classes:
            continue

        equation_id = _html_equation_number(node)
        if equation_id in equation_ids and equation_id not in heading_map:
            heading_map[equation_id] = dict(headings)

    return heading_map


def _markdown_heading_maps(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    md_path = _find_markdown_path(paper_id, cache_dirs)
    if not md_path:
        return {}

    try:
        lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}

    heading_map: Dict[str, Dict[str, str]] = {}
    headings = _blank_headings()

    for line in lines:
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            text = _clean_context_text(heading_match.group(2))
            if level <= 2:
                headings["heading"] = text
                headings["subheading"] = ""
                headings["subsubheading"] = ""
            elif level == 3:
                headings["subheading"] = text
                headings["subsubheading"] = ""
            else:
                headings["subsubheading"] = text
            continue

        normalized_line = _normalize_match_text(line)
        if not normalized_line:
            continue

        for equation_id, eq_info in paper_data.items():
            if str(equation_id) in heading_map or not isinstance(eq_info, dict):
                continue
            equation = eq_info.get("equation", "")
            needle = _normalize_match_text(equation)[:80]
            if needle and needle in normalized_line:
                heading_map[str(equation_id)] = dict(headings)

    return heading_map


def _heading_maps(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    return (
        _html_heading_maps(paper_id, paper_data, cache_dirs)
        or _markdown_heading_maps(paper_id, paper_data, cache_dirs)
    )


def _source_contexts(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Any]:
    return (
        _html_local_contexts(paper_id, paper_data, cache_dirs)
        or _markdown_local_contexts(paper_id, paper_data, cache_dirs)
        or {}
    )


def build_equation_contexts(
    dataset: Optional[Dict[str, Any]] = None,
    dataset_path: str = DATASET_PATH,
    output_path: str = OUTPUT_PATH,
    cache_dirs: Iterable[str] = CACHE_DIRS,
) -> Tuple[Dict[str, Any], bool]:
    """
    Construct paragraph-level contexts for each equation, extracting section headings 
    and surrounding text. It masks other equations in the text to prevent parser confusion.
    
    Args:
        dataset (Optional[Dict[str, Any]]): The dataset dictionary in memory.
        dataset_path (str): Path to load dataset if not provided.
        output_path (str): Where to save the output contexts.
        cache_dirs (Iterable[str]): Directories containing cached source files.
        
    Returns:
        Tuple[Dict[str, Any], bool]: The constructed context dictionary and a boolean indicating if it was loaded from file.
    """
    we_loaded = False
    if dataset is None:
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        we_loaded = True

    output: Dict[str, Any] = {}

    for paper_id, paper_data in dataset.items():
        if not isinstance(paper_data, dict):
            continue

        heading_map = _heading_maps(paper_id, paper_data, cache_dirs)
        source_contexts = _source_contexts(paper_id, paper_data, cache_dirs)
        output[paper_id] = {}

        for equation_id, eq_info in paper_data.items():
            if not isinstance(eq_info, dict) or "equation" not in eq_info:
                continue

            equation_id = str(equation_id)
            mask = _equation_mask(equation_id)
            equation = eq_info.get("equation", "")
            headings = heading_map.get(equation_id, _blank_headings())
            source_context = source_contexts.get(equation_id, {})
            if isinstance(source_context, dict):
                local_context = source_context.get("local_context_window", "")
            else:
                local_context = source_context
            local_context = local_context or eq_info.get("local_context_window", "")
            masked_context = _mask_all_equations(local_context, paper_data)

            output[paper_id][equation_id] = {
                "paper_id": paper_id,
                "equation_id": equation_id,
                "equation_mask": mask,
                **headings,
                **_sentence_plus_after_window(masked_context, mask),
            }

    if we_loaded:
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output, we_loaded


def build_sentence_splits(contexts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the raw text contexts for each equation, splits them into individual sentences,
    and performs SpaCy dependency parsing to rank noun phrases for naming the equation.
    
    Args:
        contexts (Dict[str, Any]): The output dictionary from build_equation_contexts.
        
    Returns:
        Dict[str, Any]: A dictionary containing sentence analysis and ranked equation name candidates.
    """
    sentence_splits: Dict[str, Any] = {}
    nlp = _load_dependency_parser()

    for paper_id, paper_contexts in contexts.items():
        if not isinstance(paper_contexts, dict):
            continue

        sentence_splits[paper_id] = {}
        for equation_id, context in paper_contexts.items():
            if not isinstance(context, dict):
                continue

            sentence_items = _sentence_items(context)
            sentences = [item["sentence"] for item in sentence_items]
            sentence_analysis = []
            mask = context.get("equation_mask", _equation_mask(str(equation_id)))
            for index, item in enumerate(sentence_items):
                analysis = analyze_sentence(item["sentence"], nlp, mask)
                analysis.update({
                    "sentence_index": index,
                    "sentence_role": item["sentence_role"],
                    "sentence_distance": item["sentence_distance"],
                })
                sentence_analysis.append(analysis)

            sentence_splits[paper_id][str(equation_id)] = {
                "equation_mask": mask,
                "sentences": sentences,
                "sentence_analysis": sentence_analysis,
                **rank_candidates(sentence_analysis, mask, str(equation_id)),
            }

    return sentence_splits


def write_sentence_splits(
    contexts: Dict[str, Any],
    we_loaded: bool = False,
    output_path: str = SENTENCE_SPLITS_OUTPUT_PATH,
) -> Dict[str, Any]:
    sentence_splits = build_sentence_splits(contexts)
    if we_loaded:
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(sentence_splits, f, ensure_ascii=False, indent=2)
    return sentence_splits


def main(dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    output, we_loaded = build_equation_contexts(dataset=dataset)
    sentence_splits = write_sentence_splits(output, we_loaded)
    equation_count = sum(len(paper) for paper in output.values())
    split_count = sum(len(paper) for paper in sentence_splits.values())
    print(f"Built {equation_count} equation context record(s)")
    print(f"Built {split_count} sentence split record(s)")
    return sentence_splits


if __name__ == "__main__":
    main()
