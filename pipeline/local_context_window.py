"""
Add paragraph-level local context windows to each extracted equation.

Each equation entry in ``dataset.json`` receives a ``local_context_window``
string made from the previous paragraph, the current paragraph containing the
equation, and the next paragraph.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from bs4 import BeautifulSoup

from parser import (
    _clean_context_text,
    _html_equation_number,
    _html_text_with_math_alt,
)


DATASET_PATH = "dataset.json"
CACHE_DIRS = (
    "arxiv_cache",
    "arxiv_cache_1",
    "arxiv_cache_here",
    os.path.join("Extraction_by_page", "arxiv_cache"),
)


def _find_paper_dir(paper_id: str, cache_dirs: Iterable[str]) -> Optional[Path]:
    """Find the cached directory for a paper."""
    for cache_dir in cache_dirs:
        paper_dir = Path(cache_dir) / paper_id
        if paper_dir.exists():
            return paper_dir
    return None


def _find_html_path(paper_id: str, cache_dirs: Iterable[str]) -> Optional[Path]:
    """Find a cached arXiv HTML file for a paper."""
    paper_dir = _find_paper_dir(paper_id, cache_dirs)
    if not paper_dir:
        return None

    direct_path = paper_dir / f"{paper_id}.html"
    if direct_path.exists():
        return direct_path

    html_files = sorted(paper_dir.glob("*.html"))
    return html_files[0] if html_files else None


def _find_markdown_path(paper_id: str, cache_dirs: Iterable[str]) -> Optional[Path]:
    """Find a cached Markdown conversion for a paper."""
    paper_dir = _find_paper_dir(paper_id, cache_dirs)
    if not paper_dir:
        return None

    candidates = [
        paper_dir / f"{paper_id}.md",
        paper_dir / "paper.md",
        paper_dir / paper_id / f"{paper_id}.md",
        paper_dir / "paper" / "paper.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    md_files = sorted(paper_dir.rglob("*.md"))
    return md_files[0] if md_files else None


def _equation_ids(paper_data: Dict[str, Any]) -> List[str]:
    """Return equation IDs from one paper entry, preserving dataset order."""
    return [
        str(equation_id)
        for equation_id, equation_info in paper_data.items()
        if isinstance(equation_info, dict) and "equation" in equation_info
    ]


def _html_node_text(node: Any) -> str:
    """Extract clean text from an HTML node, preserving math as LaTeX alttext."""
    temp = BeautifulSoup(str(node), "html.parser")
    for note in temp.find_all(class_=re.compile(r"\bltx_note\b")):
        note.decompose()
    return _clean_context_text(_html_text_with_math_alt(temp))


def _html_paragraph_nodes(soup: BeautifulSoup) -> List[Any]:
    """Collect paragraph-level LateXML nodes in document order."""
    paragraphs = []
    for node in soup.find_all(class_="ltx_para"):
        if _html_node_text(node):
            paragraphs.append(node)
    return paragraphs


def _combine_parts(parts: List[str]) -> str:
    """Combine paragraph fragments into one compact context string."""
    return " ".join(" ".join(part for part in parts if part).split())


def _html_local_contexts(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Any]:
    """Build local context windows from cached arXiv HTML."""
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

    equation_ids = set(_equation_ids(paper_data))
    paragraphs = _html_paragraph_nodes(soup)
    paragraph_index = {id(node): index for index, node in enumerate(paragraphs)}

    contexts: Dict[str, str] = {}
    for eq_el in soup.find_all(class_="ltx_equation"):
        if eq_el.name not in {"table", "div"}:
            continue

        equation_id = _html_equation_number(eq_el)
        if equation_id not in equation_ids or equation_id in contexts:
            continue

        current_paragraph = eq_el.find_parent(class_="ltx_para")
        current_index = paragraph_index.get(id(current_paragraph))
        if current_index is None:
            prev_para = eq_el.find_previous(class_="ltx_para")
            if prev_para and id(prev_para) in paragraph_index:
                prev_index = paragraph_index[id(prev_para)]
                parts = [_html_node_text(paragraphs[prev_index]), _html_node_text(eq_el)]
                if prev_index + 1 < len(paragraphs):
                    parts.append(_html_node_text(paragraphs[prev_index + 1]))
                context = _combine_parts(parts)
                
                wide_parts = []
                start_p = max(0, prev_index - 1)
                end_p = min(len(paragraphs), prev_index + 3)
                for i in range(start_p, end_p):
                    if i == prev_index + 1:
                        wide_parts.append(_html_node_text(eq_el))
                    wide_parts.append(_html_node_text(paragraphs[i]))
                chunk_text = _combine_parts(wide_parts)
                
                equation_text = _html_node_text(eq_el)
                before_text, after_text = "", ""
                if equation_text:
                    before_text, after_text = _extract_before_after_sentences(chunk_text, equation_text, num_before=2, num_after=2)
                if chunk_text:
                    contexts[equation_id] = {"local_context_window": chunk_text, "before_text": before_text, "after_text": after_text}
            else:
                equation_text = _html_node_text(eq_el)
                if equation_text:
                    contexts[equation_id] = {"local_context_window": equation_text, "before_text": "", "after_text": ""}
            continue

        parts: List[str] = []
        if current_index > 0:
            parts.append(_html_node_text(paragraphs[current_index - 1]))

        parts.append(_html_node_text(paragraphs[current_index]))

        if current_index + 1 < len(paragraphs):
            parts.append(_html_node_text(paragraphs[current_index + 1]))

        context = _combine_parts(parts)
        
        # Build wider chunk for sentence extraction
        wide_parts: List[str] = []
        start_p = max(0, current_index - 2)
        end_p = min(len(paragraphs), current_index + 3)
        for i in range(start_p, end_p):
            wide_parts.append(_html_node_text(paragraphs[i]))
        chunk_text = _combine_parts(wide_parts)
        
        equation_text = _html_node_text(eq_el)
        before_text, after_text = "", ""
        if equation_text:
            before_text, after_text = _extract_before_after_sentences(chunk_text, equation_text, num_before=2, num_after=2)
            
        if chunk_text:
            contexts[equation_id] = {
                "local_context_window": chunk_text,
                "before_text": before_text,
                "after_text": after_text
            }

    return contexts


def _normalize_match_text(text: str) -> str:
    """Normalize text for loose equation matching in Markdown fallback."""
    text = re.sub(r"\\(?:display|text)style\s*", "", text or "")
    text = re.sub(r"\\label\s*\{[^}]+\}", "", text)
    text = text.replace("$$", "").replace("$", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    return re.sub(r"\s+", "", text)


def _markdown_blocks(md_text: str) -> List[str]:
    """Split Markdown into paragraph-like blocks."""
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md_text)
    md_text = re.sub(r"<[^>]+>", " ", md_text)
    blocks = []
    for block in re.split(r"\n\s*\n+", md_text):
        cleaned = _clean_context_text(block)
        if cleaned:
            blocks.append(cleaned)
    return blocks


def _is_markdown_math_block(block: str) -> bool:
    block = (block or "").strip()
    return block.startswith("$$") or block.startswith(r"\[")


def _markdown_equation_span(blocks: List[str], current_index: int) -> Tuple[int, int]:
    """Return the full adjacent display-math span for a matched equation block."""
    start_index = current_index
    end_index = current_index + 1

    if not _is_markdown_math_block(blocks[current_index]):
        return start_index, end_index

    while start_index > 0 and _is_markdown_math_block(blocks[start_index - 1]):
        start_index -= 1
    while end_index < len(blocks) and _is_markdown_math_block(blocks[end_index]):
        end_index += 1

    return start_index, end_index


def _markdown_local_contexts(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Dict[str, Any]:
    """Build local context windows from cached Markdown when HTML is absent."""
    md_path = _find_markdown_path(paper_id, cache_dirs)
    if not md_path:
        return {}

    try:
        blocks = _markdown_blocks(md_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

    contexts: Dict[str, Any] = {}
    normalized_blocks = [_normalize_match_text(block) for block in blocks]
    for equation_id, eq_info in paper_data.items():
        if not isinstance(eq_info, dict):
            continue

        equation = eq_info.get("equation", "")
        needle = _normalize_match_text(equation)[:80]
        if not needle:
            continue

        current_index = None
        for index, block in enumerate(normalized_blocks):
            if needle and needle in block:
                current_index = index
                break

        if current_index is None:
            continue

        equation_start, equation_end = _markdown_equation_span(blocks, current_index)
        start_index = max(0, equation_start - 1)
        end_index = min(len(blocks), equation_end + 1)
        context = _combine_parts(blocks[start_index:end_index])
        
        before_text, after_text = _extract_before_after_sentences(context, equation, num_before=2, num_after=2)
        
        if context:
            contexts[str(equation_id)] = {
                "local_context_window": context,
                "before_text": before_text,
                "after_text": after_text
            }

    return contexts


def _extract_before_after_sentences(text: str, match_text: str, num_before: int = 2, num_after: int = 2) -> Tuple[str, str]:
    """
    Extract a specified number of sentences immediately before and after a matched 
    equation text within a larger paragraph chunk.
    
    Args:
        text (str): The full paragraph context text.
        match_text (str): The equation text to locate.
        num_before (int): Number of sentences to extract before the match.
        num_after (int): Number of sentences to extract after the match.
        
    Returns:
        Tuple[str, str]: A tuple containing (before_text, after_text).
    """
    sentences = _sentence_chunks(text)
    if not sentences:
        return "", ""

    norm_match = _normalize_match_text(match_text)
    
    match_idx = -1
    for i, s in enumerate(sentences):
        if norm_match in _normalize_match_text(s):
            match_idx = i
            break
            
    if match_idx != -1:
        start_idx = max(0, match_idx - num_before)
        end_idx = min(len(sentences), match_idx + num_after + 1)
        match_anchor = _text_anchor(sentences[match_idx], match_text)

        before_parts = sentences[start_idx:match_idx]
        after_parts = sentences[match_idx + 1:end_idx]
        if match_anchor:
            before_tail = _clean_context_fragment(sentences[match_idx][:match_anchor[0]])
            after_head = _clean_context_fragment(sentences[match_idx][match_anchor[1]:])
            if before_tail:
                before_parts.append(before_tail)
            if after_head:
                after_parts.insert(0, after_head)

        before_str = " ".join(before_parts[-num_before:])
        after_str = " ".join(after_parts[:num_after])
        if before_str or after_str:
            return before_str, after_str

    anchor = _text_anchor(text, match_text)
    if anchor:
        before_part = text[:anchor[0]]
        after_part = text[anchor[1]:]
        before_sentences = _sentence_chunks(before_part)
        after_sentences = _sentence_chunks(after_part)
        return (
            " ".join(before_sentences[-num_before:]),
            " ".join(after_sentences[:num_after]),
        )

    return "", ""


def _clean_context_fragment(text: str) -> str:
    """Clean text left in the same sentence as a matched equation."""
    text = _clean_context_text(text)
    text = re.sub(r"^(?:\$\$|\$)?\s*[,.;:]*\s*(?:\$\$|\$)?\s*", "", text)
    text = re.sub(r"^\(?\s*(?:[A-Z]\.?)?\d+(?:\.\d+)?[a-z]?\s*\)?\s*", "", text)
    text = re.sub(r"\s*\$\$\s*$", "", text).strip(" ,;:")
    return text


def _sentence_chunks(text: str) -> List[str]:
    """
    Split compact paper text into sentence-like chunks using NLTK's sentence tokenizer.
    Ensures that numbered equation patterns (e.g., '(1)') aren't accidentally merged 
    if they appear before a capital letter.
    
    Args:
        text (str): The continuous text to split.
        
    Returns:
        List[str]: A list of separated sentences.
    """
    text = re.sub(
        r"(\(\s*(?:[A-Z]\.?)?\d+(?:\.\d+)?[a-z]?\s*\))\s+(?=[A-Z])",
        r"\1\n",
        text or "",
    )
    
    chunks = []
    for block in re.split(r'\n+', text):
        if not block.strip():
            continue
        for sentence in nltk.sent_tokenize(block):
            if sentence.strip():
                chunks.append(sentence.strip())
                
    return chunks


def _text_anchor(text: str, match_text: str) -> Optional[Tuple[int, int]]:
    """
    Find the exact string match span of an equation inside a text block, 
    allowing for minor whitespace or formatting differences.
    
    Args:
        text (str): The larger text block.
        match_text (str): The equation text to find.
        
    Returns:
        Optional[Tuple[int, int]]: The (start_index, end_index) of the match, or None.
    """
    if not text or not match_text:
        return None

    start = text.find(match_text)
    if start >= 0:
        return start, start + len(match_text)

    compact_text = _normalize_match_text(text)
    compact_match = _normalize_match_text(match_text)
    if not compact_match:
        return None

    compact_start = compact_text.find(compact_match)
    if compact_start < 0:
        return None

    compact_index = 0
    start_char = None
    end_char = None
    for char_index, char in enumerate(text):
        if char.isspace() or char == "$":
            continue
        if compact_index == compact_start and start_char is None:
            start_char = char_index
        compact_index += 1
        if compact_index == compact_start + len(compact_match):
            end_char = char_index + 1
            break

    if start_char is None or end_char is None:
        return None
    return start_char, end_char


def _fallback_context(eq_info: Dict[str, Any]) -> str:
    """
    Construct a fallback local context string using the existing before/after 
    text fields if paragraph sources are unavailable.
    
    Args:
        eq_info (Dict[str, Any]): The equation dictionary entry.
        
    Returns:
        str: A combined context string.
    """
    return _combine_parts(
        [
            eq_info.get("before_text", ""),
            eq_info.get("equation", ""),
            eq_info.get("after_text", ""),
        ]
    )


def add_local_context_windows(
    dataset: Optional[Dict[str, Any]] = None,
    dataset_path: str = DATASET_PATH,
    cache_dirs: Iterable[str] = CACHE_DIRS,
) -> Dict[str, Any]:
    """
    Add ``local_context_window``, ``before_text``, and ``after_text`` fields 
    to every equation in a dataset. It processes each paper by looking for cached 
    HTML or Markdown files and extracting the surrounding paragraphs for each equation.

    Args:
        dataset (Optional[Dict[str, Any]]): The dataset dict in memory.
        dataset_path (str): Fallback path to load the dataset if not provided in memory.
        cache_dirs (Iterable[str]): Directories to search for cached paper source files.

    Returns:
        dict: Counts and skipped-paper details for logging or inspection.
    """
    we_loaded = False
    if dataset is None:
        dataset_file = Path(dataset_path)
        with dataset_file.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        we_loaded = True

    summary: Dict[str, Any] = {
        "papers": {},
        "updated_equations": 0,
        "skipped": {},
    }

    for paper_id, paper_data in dataset.items():
        if not isinstance(paper_data, dict):
            summary["skipped"][paper_id] = "paper entry is not a dictionary"
            continue

        contexts = _html_local_contexts(paper_id, paper_data, cache_dirs)
        source = "html_paragraph_window"
        if not contexts:
            contexts = _markdown_local_contexts(paper_id, paper_data, cache_dirs)
            source = "markdown_paragraph_window"

        updated_count = 0
        fallback_count = 0
        for equation_id, eq_info in paper_data.items():
            if not isinstance(eq_info, dict) or "equation" not in eq_info:
                continue

            context_record = contexts.get(str(equation_id))
            context_source = source

            context = ""
            before_text = ""
            after_text = ""
            if isinstance(context_record, dict):
                context = context_record.get("local_context_window", "")
                before_text = context_record.get("before_text", "")
                after_text = context_record.get("after_text", "")
            elif isinstance(context_record, str):
                context = context_record
            
            if not context:
                context = _fallback_context(eq_info)
                context_source = "existing_before_after_fallback"
                fallback_count += 1

            if not context:
                continue

            if not before_text and not after_text:
                before_text, after_text = _extract_before_after_sentences(
                    context,
                    eq_info.get("equation", ""),
                    num_before=2,
                    num_after=2,
                )

            eq_info["local_context_window"] = context
            eq_info["before_text"] = before_text
            eq_info["after_text"] = after_text
            updated_count += 1

        summary["papers"][paper_id] = {
            "updated_equations": updated_count,
            "fallback_equations": fallback_count,
            "source": source if contexts else "existing_before_after_fallback",
        }
        summary["updated_equations"] += updated_count

    if we_loaded:
        dataset_file = Path(dataset_path)
        with dataset_file.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    summary = add_local_context_windows()
    print(
        f"Updated {summary['updated_equations']} equation(s) "
        f"in {DATASET_PATH}."
    )


if __name__ == "__main__":
    main()
