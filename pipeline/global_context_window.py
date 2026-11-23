"""
Build paper-level context windows around extracted equations.

For each paper in ``dataset.json``, this writes one combined JSON file mapping
paper IDs to global context strings. The exporter prefers cached arXiv HTML
paragraph text scoped by PDF page boundaries, falls back to cached Markdown, and
only then uses PDF page text as a last resort. The main dataset is intentionally
left unchanged.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from parser import (
    _clean_context_text,
    _detect_equation_pages_from_pdf,
    _extract_pdf_page_texts,
    _html_equation_number,
    _html_text_with_math_alt,
)


DATASET_PATH = "dataset.json"
OUTPUT_DIR = "global_context_windows"
GLOBAL_CONTEXT_OUTPUT_FILENAME = "global_context.json"
GLOBAL_CONTEXT_PARAGRAPHS_BEFORE = 15
GLOBAL_CONTEXT_PARAGRAPHS_AFTER = 4
CACHE_DIRS = (
    "arxiv_cache",
)
EXCLUDED_HTML_CONTEXT_CLASSES = {
    "ltx_caption",
    "ltx_equation",
    "ltx_equationgroup",
    "ltx_figure",
    "ltx_float",
    "ltx_graphics",
    "ltx_listing",
    "ltx_page_footer",
    "ltx_page_navbar",
    "ltx_table",
}
MATCH_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "been",
    "being",
    "between",
    "both",
    "each",
    "from",
    "have",
    "into",
    "more",
    "only",
    "such",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "this",
    "through",
    "using",
    "where",
    "which",
    "with",
}
LATEX_MATCH_WORDS = {
    "begin",
    "displaystyle",
    "end",
    "forall",
    "frac",
    "left",
    "right",
    "textstyle",
}


def _equation_ids(paper_data: Dict[str, Any]) -> List[str]:
    """Return equation IDs from one paper entry, preserving dataset order."""
    equation_ids = []
    for equation_id, equation_info in paper_data.items():
        if isinstance(equation_info, dict) and "equation" in equation_info:
            equation_ids.append(str(equation_id))
    return equation_ids


def _find_pdf_path(paper_id: str, cache_dirs: Iterable[str]) -> Optional[Path]:
    """Find a cached PDF for an arXiv paper ID."""
    for cache_dir in cache_dirs:
        paper_dir = Path(cache_dir) / paper_id
        if not paper_dir.exists():
            continue

        direct_candidates = [
            paper_dir / f"{paper_id}.pdf",
            paper_dir / "paper.pdf",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate

        pdf_files = sorted(paper_dir.glob("*.pdf"))
        if pdf_files:
            return pdf_files[0]

    return None


def _find_paper_dir(paper_id: str, cache_dirs: Iterable[str]) -> Optional[Path]:
    """Find the cached directory for an arXiv paper ID."""
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


def _detected_equation_pages(
    equation_ids: List[str],
    all_equation_pages: Dict[str, int],
) -> List[Tuple[str, int]]:
    """Keep detected pages for the requested equations in dataset order."""
    detected = []
    for equation_id in equation_ids:
        page_index = all_equation_pages.get(equation_id)
        if page_index is not None:
            detected.append((equation_id, page_index))
    return detected


def _page_role(
    page_index: int,
    first_equation_page_index: int,
    last_equation_page_index: int,
) -> str:
    if page_index < first_equation_page_index:
        return "page_before_first_equation"
    if page_index > last_equation_page_index:
        return "page_after_last_equation"
    return "equation_window"


def _html_node_text(node: Any) -> str:
    """Extract clean text from an HTML node, preserving math as LaTeX alttext."""
    return _clean_context_text(_html_text_with_math_alt(node))


def _html_node_match_text(node: Any) -> str:
    """Extract lightweight paragraph text for matching against PDF pages."""
    temp = BeautifulSoup(str(node), "html.parser")
    for math in temp.find_all("math"):
        math.replace_with(" ")
    return _clean_context_text(temp.get_text(" ", strip=True))


def _has_excluded_context_ancestor(node: Any) -> bool:
    """Return True when a paragraph is inside a table, figure, or float."""
    current = node
    while current and getattr(current, "name", None):
        classes = set(current.get("class", []) or [])
        if classes & EXCLUDED_HTML_CONTEXT_CLASSES:
            return True
        current = current.parent
    return False


def _html_paragraph_nodes(soup: BeautifulSoup) -> List[Any]:
    """Collect body paragraph nodes, excluding tables, figures, and floats."""
    paragraphs = []
    for node in soup.find_all(class_="ltx_para"):
        if node.find(class_="ltx_para"):
            continue
        if _has_excluded_context_ancestor(node):
            continue
        if _html_node_text(node):
            paragraphs.append(node)
    return paragraphs


def _combine_parts(parts: List[str]) -> str:
    """Combine text fragments into one compact context string."""
    return " ".join(" ".join(part for part in parts if part).split())


def _match_tokens(text: str) -> List[str]:
    """Return significant word tokens for loose HTML-to-PDF page matching."""
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text or ""):
        key = token.lower()
        if key in MATCH_STOPWORDS or key in LATEX_MATCH_WORDS:
            continue
        tokens.append(key)
    return tokens


def _ordered_token_hits(needle_tokens: List[str], haystack_tokens: List[str]) -> int:
    """Count how many needle tokens appear in order inside haystack tokens."""
    if not needle_tokens or not haystack_tokens:
        return 0

    hits = 0
    cursor = 0
    for token in needle_tokens:
        while cursor < len(haystack_tokens) and haystack_tokens[cursor] != token:
            cursor += 1
        if cursor >= len(haystack_tokens):
            continue
        hits += 1
        cursor += 1
    return hits


def _compact_match_text(text: str) -> str:
    """Normalize text for exact substring checks across HTML/PDF variants."""
    text = re.sub(r"\\[A-Za-z]+", " ", text or "")
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()


def _text_matches_page_window(text: str, page_window_text: str) -> bool:
    """Return True when paragraph text appears to belong to the page window."""
    compact_text = _compact_match_text(text)
    compact_window = _compact_match_text(page_window_text)

    if len(compact_text) >= 60 and compact_text[:60] in compact_window:
        return True
    if len(compact_text) >= 100 and compact_text[-80:] in compact_window:
        return True

    tokens = _match_tokens(text)
    if len(tokens) < 3:
        return False

    window_tokens = _match_tokens(page_window_text)
    hits = _ordered_token_hits(tokens, window_tokens)
    required_hits = min(len(tokens), max(3, int(len(tokens) * 0.6)))
    return hits >= required_hits


def _build_html_context_record(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a paper-start global context window from cached arXiv HTML."""
    equation_ids = _equation_ids(paper_data)
    html_path = _find_html_path(paper_id, cache_dirs)
    if not html_path:
        return None, "cached HTML not found"

    pdf_path = _find_pdf_path(paper_id, cache_dirs)
    if not pdf_path:
        return None, "cached PDF not found for HTML page-window detection"

    page_texts = _extract_pdf_page_texts(str(pdf_path))
    if not page_texts:
        return None, "could not extract PDF page text for HTML page-window detection"

    all_equation_pages = _detect_equation_pages_from_pdf(str(pdf_path))
    detected_pages = _detected_equation_pages(equation_ids, all_equation_pages)
    if not detected_pages:
        return None, "could not detect pages for dataset equations"

    try:
        soup = BeautifulSoup(
            html_path.read_text(encoding="utf-8", errors="ignore"),
            "html.parser",
        )
    except Exception as exc:
        return None, f"could not read cached HTML: {exc}"

    paragraphs = _html_paragraph_nodes(soup)
    paragraph_index = {id(node): index for index, node in enumerate(paragraphs)}
    detected: List[Tuple[str, int]] = []
    wanted_ids = set(equation_ids)

    for eq_el in soup.find_all(class_="ltx_equation"):
        if eq_el.name not in {"table", "div"}:
            continue

        equation_id = _html_equation_number(eq_el)
        if equation_id not in wanted_ids:
            continue

        current_paragraph = eq_el.find_parent(class_="ltx_para")
        index = paragraph_index.get(id(current_paragraph))
        if index is None:
            continue

        if not any(existing_id == equation_id for existing_id, _ in detected):
            detected.append((equation_id, index))

    detected.sort(key=lambda item: equation_ids.index(item[0]))
    if not detected:
        return None, "could not locate dataset equations in cached HTML"

    first_equation_id, first_equation_page_index = detected_pages[0]
    last_equation_id, last_equation_page_index = detected_pages[-1]
    page_indices = [page_index for _, page_index in detected_pages]
    first_equation_page_index = min(page_indices)
    last_equation_page_index = max(page_indices)

    context_start_page_index = 0
    context_end_page_index = min(len(page_texts) - 1, last_equation_page_index + 1)
    page_window_text = _combine_parts(
        [
            page_texts[index]
            for index in range(context_start_page_index, context_end_page_index + 1)
        ]
    )

    matched_parts: List[str] = []
    matched_blocks: List[int] = []
    for index, paragraph in enumerate(paragraphs):
        match_text = _html_node_match_text(paragraph)
        full_text = _html_node_text(paragraph)
        if not full_text:
            continue
        if _text_matches_page_window(match_text, page_window_text) or _text_matches_page_window(full_text, page_window_text):
            matched_parts.append(full_text)
            matched_blocks.append(index)

    if not matched_parts:
        return None, "could not match HTML paragraphs to detected PDF page window"

    global_context_window = _combine_parts(matched_parts)

    return (
        {
            "paper_id": paper_id,
            "source_path": str(html_path),
            "pdf_source_path": str(pdf_path),
            "method": "html_paragraph_from_paper_start_to_next_page",
            "equation_ids": equation_ids,
            "detected_equation_blocks": {
                equation_id: index
                for equation_id, index in detected
            },
            "detected_equation_pages": {
                equation_id: page_index + 1
                for equation_id, page_index in detected_pages
            },
            "first_equation_id": first_equation_id,
            "first_equation_page": first_equation_page_index + 1,
            "last_equation_id": last_equation_id,
            "last_equation_page": last_equation_page_index + 1,
            "context_start_page": context_start_page_index + 1,
            "context_end_page": context_end_page_index + 1,
            "matched_context_blocks": matched_blocks,
            "context_start_block": matched_blocks[0],
            "context_end_block": matched_blocks[-1],
            "global_context_window": global_context_window,
            "global_context_window_chars": len(global_context_window),
        },
        None,
    )


def _normalize_match_text(text: str) -> str:
    """Normalize text for loose equation matching in Markdown fallback."""
    text = re.sub(r"\\(?:display|text)style\s*", "", text or "")
    text = re.sub(r"\\label\s*\{[^}]+\}", "", text)
    text = re.sub(r"\\tag\s*\{[^}]+\}", "", text)
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


def _build_markdown_context_record(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a global context window from cached Markdown."""
    equation_ids = _equation_ids(paper_data)
    md_path = _find_markdown_path(paper_id, cache_dirs)
    if not md_path:
        return None, "cached Markdown not found"

    try:
        blocks = _markdown_blocks(md_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return None, f"could not read cached Markdown: {exc}"

    normalized_blocks = [_normalize_match_text(block) for block in blocks]
    detected: List[Tuple[str, int]] = []
    search_start = 0

    for equation_id in equation_ids:
        eq_info = paper_data.get(equation_id, {})
        if not isinstance(eq_info, dict):
            continue

        needle = _normalize_match_text(eq_info.get("equation", ""))[:100]
        if not needle:
            continue

        found_index = None
        for index in range(search_start, len(normalized_blocks)):
            if needle in normalized_blocks[index]:
                found_index = index
                search_start = index
                break

        if found_index is None:
            for index, block in enumerate(normalized_blocks):
                if needle in block:
                    found_index = index
                    break

        if found_index is not None:
            detected.append((equation_id, found_index))

    if not detected:
        return None, "could not locate dataset equations in cached Markdown"

    first_equation_id, first_index = detected[0]
    last_equation_id, last_index = detected[-1]
    if last_index < first_index:
        indices = [index for _, index in detected]
        first_index = min(indices)
        last_index = max(indices)

    context_start_index = 0
    context_end_index = min(
        len(blocks) - 1,
        last_index + GLOBAL_CONTEXT_PARAGRAPHS_AFTER,
    )
    global_context_window = _combine_parts(
        blocks[index] for index in range(context_start_index, context_end_index + 1)
    )

    return (
        {
            "paper_id": paper_id,
            "source_path": str(md_path),
            "method": "markdown_from_paper_start_to_after_last_equation",
            "equation_ids": equation_ids,
            "detected_equation_blocks": {
                equation_id: index
                for equation_id, index in detected
            },
            "first_equation_id": first_equation_id,
            "first_equation_block": first_index,
            "last_equation_id": last_equation_id,
            "last_equation_block": last_index,
            "context_start_block": context_start_index,
            "context_end_block": context_end_index,
            "global_context_window": global_context_window,
            "global_context_window_chars": len(global_context_window),
        },
        None,
    )


def _build_pdf_context_record(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a global context window from PDF page text as a last fallback."""
    equation_ids = _equation_ids(paper_data)
    pdf_path = _find_pdf_path(paper_id, cache_dirs)
    if not pdf_path:
        return None, "cached PDF not found"

    page_texts = _extract_pdf_page_texts(str(pdf_path))
    if not page_texts:
        return None, "could not extract PDF page text"

    all_equation_pages = _detect_equation_pages_from_pdf(str(pdf_path))
    detected_pages = _detected_equation_pages(equation_ids, all_equation_pages)
    if not detected_pages:
        return None, "could not detect pages for dataset equations"

    first_equation_id, first_equation_page_index = detected_pages[0]
    last_equation_id, last_equation_page_index = detected_pages[-1]

    if last_equation_page_index < first_equation_page_index:
        page_indices = [page_index for _, page_index in detected_pages]
        first_equation_page_index = min(page_indices)
        last_equation_page_index = max(page_indices)

    context_start_index = 0
    context_end_index = min(len(page_texts) - 1, last_equation_page_index + 1)

    window_parts = [
        page_texts[page_index]
        for page_index in range(context_start_index, context_end_index + 1)
        if page_texts[page_index]
    ]
    global_context_window = _combine_parts(window_parts)

    return (
        {
            "paper_id": paper_id,
            "source_path": str(pdf_path),
            "method": "pdf_page_text_from_paper_start_to_next_page",
            "equation_ids": equation_ids,
            "detected_equation_pages": {
                equation_id: page_index + 1
                for equation_id, page_index in detected_pages
            },
            "first_equation_id": first_equation_id,
            "first_equation_page": first_equation_page_index + 1,
            "last_equation_id": last_equation_id,
            "last_equation_page": last_equation_page_index + 1,
            "context_start_page": context_start_index + 1,
            "context_end_page": context_end_index + 1,
            "global_context_window": global_context_window,
            "global_context_window_chars": len(global_context_window),
        },
        None,
    )


def _format_text_window(paper_id: str, context_record: Dict[str, Any]) -> str:
    """Create a plain text copy of one combined context window."""
    return context_record["global_context_window"].rstrip() + "\n"


def build_context_record(
    paper_id: str,
    paper_data: Dict[str, Any],
    cache_dirs: Iterable[str] = CACHE_DIRS,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Build one comprehensive context-window record for a paper.
    It sequentially attempts to build the context from HTML, then Markdown, 
    and finally PDF page text as fallbacks.

    Args:
        paper_id (str): The arXiv ID of the paper.
        paper_data (Dict[str, Any]): The dataset dict for this specific paper.
        cache_dirs (Iterable[str]): Directories to search for cached files.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: 
            ``(record, error)`` where ``record`` is ``None`` when the paper cannot
            be processed, and ``error`` contains the error message.
    """
    equation_ids = _equation_ids(paper_data)
    if not equation_ids:
        return None, "no extracted equations in dataset"

    errors = []
    for builder in (
        _build_html_context_record,
        _build_markdown_context_record,
        _build_pdf_context_record,
    ):
        record, error = builder(paper_id, paper_data, cache_dirs)
        if record:
            return record, None
        if error:
            errors.append(error)

    return None, "; ".join(errors) or "could not build context window"


def add_global_context_windows(
    dataset: Optional[Dict[str, Any]] = None,
    dataset_path: str = DATASET_PATH,
    output_dir: str = OUTPUT_DIR,
    cache_dirs: Iterable[str] = CACHE_DIRS,
) -> Dict[str, Any]:
    """
    Add a ``global_context_window`` field to every paper entry in the dataset.
    This context represents the entire document (or a significant chunk covering 
    all equations) to be used when resolving symbol meanings globally.
    
    Args:
        dataset (Optional[Dict[str, Any]]): The dataset dict in memory.
        dataset_path (str): Fallback path to load the dataset.
        output_dir (str): Directory to save the separated global context JSON.
        cache_dirs (Iterable[str]): Directories to search for cached files.
        
    Returns:
        Dict[str, Any]: A dictionary containing metrics like paper_count and skipped_count.
    """
    we_loaded = False
    if dataset is None:
        dataset_file = Path(dataset_path)
        with dataset_file.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        we_loaded = True

    index: Dict[str, Any] = {
        "papers": {},
        "skipped": {},
    }

    for paper_id, paper_data in dataset.items():
        if not isinstance(paper_data, dict):
            index["skipped"][paper_id] = "paper entry is not a dictionary"
            continue

        record, error = build_context_record(paper_id, paper_data, cache_dirs)
        if error or record is None:
            index["skipped"][paper_id] = error or "unknown error"
            continue

        # Inject into dataset in memory
        paper_data["global_context_window"] = record["global_context_window"]

        index["papers"][paper_id] = {
            "method": record["method"],
            "source_path": record.get("source_path", ""),
            "first_equation_id": record["first_equation_id"],
            "last_equation_id": record["last_equation_id"],
            "global_context_window_chars": record["global_context_window_chars"],
        }
        for optional_key in (
            "first_equation_page",
            "last_equation_page",
            "context_start_page",
            "context_end_page",
            "first_equation_block",
            "last_equation_block",
            "context_start_block",
            "context_end_block",
        ):
            if optional_key in record:
                index["papers"][paper_id][optional_key] = record[optional_key]

    index["paper_count"] = len(index["papers"])
    index["skipped_count"] = len(index["skipped"])

    if we_loaded:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / GLOBAL_CONTEXT_OUTPUT_FILENAME
        
        # Only write the separated JSON if we were running in standalone mode
        global_context_by_paper = {
            pid: pdata.get("global_context_window", "")
            for pid, pdata in dataset.items()
            if isinstance(pdata, dict) and "global_context_window" in pdata
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(global_context_by_paper, f, ensure_ascii=False, indent=2)

    return index


def main() -> None:
    index = add_global_context_windows()
    print(
        f"Wrote {index['paper_count']} global context window(s)"
    )
    if index["skipped_count"]:
        print(f"Skipped {index['skipped_count']} paper(s).")


if __name__ == "__main__":
    main()
