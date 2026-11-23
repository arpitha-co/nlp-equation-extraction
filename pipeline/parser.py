"""
Module for parsing LaTeX files to extract equations and count references.
"""
import os
import re
import logging
import json
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

PREVIOUS_PAGE_BACKUP_PATH = "previous_page_context_backup.json"

# Equation number pattern: matches formats like 1, 1a, S1, A.1, 2.1, A.1a, etc.
_EQ_NUM_RE = r'(?:[A-Z]\.?)?[1-9]\d{0,2}(?:\.[1-9]\d{0,2})?[a-z]?'

def _clean_context_text(text: str) -> str:
    """
    Cleans extracted context text by removing noisy elements like journal citations
    and repeated metadata markers, and normalizes whitespaces.
    
    Args:
        text (str): The raw text extracted from PDF or HTML.
        
    Returns:
        str: The cleaned and normalized text.
    """
    text = text or ''
    # Remove journal citations like PhysRev.85.166 or PhysRev
    text = re.sub(r'\bPhysRev(?:\.\d+)*\b', '', text)
    # Remove repeated digit metadata markers like 1 1 1, 2 2 2, etc.
    text = re.sub(r'\b(\d+)(?:\s+\1)+\b', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _strip_trailing_punctuation(eq_content: str) -> str:
    """
    Strips trailing punctuation (like commas and periods) from an equation string,
    including those wrapped in LaTeX text formatting commands (e.g., \\text{.}).
    
    Args:
        eq_content (str): The raw LaTeX equation string.
        
    Returns:
        str: The equation string with trailing punctuation removed.
    """
    eq_content = eq_content.strip()
    while True:
        prev_len = len(eq_content)
        eq_content = eq_content.strip()
        
        # Remove trailing \text{.} or \text{,} or \mbox{.} or \mbox{,} or \textrm{.} or \textrm{,} (ignoring case)
        eq_content = re.sub(r'\\(?:text|mbox|textrm|textnormal)\s*\{\s*[,.]\s*\}\s*$', '', eq_content, flags=re.IGNORECASE)
        
        eq_content = eq_content.strip()
        
        # Remove trailing direct , or .
        if eq_content.endswith(',') or eq_content.endswith('.'):
            eq_content = eq_content[:-1]
            
        if len(eq_content) == prev_len:
            break
            
    return eq_content.strip()

def _find_pdf_path(paper_directory: str) -> str:
    """
    Finds the PDF file path within a given paper directory.
    
    Args:
        paper_directory (str): The directory containing the downloaded paper files.
        
    Returns:
        str: The absolute path to the PDF file, or an empty string if not found.
    """
    pdf_path = os.path.join(paper_directory, "paper.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    pdf_files = [f for f in os.listdir(paper_directory) if f.endswith(".pdf")]
    if pdf_files:
        return os.path.join(paper_directory, pdf_files[0])

    return ""

def _find_markdown_path(paper_directory: str) -> str:
    """
    Finds the Markdown fallback file path within a given paper directory.
    
    Args:
        paper_directory (str): The directory containing the paper files.
        
    Returns:
        str: The absolute path to the Markdown file, or an empty string if not found.
    """
    candidates = [
        os.path.join(paper_directory, "paper.md"),
        os.path.join(paper_directory, "paper", "paper.md"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    for root, _, files in os.walk(paper_directory):
        for filename in files:
            if filename.endswith(".md"):
                return os.path.join(root, filename)

    return ""

def _extract_pdf_page_texts(pdf_path: str) -> List[str]:
    """
    Extracts the raw text content of each page from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        List[str]: A list of strings, where each string is the text content of a single page.
    """
    try:
        import fitz
    except Exception as exc:
        logger.warning(f"PyMuPDF unavailable; cannot write previous-page backup: {exc}")
        return []

    page_texts = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_texts.append(_clean_context_text(page.get_text("text")))
    except Exception as exc:
        logger.warning(f"Could not extract page text from {pdf_path}: {exc}")
        return []

    return page_texts

def _detect_equation_pages_from_pdf(pdf_path: str) -> Dict[str, int]:
    """
    Detect equation page indices from right-margin equation numbers.

    Returns a mapping from equation number to zero-based page index.
    """
    try:
        import fitz
    except Exception as exc:
        logger.warning(f"PyMuPDF unavailable; cannot detect equation pages: {exc}")
        return {}

    equation_pages: Dict[str, int] = {}
    try:
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                page_width = page.rect.width
                blocks = page.get_text("dict").get("blocks", [])
                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = _clean_context_text(span.get("text", ""))
                            if not text:
                                continue
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            x0, _, x1, _ = bbox
                            in_margin = x0 > page_width * 0.60 or x1 > page_width * 0.78
                            if not in_margin:
                                continue
                            for match in re.finditer(rf"\(\s*({_EQ_NUM_RE})\s*\)", text):
                                equation_pages.setdefault(match.group(1), page_index)
    except Exception as exc:
        logger.warning(f"Could not detect equation pages from {pdf_path}: {exc}")
        return {}

    return equation_pages

def write_previous_page_context_backup(
    paper_id: str,
    equations: List[Dict[str, Any]],
    pdf_path: str,
    output_path: str = PREVIOUS_PAGE_BACKUP_PATH,
) -> Dict[str, Dict[str, Any]]:
    """
    Store adjacent PDF page text in a separate backup JSON file.

    The main dataset intentionally does not receive this text. Symbol-definition
    extraction can consult this file only as a fallback when local context and
    paper memory do not resolve a symbol.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return {}

    page_texts = _extract_pdf_page_texts(pdf_path)
    equation_pages = _detect_equation_pages_from_pdf(pdf_path)
    if not page_texts or not equation_pages:
        logger.info(f"No previous-page backup written for {paper_id}; equation pages not detected.")
        return {}

    paper_backup: Dict[str, Dict[str, Any]] = {}
    for equation in equations:
        equation_id = str(equation.get("equation_num", ""))
        page_index = equation_pages.get(equation_id)
        if page_index is None:
            continue

        entry: Dict[str, Any] = {
            "paper_id": paper_id,
            "equation_id": equation_id,
            "equation_page": page_index + 1,
            "method": "pdf_adjacent_page_text_backup",
        }

        if page_index > 0 and page_texts[page_index - 1]:
            entry.update({
                "previous_page": page_index,
                "previous_page_text": page_texts[page_index - 1],
            })

        next_page_index = page_index + 1
        if next_page_index < len(page_texts) and page_texts[next_page_index]:
            entry.update({
                "next_page": next_page_index + 1,
                "next_page_text": page_texts[next_page_index],
            })

        if "previous_page_text" not in entry and "next_page_text" not in entry:
            continue

        paper_backup[equation_id] = entry

    if not paper_backup:
        logger.info(f"No adjacent-page backup entries found for {paper_id}.")
        return {}

    backup_data: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
        except Exception:
            backup_data = {}

    existing_paper = backup_data.setdefault(paper_id, {})
    for equation_id, entry in paper_backup.items():
        existing_paper.setdefault(equation_id, {}).update(entry)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(backup_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(paper_backup)} previous-page backup entries for {paper_id} to {output_path}")
    return paper_backup

def _html_has_class(tag, class_name: str) -> bool:
    """
    Checks if a BeautifulSoup HTML tag contains a specific CSS class.
    
    Args:
        tag: BeautifulSoup tag object.
        class_name (str): The class name to check for.
        
    Returns:
        bool: True if the tag has the specified class, False otherwise.
    """
    classes = tag.get("class", []) if hasattr(tag, "get") else []
    return class_name in classes

def _is_html_formula_node(node) -> bool:
    """
    Determines if a BeautifulSoup node represents an equation/formula 
    based on standard arXiv/LaTeXML class names.
    
    Args:
        node: BeautifulSoup node.
        
    Returns:
        bool: True if the node is an equation, False otherwise.
    """
    if not getattr(node, "name", None):
        return False
    if _html_has_class(node, "ltx_equation") or _html_has_class(node, "ltx_equationgroup"):
        return True
    return bool(node.find(class_=re.compile(r'\bltx_equation(group)?\b')))

def _html_text_with_math_alt(node) -> str:
    """
    Extracts text from an HTML node, replacing inline <math> tags with 
    their 'alttext' attribute (usually the raw LaTeX) to preserve math notation.
    
    Args:
        node: BeautifulSoup node.
        
    Returns:
        str: The extracted text with LaTeX math embedded.
    """
    temp = BeautifulSoup(str(node), "html.parser")
    for math in temp.find_all("math"):
        alttext = math.get("alttext", "")
        math.replace_with(f" {alttext} ")
    return temp.get_text(" ", strip=True)

def _looks_like_formula_intro(text: str) -> bool:
    """
    Checks if a text string looks like an introductory phrase for an equation 
    using common physics/math terminology (e.g., 'where the quantity').
    
    Args:
        text (str): The text string to check.
        
    Returns:
        bool: True if it matches an intro pattern, False otherwise.
    """
    text = _clean_context_text(text).lower().rstrip(":,.;")
    intro_words = (
        "term", "quantity", "expression", "function", "operator",
        "potential", "hamiltonian", "lagrangian", "observable",
        "state", "matrix", "tensor", "coefficient", "variable",
    )
    return bool(re.search(rf"\b(?:where|with)\s+(?:the\s+)?(?:{'|'.join(intro_words)})$", text))

def _html_formula_text(node) -> str:
    """
    Extracts the LaTeX representation of an equation from an HTML formula node.
    
    Args:
        node: BeautifulSoup node representing a formula.
        
    Returns:
        str: The raw LaTeX text of the formula.
    """
    if not getattr(node, "name", None):
        return ""

    math_contents = []
    for math in node.find_all("math"):
        alttext = math.get("alttext")
        if alttext:
            # Clean up display/text style commands
            alttext = re.sub(r'\\(display|text)style\s*', '', alttext)
            math_contents.append(alttext)

    if math_contents:
        return _clean_context_text(" ".join(math_contents))

    # Fallback to textual extraction if no alttext
    temp = BeautifulSoup(str(node), "html.parser")
    for eqno in temp.find_all(class_="ltx_eqn_eqno"):
        eqno.extract()
    return _clean_context_text(temp.get_text(" ", strip=True))

def _html_next_text_after(node) -> Tuple[str, int]:
    """
    Finds the first non-equation text block immediately following an HTML node.
    
    Args:
        node: BeautifulSoup node to start searching after.
        
    Returns:
        Tuple[str, int]: The next text string found, and the number of formula nodes skipped.
    """
    skipped_formulas = 0
    while node:
        node = node.next_sibling
        if not node:
            break
        if isinstance(node, str):
            text = _clean_context_text(node)
        elif _is_html_formula_node(node):
            skipped_formulas += 1
            continue
        else:
            text = _clean_context_text(_html_text_with_math_alt(node))
        if text:
            return text, skipped_formulas
    return "", skipped_formulas

def _html_extend_formula_intro(node, text: str) -> Tuple[str, int]:
    if not _looks_like_formula_intro(text):
        return text, 0

    skipped_formulas = 0
    cursor = node
    while cursor:
        cursor = cursor.next_sibling
        if not cursor:
            break
        if isinstance(cursor, str):
            if _clean_context_text(cursor):
                return text, skipped_formulas
            continue
        if not _is_html_formula_node(cursor):
            next_text = _clean_context_text(_html_text_with_math_alt(cursor))
            if next_text:
                return f"{text} {next_text}", skipped_formulas
            continue

        formula_text = _html_formula_text(cursor)
        skipped_formulas += 1
        following_text, extra_skipped = _html_next_text_after(cursor)
        skipped_formulas += extra_skipped
        parts = [text, formula_text, following_text]
        return _clean_context_text(" ".join(part for part in parts if part)), skipped_formulas

    return text, skipped_formulas

def _split_into_sentences(text: str) -> List[str]:
    """
    Splits a continuous block of text into sentences based on punctuation,
    while ignoring common abbreviations (e.g., e.g., i.e., Ref., Eq.) to avoid false splits.
    
    Args:
        text (str): The continuous text.
        
    Returns:
        List[str]: A list of separated sentences.
    """
    text = text.strip()
    if not text:
        return []
    sentence_end = re.compile(
        r'(?<!\bRef)(?<!\bEq)(?<!\bEqs)(?<!\bFig)(?<!\bFigs)(?<!\be\.g)(?<!\bi\.e)'
        r'(?<!\bal)(?<!\bvs)(?<!\bst)(?<!\bdr)(?<!\bmr)(?<!\bmrs)(?<!\bprof)'
        r'(?<!\bdept)(?<!\bapprox)(?<!\bca)(?<!\bcf)(?<!\betc)(?<=[.!?])\s+'
    )
    parts = sentence_end.split(text)
    merged = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Re-merge if it matches an abbreviation case that wasn't covered fully
        if merged and re.search(r'\b(?:Ref|Eq|Eqs|Fig|Figs|e\.g|i\.e|al|vs|st|dr|mr|mrs|prof|dept|approx|ca|cf|etc)\.?$', merged[-1], re.IGNORECASE):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged

def _html_context(anchor, direction: str) -> Tuple[str, int]:
    """
    Return the nearest 3 sentences of text before/after an HTML equation.
    If adjacent/intervening siblings are equations, include them in the context
    (wrapped in $$) without counting them towards the 3 sentences limit.
    
    Args:
        anchor: BeautifulSoup node to start searching from.
        direction (str): Either "before" or "after".
        
    Returns:
        Tuple[str, int]: The collected context string and the number of skipped formula nodes.
    """
    items = []
    node = anchor
    sentence_count = 0
    siblings_checked = 0
    
    while node and siblings_checked < 100:
        next_node = node.previous_sibling if direction == "before" else node.next_sibling
        if not next_node:
            break
            
        # Skip whitespace string siblings when checking if the next block is a formula
        next_non_ws = next_node
        while next_non_ws and isinstance(next_non_ws, str) and not next_non_ws.strip():
            next_non_ws = next_non_ws.previous_sibling if direction == "before" else next_non_ws.next_sibling
            
        if sentence_count >= 3:
            # We already have 3 sentences. We only continue if next_non_ws is a formula.
            if not (next_non_ws and _is_html_formula_node(next_non_ws)):
                break
                
        node = next_node
        siblings_checked += 1
        
        node_items = []
        if _is_html_formula_node(node):
            formula_text = _html_formula_text(node)
            if formula_text:
                node_items.append({"type": "math", "content": f"$$ {formula_text} $$"})
        else:
            if isinstance(node, str):
                text = _clean_context_text(node)
            else:
                text = _clean_context_text(_html_text_with_math_alt(node))
            if text:
                sentences = _split_into_sentences(text)
                for s in sentences:
                    node_items.append({"type": "sentence", "content": s})
                    sentence_count += 1
                    
        if direction == "before":
            items = node_items + items
        else:
            items = items + node_items

    # Slice the items to get exactly 3 sentences and adjacent math blocks at boundary
    if direction == "before":
        collected_sentences_count = 0
        start_item_idx = len(items)
        while start_item_idx > 0:
            if collected_sentences_count >= 3:
                if items[start_item_idx - 1]["type"] != "math":
                    break
            start_item_idx -= 1
            if items[start_item_idx]["type"] == "sentence":
                collected_sentences_count += 1
        items_slice = items[start_item_idx:]
    else:
        collected_sentences_count = 0
        end_item_idx = 0
        while end_item_idx < len(items):
            if collected_sentences_count >= 3:
                if items[end_item_idx]["type"] != "math":
                    break
            if items[end_item_idx]["type"] == "sentence":
                collected_sentences_count += 1
            end_item_idx += 1
        items_slice = items[:end_item_idx]

    # Combine the items into a single string
    final_parts = []
    skipped_formulas = 0
    for item in items_slice:
        if item["type"] == "math":
            final_parts.append(item["content"])
            skipped_formulas += 1
        else:
            final_parts.append(item["content"])
            
    final_text = " ".join(final_parts)
    return _clean_context_text(final_text), skipped_formulas

def _html_backup_text_from_node(node) -> str:
    """
    Helper function to extract text from a BeautifulSoup node specifically for backup context.
    It ignores formula nodes entirely to prevent mathematical notation from cluttering the text.
    
    Args:
        node: BeautifulSoup node.
        
    Returns:
        str: The extracted plain text.
    """
    if isinstance(node, str):
        return _clean_context_text(node)
    if _is_html_formula_node(node):
        return ""
    return _clean_context_text(_html_text_with_math_alt(node))

def _html_wider_previous_context(anchor, max_chars: int = 8000) -> str:
    """
    Collect wider previous HTML text before an equation as backup context.

    This is not PDF page text. It is a DOM-based fallback used only when
    local equation context and paper memory fail to resolve a symbol.
    It collects up to max_chars characters of preceding text.
    
    Args:
        anchor: The BeautifulSoup node of the equation.
        max_chars (int): The maximum character limit for the backup text.
        
    Returns:
        str: A large block of preceding context text.
    """
    parts: List[str] = []
    total_chars = 0
    node = anchor

    while node and getattr(node, "name", None) not in {"body", "html"}:
        sibling = node.previous_sibling
        while sibling:
            text = _html_backup_text_from_node(sibling)
            if text:
                parts.append(text)
                total_chars += len(text)
                if total_chars >= max_chars:
                    break
            sibling = sibling.previous_sibling
        if total_chars >= max_chars:
            break
        node = node.parent

    if not parts:
        return ""

    text = _clean_context_text(" ".join(reversed(parts)))
    return text[-max_chars:]

def _html_equation_number(eq_el) -> str:
    """
    Extracts the official equation number (e.g., '1', 'A.1', 'S2') from an HTML formula element.
    
    Args:
        eq_el: BeautifulSoup node representing the equation.
        
    Returns:
        str: The extracted equation number.
    """
    # First try the displayed equation number tag (e.g., "(S2)", "(3a)").
    # This is the most reliable source and correctly captures supplementary prefixes.
    eqno = eq_el.find(class_="ltx_eqn_eqno")
    if eqno:
        match = re.search(rf"\(\s*({_EQ_NUM_RE})\s*\)", eqno.get_text(" ", strip=True))
        if match:
            return match.group(1)

    # Fall back to parsing the element id (e.g., "Sx1.E3" -> "3").
    eq_id = eq_el.get("id", "")
    match = re.search(r"\bE([1-9]\d{0,2}[a-z]?)\b", eq_id)
    if match:
        return match.group(1)

    return ""

def write_html_context_backup(
    paper_id: str,
    html_path: str,
    equations: List[Dict[str, Any]],
    output_path: str = PREVIOUS_PAGE_BACKUP_PATH,
) -> Dict[str, Dict[str, Any]]:
    """
    Store wider previous HTML context in the separate backup JSON file.
    """
    if not html_path or not os.path.exists(html_path):
        return {}

    equation_ids = {str(equation.get("equation_num", "")) for equation in equations}
    if not equation_ids:
        return {}

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
    except Exception as exc:
        logger.warning(f"Could not read HTML for backup {html_path}: {exc}")
        return {}

    paper_backup: Dict[str, Dict[str, Any]] = {}
    for eq_el in soup.find_all(class_="ltx_equation"):
        if eq_el.name not in {"table", "div"}:
            continue
        equation_id = _html_equation_number(eq_el)
        if equation_id not in equation_ids or equation_id in paper_backup:
            continue

        context_text = _html_wider_previous_context(eq_el)
        if not context_text:
            continue

        paper_backup[equation_id] = {
            "paper_id": paper_id,
            "equation_id": equation_id,
            "html_previous_context": context_text,
            "html_context_chars": len(context_text),
            "html_method": "html_dom_previous_context_backup",
        }

    if not paper_backup:
        logger.info(f"No HTML backup entries found for {paper_id}.")
        return {}

    backup_data: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
        except Exception:
            backup_data = {}

    existing_paper = backup_data.setdefault(paper_id, {})
    for equation_id, entry in paper_backup.items():
        existing_paper.setdefault(equation_id, {}).update(entry)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(backup_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(paper_backup)} HTML backup entries for {paper_id} to {output_path}")
    return paper_backup

def _markdown_equation_pattern(equation: Dict[str, Any]) -> str:
    """
    Creates a regex pattern to locate a specific equation in a Markdown document.
    It cleans up LaTeX labels and returns an escaped string for safe regex searching.
    
    Args:
        equation (Dict[str, Any]): The equation dictionary.
        
    Returns:
        str: An escaped regex pattern string to find the equation.
    """
    equation_content = re.sub(r"\\label\s*\{[^}]+\}", "", equation.get("equation_content", ""))
    equation_content = _clean_context_text(equation_content)
    if not equation_content:
        return ""
    return re.escape(equation_content[:160])

def _markdown_wider_previous_context(md_text: str, position: int, max_chars: int = 8000) -> str:
    """
    Extracts a large chunk of text preceding a given position in a Markdown document.
    Strips out markdown image syntax and HTML tags to keep the text clean.
    
    Args:
        md_text (str): The full Markdown text.
        position (int): The starting index (from where to look backwards).
        max_chars (int): The maximum character limit for the context.
        
    Returns:
        str: The cleaned preceding text block.
    """
    if position <= 0:
        return ""
    previous = md_text[:position]
    previous = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", previous)
    previous = re.sub(r"<[^>]+>", " ", previous)
    return _clean_context_text(previous)[-max_chars:]

def write_markdown_context_backup(
    paper_id: str,
    md_path: str,
    equations: List[Dict[str, Any]],
    output_path: str = PREVIOUS_PAGE_BACKUP_PATH,
) -> Dict[str, Dict[str, Any]]:
    """
    Store wider previous Markdown context in the separate backup JSON file.

    This is used only when HTML is not available. It is not page-faithful PDF
    context; it is a fallback from the converted Markdown text.
    
    Args:
        paper_id (str): The arXiv ID.
        md_path (str): The path to the Markdown file.
        equations (List[Dict[str, Any]]): Extracted equations to find contexts for.
        output_path (str): The path to the backup JSON file.
        
    Returns:
        Dict: A dictionary containing the backup context for each equation.
    """
    if not md_path or not os.path.exists(md_path):
        return {}

    try:
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            md_text = f.read()
    except Exception as exc:
        logger.warning(f"Could not read Markdown for backup {md_path}: {exc}")
        return {}

    paper_backup: Dict[str, Dict[str, Any]] = {}
    search_start = 0
    for equation in equations:
        equation_id = str(equation.get("equation_num", ""))
        pattern = _markdown_equation_pattern(equation)
        if not equation_id or not pattern:
            continue

        match = re.search(pattern, md_text[search_start:], re.DOTALL)
        if not match:
            match = re.search(pattern, md_text, re.DOTALL)
            if not match:
                continue
            position = match.start()
        else:
            position = search_start + match.start()
            search_start = search_start + match.end()

        context_text = _markdown_wider_previous_context(md_text, position)
        if not context_text:
            continue

        paper_backup[equation_id] = {
            "paper_id": paper_id,
            "equation_id": equation_id,
            "markdown_previous_context": context_text,
            "markdown_context_chars": len(context_text),
            "markdown_method": "markdown_previous_context_backup",
        }

    if not paper_backup:
        logger.info(f"No Markdown backup entries found for {paper_id}.")
        return {}

    backup_data: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
        except Exception:
            backup_data = {}

    existing_paper = backup_data.setdefault(paper_id, {})
    for equation_id, entry in paper_backup.items():
        existing_paper.setdefault(equation_id, {}).update(entry)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(backup_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(paper_backup)} Markdown backup entries for {paper_id} to {output_path}")
    return paper_backup

def _markdown_context(body: str, start_pos: int, end_pos: int, direction: str) -> Tuple[str, int]:
    """
    Return the nearest 3 sentences of Markdown text before/after a math block.
    If adjacent/intervening blocks are other equations/math blocks, include them
    (wrapped in $$) without counting them towards the 3 sentences limit.
    
    Args:
        body (str): The complete Markdown document text.
        start_pos (int): The starting index of the math block.
        end_pos (int): The ending index of the math block.
        direction (str): Either "before" or "after".
        
    Returns:
        Tuple[str, int]: The extracted context text and the number of skipped formula blocks.
    """
    math_re = re.compile(r'\$\$.*?\$\$|\\\[.*?\\\]', re.DOTALL)
    
    if direction == "before":
        parts = []
        last_idx = 0
        for m in math_re.finditer(body, 0, start_pos):
            if m.start() > last_idx:
                parts.append({"type": "text", "content": body[last_idx:m.start()]})
            math_content = m.group(0).strip("$[]\\").strip()
            parts.append({"type": "math", "content": f"$$ {math_content} $$"})
            last_idx = m.end()
        if last_idx < start_pos:
            parts.append({"type": "text", "content": body[last_idx:start_pos]})
            
        items = []
        for part in parts:
            if part["type"] == "math":
                items.append(part)
            else:
                cleaned_text = _clean_context_text(part["content"])
                if cleaned_text:
                    sentences = _split_into_sentences(cleaned_text)
                    for s in sentences:
                        items.append({"type": "sentence", "content": s})
                        
        collected_sentences_count = 0
        start_item_idx = len(items)
        while start_item_idx > 0:
            if collected_sentences_count >= 3:
                if items[start_item_idx - 1]["type"] != "math":
                    break
            start_item_idx -= 1
            if items[start_item_idx]["type"] == "sentence":
                collected_sentences_count += 1
        items_slice = items[start_item_idx:]
        
    else:
        parts = []
        last_idx = end_pos
        for m in math_re.finditer(body, end_pos):
            if m.start() > last_idx:
                parts.append({"type": "text", "content": body[last_idx:m.start()]})
            math_content = m.group(0).strip("$[]\\").strip()
            parts.append({"type": "math", "content": f"$$ {math_content} $$"})
            last_idx = m.end()
        if last_idx < len(body):
            parts.append({"type": "text", "content": body[last_idx:]})
            
        items = []
        for part in parts:
            if part["type"] == "math":
                items.append(part)
            else:
                cleaned_text = _clean_context_text(part["content"])
                if cleaned_text:
                    sentences = _split_into_sentences(cleaned_text)
                    for s in sentences:
                        items.append({"type": "sentence", "content": s})
                        
        collected_sentences_count = 0
        end_item_idx = 0
        while end_item_idx < len(items):
            if collected_sentences_count >= 3:
                if items[end_item_idx]["type"] != "math":
                    break
            if items[end_item_idx]["type"] == "sentence":
                collected_sentences_count += 1
            end_item_idx += 1
        items_slice = items[:end_item_idx]

    # Combine the items into a single string
    final_parts = []
    skipped_formulas = 0
    for item in items_slice:
        if item["type"] == "math":
            final_parts.append(item["content"])
            skipped_formulas += 1
        else:
            final_parts.append(item["content"])
            
    final_text = " ".join(final_parts)
    return _clean_context_text(final_text), skipped_formulas

def extract_equations_from_html(html_path: str) -> List[Dict[str, Any]]:
    """
    Parses an arXiv HTML file to extract numbered equations and reference counts.
    
    Parameters
    ----------
    html_path : str
        The path to the HTML file.
        
    Returns
    -------
    List[Dict[str, Any]]
        A list of parsed equations.
    """
    equations = []
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_text = f.read()
            
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Get full text for reference counting
        full_text = soup.get_text(separator=' ')
        
        # Find all equation tables/rows
        eq_elements = soup.find_all(class_="ltx_equation")
        
        processed_elements = set()
        for eq_el in eq_elements:
            if eq_el in processed_elements:
                continue

            # Find parent equation group if exists
            group_el = eq_el.find_parent(class_=re.compile(r'\bltx_equationgroup\b'))
            if group_el:
                group_rows = group_el.find_all(class_="ltx_equation")
                context_anchor = group_el
            else:
                group_rows = [eq_el]
                context_anchor = eq_el
                
            for row in group_rows:
                processed_elements.add(row)
                
            # Group rows within the equation group
            current_eq_parts = []
            equations_in_group = []
            
            for row in group_rows:
                math_tags = row.find_all('math')
                math_contents = [tag.get('alttext') for tag in math_tags if tag.get('alttext')]
                row_text = " ".join(math_contents)
                
                is_continuation = False
                if current_eq_parts:
                    current_has_eqno = bool(row.find(class_="ltx_eqn_eqno"))
                    previous_block_has_eqno = any(r.find(class_="ltx_eqn_eqno") for r in current_eq_parts)
                    
                    # A row continues a previous block if there's no conflict of separate equation numbers
                    if not (current_has_eqno and previous_block_has_eqno):
                        clean_start = row_text.strip()
                        while True:
                            prev = clean_start
                            clean_start = re.sub(r'^\\(display|text)style\s*', '', clean_start)
                            clean_start = re.sub(r'^\\phantom\{[^}]+\}\s*', '', clean_start)
                            clean_start = re.sub(r'^\\hspace\*?\{[^}]+\}\s*', '', clean_start)
                            clean_start = re.sub(r'^\\(quad|qquad|thinspace|negspace|!|,|;|:)\s*', '', clean_start)
                            clean_start = clean_start.strip()
                            if clean_start == prev:
                                break
                        
                        if (clean_start.startswith('+') or 
                            clean_start.startswith('-') or 
                            clean_start.startswith(r'\times') or 
                            clean_start.startswith(r'\cdot') or 
                            clean_start.startswith(r'\oplus') or 
                            clean_start.startswith(r'\otimes') or 
                            clean_start.startswith(r'\dots') or 
                            clean_start.startswith(r'\dotsb') or 
                            clean_start.startswith(r'\dotsc') or 
                            clean_start.startswith(r'\dotsi') or 
                            clean_start.startswith(r'\dotso')):
                            is_continuation = True
                        elif not any(rel in row_text for rel in ['=', r'\le', r'\ge', r'\approx', r'\equiv', r'\sim', r'\Rightarrow', r'\rightarrow', r'\to', r'\neq', r'\cong']):
                            is_continuation = True
                            
                if is_continuation:
                    current_eq_parts.append(row)
                else:
                    if current_eq_parts:
                        equations_in_group.append(current_eq_parts)
                    current_eq_parts = [row]
                    
            if current_eq_parts:
                equations_in_group.append(current_eq_parts)
                
            # Process each resolved equation block
            for block in equations_in_group:
                eqno_tag = None
                for row in block:
                    eqno_tag = row.find(class_="ltx_eqn_eqno")
                    if eqno_tag:
                        break
                        
                if not eqno_tag:
                    continue
                    
                eq_num = eqno_tag.get_text(" ", strip=True).strip("()").strip()
                
                # Gather all math alttext strings and image placeholders from the block
                math_contents = []
                for row in block:
                    math_tags = row.find_all('math')
                    for tag in math_tags:
                        if tag.get('alttext'):
                            math_contents.append(tag.get('alttext'))
                    
                    # Also look for images that replace complex math/diagrams
                    img_tags = row.find_all('img')
                    for img in img_tags:
                        classes = img.get('class', [])
                        if 'ltx_markedasmath' in classes or 'ltx_graphics' in classes:
                            src = img.get('src', 'unknown.png')
                            math_contents.append(f"[Image Equation: {src}]")
                            
                if math_contents:
                    has_displaystyle = any(c.strip().startswith(r'\displaystyle') for c in math_contents)
                    has_textstyle = any(c.strip().startswith(r'\textstyle') for c in math_contents)
                    
                    cleaned_contents = []
                    for c in math_contents:
                        # Clean up nested PGF/TikZ graphics blocks
                        c = re.sub(r'\\leavevmode\\hbox to.*?\\endpgfpicture\}\}', '[Diagram]', c, flags=re.DOTALL)
                        c = re.sub(r'\\pgfpicture.*?\\endpgfpicture', '[Diagram]', c, flags=re.DOTALL)
                        c_clean = re.sub(r'\\(display|text)style\s*', '', c)
                        cleaned_contents.append(c_clean)
                        
                    eq_content = " ".join(cleaned_contents)
                    if has_displaystyle:
                        eq_content = r'\displaystyle ' + eq_content
                    elif has_textstyle:
                        eq_content = r'\textstyle ' + eq_content
                else:
                    for row in block:
                        temp_eqno = row.find(class_="ltx_eqn_eqno")
                        if temp_eqno:
                            temp_eqno.extract()
                    eq_content = " ".join(row.get_text().strip() for row in block)
                    
                eq_content = " ".join(eq_content.split())
                eq_content = _strip_trailing_punctuation(eq_content)
                
                # Count references
                text_pattern = rf"\bEq(?:uation)?s?\.?[\s~]*\(?{re.escape(eq_num)}\)?\b"
                ref_count = len(re.findall(text_pattern, full_text, re.IGNORECASE))
                before_text, skipped_before = _html_context(context_anchor, "before")
                after_text, skipped_after = _html_context(context_anchor, "after")
                
                if not any(eq["equation_num"] == eq_num for eq in equations):
                    equations.append({
                        "equation_num": eq_num,
                        "label": "",
                        "equation_content": eq_content,
                        "reference_count": ref_count,
                        "before_text": before_text,
                        "after_text": after_text,
                        "context_note": f"Skipped {skipped_before} formula(s) before and {skipped_after} formula(s) after while finding context."
                    })
                    
        def sort_key(eq):
            eq_num = eq["equation_num"]
            # Parse: optional letter prefix (with optional dot), main number,
            # optional section sub-number, optional letter suffix.
            m = re.match(r'(?:([A-Z])\.?)?(\d+)(?:\.(\d+))?([a-z])?$', eq_num)
            if m:
                prefix = m.group(1) or ''
                main_num = int(m.group(2))
                sub_num = int(m.group(3)) if m.group(3) else 0
                suffix = m.group(4) or ''
                # Prefixed equations (S1, A.1) sort after non-prefixed (1, 2.1)
                has_prefix = 1 if prefix else 0
                return (has_prefix, prefix, main_num, sub_num, suffix)
            return (2, '', 999, 0, eq_num)
            
        equations.sort(key=sort_key)
        return equations
        
    except Exception as e:
        logger.error(f"Failed to parse equations from HTML {html_path}: {e}")
        return []
def extract_equations_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """
    Parses Markdown text converted by marker-pdf to extract numbered equations.
    Handles multiline formulas split across consecutive $$ blocks by merging
    unnumbered blocks that immediately precede a numbered block.
    Parameters
    ----------
    md_text : str
        The full text of the markdown paper.
    Returns
    -------
    List[Dict[str, Any]]
        A list of parsed equations.
    """
    equations = []
    
    # Strip references section to avoid parsing bibliography equations
    ref_match = re.search(r'\n#+\s*(?:References|Bibliography)\b', md_text, re.IGNORECASE)
    if ref_match:
        body_text = md_text[:ref_match.start()]
    else:
        body_text = md_text

    def _get_eq_num_from_block(content_str, end_position, post_body):
        """
        Check if a block has an equation number via \\tag{N}, (N) inside, or (N) after.
        Returns (eq_num, cleaned_content) or (None, content_str) if unnumbered.
        """
        # Check 1: \tag{N} inside block
        tag_match = re.search(rf'\\tag\{{\s*({_EQ_NUM_RE})\s*\}}', content_str)
        if tag_match:
            eq_num = tag_match.group(1).replace(" ", "")
            cleaned = re.sub(r'\\tag\{\s*[^}]+\s*\}', '', content_str).strip()
            return eq_num, cleaned
        
        # Check 2c: \\ (N) \end{...} - Check this FIRST to preserve \\ if needed
        embedded_match = re.search(rf'(\\\\)\s*\(\s*({_EQ_NUM_RE})\s*\)\s*(\\end\{{(?:split|aligned|align\*?)\}})\s*$', content_str)
        if embedded_match:
            eq_num = embedded_match.group(2).replace(" ", "")
            cleaned = content_str[:embedded_match.start()].strip() + ' ' + embedded_match.group(1) + ' ' + embedded_match.group(3)
            return eq_num, cleaned

        # Check 2: (N) at the end of block (including inside \end{split}/\end{aligned})
        inside_match = re.search(rf'\(\s*({_EQ_NUM_RE})\s*\)\s*(?:\\end\{{(?:split|aligned|align\*?)\}}\s*)?$', content_str)
        if inside_match:
            eq_num = inside_match.group(1).replace(" ", "")
            end_tag_match = re.search(r'\\end\{(?:split|aligned|align\*?)\}', content_str[inside_match.start():])
            cleaned = content_str[:inside_match.start()].strip()
            if end_tag_match:
                cleaned += ' ' + end_tag_match.group(0)
            return eq_num, cleaned
        
        # Check 2b: \end{split/aligned} ... (N) at end
        end_env_match = re.search(rf'(\\end\{{(?:split|aligned|align\*?)\}})\s*\(\s*({_EQ_NUM_RE})\s*\)\s*$', content_str)
        if end_env_match:
            eq_num = end_env_match.group(2).replace(" ", "")
            cleaned = content_str[:end_env_match.start()].strip() + ' ' + end_env_match.group(1)
            return eq_num, cleaned
        
        # Check 3: (N) on first non-empty line following the block
        post_text = post_body[end_position:end_position + 100]
        post_lines = [line.strip() for line in post_text.split("\n") if line.strip()]
        if post_lines:
            post_match = re.search(rf'^\s*\(\s*({_EQ_NUM_RE})\s*\)', post_lines[0])
            if post_match:
                eq_num = post_match.group(1).replace(" ", "")
                return eq_num, content_str
        
        return None, content_str

    def _text_between_is_trivial(text_between):
        """
        Check if the text between two $$ blocks is trivial (whitespace, empty lines,
        <span> tags, or very short connectors like commas/semicolons).
        Returns True if blocks should be considered consecutive parts of one equation.
        """
        cleaned = re.sub(r'<span[^>]*>\s*</span>', '', text_between)
        cleaned = cleaned.strip()
        # Allow empty text, or just whitespace/newlines
        if not cleaned:
            return True
        # Allow very short connectors (comma, semicolon, "and", "where")
        if len(cleaned) <= 5 and cleaned in (',', ';', 'and', 'where', '.'):
            return True
        return False

    def _process_blocks(blocks_list, body):
        """
        Process a list of regex match objects for math blocks ($$ or \\[).
        For each numbered block, merge preceding unnumbered consecutive blocks.
        """
        result = []
        # Track which block indices are "consumed" as part of a multiline merge
        consumed = set()
        
        for i, m in enumerate(blocks_list):
            if i in consumed:
                continue
                
            content = m.group(1).strip()
            start_pos = m.start()
            end_pos = m.end()
            
            eq_num, cleaned_content = _get_eq_num_from_block(content, end_pos, body)
            
            if eq_num is None:
                # Unnumbered block — skip, it may be consumed by a later numbered block
                continue
            
            # We found a numbered block. Now look backward for consecutive unnumbered blocks.
            merged_parts = []
            j = i - 1
            while j >= 0 and j not in consumed:
                prev_content = blocks_list[j].group(1).strip()
                prev_end = blocks_list[j].end()
                prev_eq_num, _ = _get_eq_num_from_block(prev_content, prev_end, body)
                
                if prev_eq_num is not None:
                    # Previous block is numbered — stop merging
                    break
                
                # Check text between blocks[j] and blocks[j+1]
                next_block = blocks_list[j + 1]
                text_between = body[prev_end:next_block.start()]
                
                if _text_between_is_trivial(text_between):
                    merged_parts.insert(0, prev_content)
                    consumed.add(j)
                    j -= 1
                else:
                    # Significant text between blocks — stop merging
                    break
            
            # If the cleaned_content of the numbered block is empty (tag-only block),
            # and we have preceding parts, the content is from the previous blocks
            if not cleaned_content and merged_parts:
                cleaned_content = " ".join(merged_parts)
                merged_parts = []
            elif not cleaned_content and not merged_parts and i > 0:
                # Fallback: use the previous block content if available
                prev = blocks_list[i - 1]
                prev_eq_num_check, _ = _get_eq_num_from_block(prev.group(1).strip(), prev.end(), body)
                if prev_eq_num_check is None:
                    cleaned_content = prev.group(1).strip()
            
            # Build final content: preceding parts + this block's content
            if merged_parts:
                all_parts = merged_parts + [cleaned_content]
                final_content = " ".join(all_parts)
            else:
                final_content = cleaned_content
            
            consumed.add(i)
            
            if not any(eq["equation_num"] == eq_num for eq in result):
                text_pattern = rf"\bEq(?:uation)?s?\.?[\s~]*\(?{re.escape(eq_num)}\)?\b"
                ref_count = len(re.findall(text_pattern, body, re.IGNORECASE))
                
                label_match = re.search(r'\\label\s*\{([^}]+)\}', final_content)
                label = label_match.group(1) if label_match else ""
                clean_content = re.sub(r'\\label\s*\{[^}]+\}', '', final_content).strip()
                clean_content = _strip_trailing_punctuation(clean_content)
                before_text, skipped_before = _markdown_context(body, start_pos, end_pos, "before")
                after_text, skipped_after = _markdown_context(body, start_pos, end_pos, "after")
                
                result.append({
                    "equation_num": eq_num,
                    "label": label,
                    "equation_content": clean_content,
                    "reference_count": ref_count,
                    "before_text": before_text,
                    "after_text": after_text,
                    "context_note": f"Skipped {skipped_before} formula(s) before and {skipped_after} formula(s) after while finding context."
                })

        return result

    # Match $$ ... $$ math blocks
    blocks = list(re.finditer(r'\$\$(.*?)\$\$', body_text, re.DOTALL))
    equations.extend(_process_blocks(blocks, body_text))
    
    # Match \[ ... \] math blocks
    blocks_bracket = list(re.finditer(r'\\\[(.*?)\\\]', body_text, re.DOTALL))
    equations.extend(_process_blocks(blocks_bracket, body_text))

    def sort_key(eq):
        num_match = re.match(r'(\d+)([a-z]*)', eq["equation_num"])
        if num_match:
            return (int(num_match.group(1)), num_match.group(2))
        return (999, eq["equation_num"])
        
    equations.sort(key=sort_key)
    return equations
    
def extract_equations_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts numbered equations from a PDF file using marker-pdf conversion,
    falling back to rule-based pypdf extraction if anything fails.
    Parameters
    ----------
    pdf_path : str
        The path to the PDF file to parse.
    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing the extracted equations.
    """
    import subprocess
    
    paper_dir = os.path.dirname(pdf_path)
    md_path = None
    
    # Check if a converted markdown file already exists (caching)
    if os.path.exists(os.path.join(paper_dir, "paper.md")):
        md_path = os.path.join(paper_dir, "paper.md")
    elif os.path.exists(os.path.join(paper_dir, "paper", "paper.md")):
        md_path = os.path.join(paper_dir, "paper", "paper.md")
    else:
        for root, _, files in os.walk(paper_dir):
            for f in files:
                if f.endswith(".md"):
                    md_path = os.path.join(root, f)
                    break
            if md_path:
                break
                
    # If not converted yet, run marker-pdf
    if not md_path:
        import shutil
        marker_single_bin = shutil.which("marker_single")
        if marker_single_bin:
            logger.info(f"Converting PDF to Markdown using marker-pdf: {pdf_path}")
            cmd = [marker_single_bin, pdf_path, "--output_dir", paper_dir]
            env = os.environ.copy()
            # Only use GPU 0 - make it the only visible device (remapped to index 0)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # 40GB VRAM Optimization: Increase batch sizes for underlying Surya models
            env["DETECTOR_BATCH_SIZE"] = "24"
            env["RECOGNITION_BATCH_SIZE"] = "24"
            env["LAYOUT_BATCH_SIZE"] = "24"
            env["TABLE_REC_BATCH_SIZE"] = "24"
            
            try:
                # Debug: check what GPU the subprocess sees
                subprocess.run(
                    [
                        "python",
                        "-c",
                        """
import torch
print("CUDA_VISIBLE_DEVICES:", __import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")
"""
                    ],
                    env=env,
                )
                # Run marker
                subprocess.run(
                    cmd,
                    env=env,
                    timeout=1800,
                    capture_output=True
                )
                # Check for created markdown file again
                if os.path.exists(os.path.join(paper_dir, "paper.md")):
                    md_path = os.path.join(paper_dir, "paper.md")
                elif os.path.exists(os.path.join(paper_dir, "paper", "paper.md")):
                    md_path = os.path.join(paper_dir, "paper", "paper.md")
                else:
                    for root, _, files in os.walk(paper_dir):
                        for f in files:
                            if f.endswith(".md"):
                                md_path = os.path.join(root, f)
                                break
                        if md_path:
                            break
            except Exception as e:
                logger.error(f"marker-pdf execution failed for {pdf_path}: {e}")
        else:
            logger.warning("marker_single binary not found in virtual environment.")
    # Try parsing equations from the converted markdown file
    if md_path and os.path.exists(md_path):
        try:
            logger.info(f"Parsing equations from converted Markdown file: {md_path}")
            with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                md_text = f.read()
            equations = extract_equations_from_markdown(md_text)
            if equations:
                return equations
            logger.warning(f"No equations found in Markdown {md_path}.")
        except Exception as e:
            logger.error(f"Failed to parse equations from markdown {md_path}: {e}")
    logger.warning(f"No equations extracted from {pdf_path} using marker-pdf")
    return []
def parse_paper_equations(paper_directory: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parses PDF or HTML in directory to extract equations and reference counts.
    Parameters
    ----------
    paper_directory : str
        Path to the directory containing the paper files.
    Returns
    -------
    Tuple[List[Dict[str, Any]], str]
        List of parsed equations and a string indicating the extraction source.
    """
    paper_id = os.path.basename(os.path.normpath(paper_directory))
    pdf_path = _find_pdf_path(paper_directory)
    md_path = _find_markdown_path(paper_directory)

    # Try HTML Extraction First
    html_files = [f for f in os.listdir(paper_directory) if f.endswith(".html")]
    if html_files:
        html_path = os.path.join(paper_directory, html_files[0])
        logger.info(f"Trying HTML extraction for {html_path}")
        equations = extract_equations_from_html(html_path)
        if equations:
            logger.info(f"Extracted {len(equations)} equations from HTML {html_path}")
            
            needs_pdf_fallback = any("[Image Equation:" in eq.get("equation_content", "") for eq in equations)
            if needs_pdf_fallback:
                logger.info("Image equations found in HTML. Falling back to PDF to extract placeholders.")
                if os.path.exists(pdf_path):
                    pdf_equations = extract_equations_from_pdf(pdf_path)
                    pdf_eq_dict = {str(eq["equation_num"]): eq for eq in pdf_equations}
                    
                    for eq in equations:
                        if "[Image Equation:" in eq.get("equation_content", ""):
                            eq_num = str(eq["equation_num"])
                            if eq_num in pdf_eq_dict:
                                eq["equation_content"] = pdf_eq_dict[eq_num]["equation_content"]
                                eq["context_note"] += " (Formula extracted from PDF fallback)"
                else:
                    logger.warning(f"PDF {pdf_path} not found for fallback extraction.")

            # Filter out any equations that are STILL image placeholders (because they failed PDF fallback)
            initial_count = len(equations)
            equations = [eq for eq in equations if "[Image Equation:" not in eq.get("equation_content", "")]
            filtered_count = initial_count - len(equations)
            if filtered_count > 0:
                logger.info(f"Skipped {filtered_count} image equations that could not be extracted. Kept {len(equations)} equations.")

            write_html_context_backup(paper_id, html_path, equations)
            write_previous_page_context_backup(paper_id, equations, pdf_path)
            
            source_msg = "HTML Extraction (with PDF fallback)" if needs_pdf_fallback else "HTML Extraction"
            return equations, source_msg
        else:
            logger.info(f"No equations found in HTML. Falling back to PDF.")
    # Try PDF Extraction
    if os.path.exists(pdf_path):
        logger.info(f"Using PDF extraction for {pdf_path}")
        equations = extract_equations_from_pdf(pdf_path)
        logger.info(f"Extracted {len(equations)} equations from PDF {pdf_path}")
        if not html_files:
            write_markdown_context_backup(paper_id, md_path, equations)
        write_previous_page_context_backup(paper_id, equations, pdf_path)
        return equations, "PDF Extraction"
        
    logger.error(f"No valid .pdf files found in {paper_directory}")
    return [], "None"
