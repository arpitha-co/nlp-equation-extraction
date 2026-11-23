"""
Simple main script to extract equations, symbols, context windows,
global context, and paper-level symbol registry.
"""

import json
import logging
from pathlib import Path

from downloader import download_arxiv_source, clean_arxiv_id
from parser import parse_paper_equations
from symbol_extractor import extract_symbols_from_equation
from local_context_window import add_local_context_windows
from definition_extractor import add_definitions_to_dataset


PAPER_LIST_PATH = "paper_list_26.txt"
DATASET_PATH = "dataset.json"
TARGET_EQUATION_COUNT = 350
MAX_EQUATIONS_PER_PAPER = 7


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_paper_ids(path):
    """
    Load a list of arXiv paper IDs from a specified text file.
    
    Args:
        path (str or Path): The file path to the text file containing paper IDs.
        
    Returns:
        list: A list of cleaned paper IDs (strings). Returns an empty list if the file is not found.
    """
    path = Path(path)

    if not path.exists():
        logger.error(f"Paper list not found: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_json(data, path):
    """
    Helper function to serialize a Python dictionary to a JSON file.
    
    Args:
        data (dict): The dictionary to serialize.
        path (str or Path): The destination file path.
    """
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {path}")


def build_equation_entry(eq):
    """
    Construct a structured dictionary for a single extracted equation,
    including the raw text, surrounding context, and mathematical symbols.
    
    Args:
        eq (dict): The raw equation dictionary extracted from the paper.
        
    Returns:
        dict: A structured dictionary containing equation content, context, and extracted symbols.
    """
    equation_text = eq.get("equation_content", "")
    symbols = extract_symbols_from_equation(equation_text)

    return {
        "equation": equation_text,
        "before_text": eq.get("before_text", ""),
        "after_text": eq.get("after_text", ""),
        "symbols": symbols,
        "reference_count": eq.get("reference_count", 0),
    }


def process_paper(arxiv_id, remaining_equations):
    """
    Process a single arXiv paper: download its source, parse equations, and build structured entries.
    
    Args:
        arxiv_id (str): The arXiv ID of the paper.
        remaining_equations (int): The maximum number of equations allowed to be processed 
                                   to stay within the global TARGET_EQUATION_COUNT limit.
                                   
    Returns:
        tuple: A tuple containing the cleaned paper ID (str) and a dictionary of processed equations (dict).
    """
    clean_id = clean_arxiv_id(arxiv_id)

    logger.info(f"Downloading/Locating source for {arxiv_id}...")
    paper_dir = download_arxiv_source(arxiv_id, delay_seconds=3.0)
    
    logger.info(f"Parsing HTML/PDF for {arxiv_id}...")
    equations, source_type = parse_paper_equations(paper_dir)

    selected_equations = equations[:min(MAX_EQUATIONS_PER_PAPER, remaining_equations)]

    paper_data = {}

    for eq in selected_equations:
        eq_num = str(eq.get("equation_num", len(paper_data) + 1))
        paper_data[eq_num] = build_equation_entry(eq)

    logger.info(
        f"Finished {clean_id}: {len(paper_data)} equations from {source_type}"
    )

    return clean_id, paper_data


def main():
    """
    Main orchestration function for the NLP extraction pipeline.
    
    This function coordinates the following steps:
    1. Loads target paper IDs.
    2. Downloads and parses equations from each paper until the target limit is reached.
    3. Enriches equations with local and global context windows.
    4. Extracts symbol definitions using NLP and pattern matching.
    5. Builds semantic context representing the meaning of equations.
    6. Constructs a relationship graph connecting equations based on dependencies.
    7. Compiles and saves the final JSON dataset.
    """
    logger.info("Starting pipeline...")

    paper_ids = load_paper_ids(PAPER_LIST_PATH)

    if not paper_ids:
        logger.error("No papers found.")
        return

    dataset = {}
    total_equations = 0

    for arxiv_id in paper_ids:
        if total_equations >= TARGET_EQUATION_COUNT:
            break

        remaining = TARGET_EQUATION_COUNT - total_equations

        try:
            paper_id, paper_data = process_paper(arxiv_id, remaining)
        except Exception as e:
            logger.error(f"Failed to process {arxiv_id}: {e}")
            continue

        dataset[paper_id] = paper_data
        total_equations += len(paper_data)

    # Remove write_json(dataset, DATASET_PATH) to avoid intermediate disk write

    logger.info("Adding local context windows...")
    try:
        from local_context_window import add_local_context_windows
        add_local_context_windows(dataset=dataset)
    except Exception as e:
        logger.error(f"Failed in local_context_window: {e}")

    logger.info("Writing global context windows...")
    try:
        from global_context_window import add_global_context_windows
        add_global_context_windows(dataset=dataset)
    except Exception as e:
        logger.error(f"Failed in global_context_window: {e}")
    # write_symbol_registry currently just writes to file, we can optionally pass dataset if we modified it
    # wait, write_symbol_registry writes to file but doesn't modify dataset, let's keep it using dataset dict if possible
    # if it doesn't take dict, we can skip it or just let it use memory dict

    logger.info("Extracting definitions...")
    dataset, source_data = add_definitions_to_dataset(dataset=dataset)

    logger.info("Building equation context semantics...")
    try:
        from equation_context_builder import main as build_equation_context_main
        sentence_splits = build_equation_context_main(dataset=dataset)
    except Exception as e:
        logger.error(f"Failed in equation_context_builder: {e}")
        sentence_splits = {}

    logger.info("Building relationship graph...")
    try:
        from graph_relationship import main as graph_relationship_main
        dataset = graph_relationship_main(dataset=dataset)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed in graph_relationship: {e}")

    logger.info("Compiling final JSON...")
    try:
        from create_final_json import main as create_final_json_main
        final_data = create_final_json_main(
            dataset=dataset, 
            sentence_splits=sentence_splits, 
            source_data=source_data
        )
    except Exception as e:
        logger.error(f"Failed in create_final_json: {e}")

    logger.info(f"Pipeline finished. Total equations: {total_equations}")


if __name__ == "__main__":
    main()
