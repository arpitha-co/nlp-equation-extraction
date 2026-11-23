# NLP Equation Extraction Pipeline

This directory contains the core Natural Language Processing pipeline designed to extract mathematical equations from arXiv papers and intelligently enrich them with deep semantic information. The entire architecture is built without the use of Large Language Models (LLMs), relying instead on rule-based heuristics, dependency parsing, and graph-based relationships to guarantee explainability and high precision.

## 🗂️ Project Structure

```text
pipeline/
├── create_final_json.py            # Compiles all processed data into the final structured JSON dataset.
├── definition_extractor.py         # Uses SpaCy POS tagging to map mathematical symbols to their text definitions (e.g., "E" -> "energy").
├── downloader.py                   # Handles downloading of arXiv HTML and PDF files with proper rate limiting.
├── equation_context_builder.py     # Parses linguistic dependencies to identify the overall "meaning" or name of an equation.
├── global_context_window.py        # Extracts broad document context (paragraphs/sections) surrounding equations.
├── graph_relationship.py           # Builds a directed graph to map relationships and dependencies between different equations.
├── local_context_window.py         # Extracts immediate sentence-level context directly adjacent to equations.
├── main.py                         # The master orchestration script that links all modules together.
├── paper_list_26.txt               # A text file containing the target arXiv IDs to process.
├── parser.py                       # The heavy-lifter that extracts numbered equations via HTML parsing, with `marker-pdf` fallback.
├── predefined.json                 # Contains universally predefined physics constants and symbols to assist extraction.
├── requirements.txt                # Lists all necessary Python packages and dependencies.
└── symbol_extractor.py             # Cleans LaTeX strings and extracts individual mathematical variables/symbols.
```

## ⚙️ Architecture and Data Flow

The pipeline operates in a strictly sequential flow, orchestrated entirely by `main.py`. The execution sequence is as follows:

1. **Downloader (`downloader.py`)**: Reads arXiv IDs from `paper_list_26.txt` and fetches the raw HTML (or PDF) document.
2. **Document Parser (`parser.py`)**: Scans the raw document, identifies LaTeX math blocks, and extracts numbered equations and their immediate structural context. Uses `marker-pdf` via subprocess if HTML is unavailable.
3. **Symbol Extractor (`symbol_extractor.py`)**: Cleans the LaTeX formatting and tokenizes the equation to extract individual mathematical symbols (e.g., `\{E, m, c\}`).
4. **Context Windows (`local_` & `global_context_window.py`)**: Analyzes the raw text to build local (sentence-level) and global (section-level) linguistic contexts for each equation.
5. **Definition Extractor (`definition_extractor.py`)**: Uses SpaCy's dependency parser and custom linguistic patterns to scan the context and match extracted symbols to noun phrases, finding their definitions.
6. **Equation Meaning (`equation_context_builder.py`)**: Masks the equations and analyzes surrounding verbs and noun phrases to extract the overarching semantic meaning of the equation (e.g., "mass-energy equivalence").
7. **Graph Relationships (`graph_relationship.py`)**: Analyzes left-hand and right-hand variables, alongside text references (e.g., "Substituting Eq. 1 into Eq. 2") to build mathematical links.
8. **Final JSON (`create_final_json.py`)**: Aggregates all extracted nodes, relationships, and metadata into a clean, hierarchical dataset.

## 🚀 Getting Started

### Prerequisites
Make sure you have all the required libraries installed. You can install them using:

```bash
pip install -r requirements.txt
```

*Note: The pipeline will also attempt to run `marker-pdf` (via the `marker_single` binary) if it needs to process a PDF fallback. Ensure your environment is configured for it if you expect heavy PDF usage.*

### Running the Pipeline
To start the extraction process, simply execute the main orchestrator script:

```bash
python main.py
```

The script will:
- Begin processing the arXiv IDs listed in `paper_list_26.txt`.
- Cache downloaded documents locally to prevent redownloading.
- Log its progress sequentially in the console.
- Output the fully enriched data to `dataset.json`.
