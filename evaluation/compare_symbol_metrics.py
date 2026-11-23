#!/usr/bin/env python3
"""Compare annotated definitions/meanings against pipeline output.

The script reads:
  - an annotated Excel workbook such as "Annotate symbols.xlsx"
  - a generated JSON file such as "final.json"

It reports coverage/recall and precision, then saves simple visualizations.
Definition extraction is evaluated at symbol-row level. Meaning extraction is
evaluated at equation level when the workbook contains a meaning column such as
"Meaning" or "Equation Meaning".
No pandas/openpyxl dependency is required; the workbook is read through the
standard XLSX XML files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass
class ComparedSymbol:
    paper_id: str
    equation_no: str
    symbol: str
    annotated_definition: str
    generated_definition: str
    present_in_output: bool
    resolved: bool
    correct: bool
    match_reason: str


@dataclass
class ComparedMeaning:
    paper_id: str
    equation_no: str
    annotated_meaning: str
    generated_meaning: str
    present_in_output: bool
    resolved: bool
    correct: bool
    match_reason: str


MEANING_COLUMNS = (
    "Meaning",
    "Equation Meaning",
    "Annotated Meaning",
    "Equation Description",
    "Equation Interpretation",
)


def excel_col_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 1
    number = 0
    for char in match.group(1):
        number = number * 26 + ord(char) - 64
    return number


def read_shared_strings(xlsx: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", XLSX_NS):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS)))
    return strings


def workbook_sheet_targets(xlsx: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

    sheets = {}
    for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[rel_id]
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheets[name] = target
    return sheets


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as xlsx:
        shared_strings = read_shared_strings(xlsx)
        sheets = workbook_sheet_targets(xlsx)
        if sheet_name not in sheets:
            available = ", ".join(sheets)
            raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {available}")

        root = ET.fromstring(xlsx.read(sheets[sheet_name]))
        raw_rows = []
        for row in root.findall("a:sheetData/a:row", XLSX_NS):
            values: dict[int, str] = {}
            max_col = 0
            for cell in row.findall("a:c", XLSX_NS):
                col = excel_col_number(cell.attrib.get("r", "A1"))
                max_col = max(max_col, col)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", XLSX_NS)
                inline_node = cell.find("a:is", XLSX_NS)

                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or 0)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(text.text or "" for text in inline_node.findall(".//a:t", XLSX_NS))
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                values[col] = value
            if values:
                raw_rows.append([values.get(i, "") for i in range(1, max_col + 1)])

    if not raw_rows:
        return []

    headers = raw_rows[0]
    return [dict(zip(headers, row)) for row in raw_rows[1:]]


def normalize_arxiv_id(value: str) -> str:
    value = (value or "").strip()
    if not value or "." in value:
        return value
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 8:
        return f"{digits[:4]}.{digits[4:]}"
    return value


def normalize_symbol(value: str) -> str:
    value = (value or "").strip()
    value = value.replace("′", "prime")
    value = value.replace("^\\prime", "^{prime}")
    return value


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    replacements = {
        "ℏ": "hbar",
        "wavefunction": "wave function",
        "tranform": "transform",
        "amplitute": "amplitude",
        "func-\ntion": "function",
        "func tion": "function",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def definition_tokens(value: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "both",
        "by",
        "corresponding",
        "defined",
        "for",
        "given",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    return {token for token in normalize_text(value).split() if token and token not in stop_words}


def is_correct_text(annotated: str, generated: str, label: str = "definition") -> tuple[bool, str]:
    annotated_norm = normalize_text(annotated)
    generated_norm = normalize_text(generated)

    if not annotated_norm or not generated_norm:
        return False, f"blank annotated/generated {label}"
    if annotated_norm == generated_norm:
        return True, "exact normalized match"
    if annotated_norm in generated_norm or generated_norm in annotated_norm:
        return True, f"{label} contains match"

    annotated_tokens = definition_tokens(annotated)
    generated_tokens = definition_tokens(generated)
    if not annotated_tokens or not generated_tokens:
        return False, "no comparable tokens"

    overlap = annotated_tokens & generated_tokens
    recall = len(overlap) / len(annotated_tokens)
    precision = len(overlap) / len(generated_tokens)
    if recall >= 0.8:
        return True, f"token recall {recall:.2f}"
    if recall >= 0.5 and precision >= 0.5 and len(overlap) >= 2:
        return True, f"token overlap r={recall:.2f}, p={precision:.2f}"
    return False, f"low token overlap r={recall:.2f}, p={precision:.2f}"


def is_correct_definition(annotated: str, generated: str) -> tuple[bool, str]:
    return is_correct_text(annotated, generated, "definition")


def is_correct_meaning(annotated: str, generated: str) -> tuple[bool, str]:
    return is_correct_text(annotated, generated, "meaning")


def filter_rows(rows: Iterable[dict[str, str]], scope: str) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        symbol = (row.get("Symbol") or "").strip()
        if not symbol:
            continue
        if scope == "defined" and not (row.get("Symbol Definition") or "").strip():
            continue
        if scope == "annotated-evidence":
            has_definition = bool((row.get("Symbol Definition") or "").strip())
            has_line = bool((row.get("Definition Line") or "").strip())
            match_type = (row.get("Match Type") or "").strip().lower()
            if not (has_definition or has_line or (match_type and match_type != "not found")):
                continue
        filtered.append(row)
    return filtered


def dedupe_rows(rows: list[dict[str, str]], dedupe_by: str) -> list[dict[str, str]]:
    if dedupe_by == "none":
        return rows

    seen = set()
    result = []
    for row in rows:
        paper_id = normalize_arxiv_id(row.get("arXiv ID", ""))
        equation_no = (row.get("Equation No") or "").strip()
        symbol = normalize_symbol(row.get("Symbol", ""))

        if dedupe_by == "symbol":
            key = (symbol,)
        elif dedupe_by == "paper-symbol":
            key = (paper_id, symbol)
        elif dedupe_by == "paper-equation-symbol":
            key = (paper_id, equation_no, symbol)
        else:
            raise ValueError(f"Unknown dedupe mode: {dedupe_by}")

        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def lookup_generated_definition(
    generated: dict,
    paper_id: str,
    equation_no: str,
    symbol: str,
) -> tuple[bool, str]:
    symbols = generated.get(paper_id, {}).get(equation_no, {}).get("symbols", {})
    if symbol in symbols:
        return True, str(symbols[symbol])

    normalized_symbol = normalize_text(symbol)
    for candidate, definition in symbols.items():
        if normalize_text(candidate) == normalized_symbol:
            return True, str(definition)

    return False, ""


def lookup_generated_meaning(
    generated: dict,
    paper_id: str,
    equation_no: str,
) -> tuple[bool, str]:
    equation = generated.get(paper_id, {}).get(equation_no)
    if not equation:
        return False, ""
    return True, str(equation.get("meaning", ""))


def compare_symbols(rows: list[dict[str, str]], generated: dict) -> list[ComparedSymbol]:
    compared = []
    for row in rows:
        paper_id = normalize_arxiv_id(row.get("arXiv ID", ""))
        equation_no = (row.get("Equation No") or "").strip()
        symbol = normalize_symbol(row.get("Symbol", ""))
        annotated_definition = (row.get("Symbol Definition") or "").strip()

        present, generated_definition = lookup_generated_definition(generated, paper_id, equation_no, symbol)
        resolved = present and bool(generated_definition.strip())
        correct, reason = is_correct_definition(annotated_definition, generated_definition) if resolved else (False, "unresolved")

        compared.append(
            ComparedSymbol(
                paper_id=paper_id,
                equation_no=equation_no,
                symbol=symbol,
                annotated_definition=annotated_definition,
                generated_definition=generated_definition,
                present_in_output=present,
                resolved=resolved,
                correct=correct,
                match_reason=reason,
            )
        )
    return compared


def row_value_case_insensitive(row: dict[str, str], candidates: Iterable[str]) -> tuple[str | None, str]:
    lower_to_key = {str(key).strip().lower(): key for key in row}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key, (row.get(key) or "").strip()
    return None, ""


def meaning_rows(
    rows: list[dict[str, str]],
    candidates: Iterable[str] = MEANING_COLUMNS,
) -> tuple[list[dict[str, str]], str | None]:
    detected_column = None
    by_equation: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        column, annotated_meaning = row_value_case_insensitive(row, candidates)
        if column is None:
            continue
        detected_column = detected_column or column
        paper_id = normalize_arxiv_id(row.get("arXiv ID", ""))
        equation_no = (row.get("Equation No") or "").strip()
        if not paper_id or not equation_no:
            continue

        key = (paper_id, equation_no)
        if key not in by_equation:
            by_equation[key] = {
                "arXiv ID": paper_id,
                "Equation No": equation_no,
                "Annotated Meaning": annotated_meaning,
            }
        elif not by_equation[key]["Annotated Meaning"] and annotated_meaning:
            by_equation[key]["Annotated Meaning"] = annotated_meaning

    return list(by_equation.values()), detected_column


def compare_meanings(rows: list[dict[str, str]], generated: dict) -> list[ComparedMeaning]:
    compared = []
    for row in rows:
        paper_id = normalize_arxiv_id(row.get("arXiv ID", ""))
        equation_no = (row.get("Equation No") or "").strip()
        annotated_meaning = (row.get("Annotated Meaning") or "").strip()

        present, generated_meaning = lookup_generated_meaning(generated, paper_id, equation_no)
        resolved = present and bool(generated_meaning.strip())
        correct, reason = is_correct_meaning(annotated_meaning, generated_meaning) if resolved else (False, "unresolved")

        compared.append(
            ComparedMeaning(
                paper_id=paper_id,
                equation_no=equation_no,
                annotated_meaning=annotated_meaning,
                generated_meaning=generated_meaning,
                present_in_output=present,
                resolved=resolved,
                correct=correct,
                match_reason=reason,
            )
        )
    return compared


def summarize(compared) -> dict[str, float | int]:
    total = len(compared)
    resolved = sum(item.resolved for item in compared)
    correct = sum(item.correct for item in compared)
    return {
        "Total symbols examined": total,
        "Symbols successfully resolved": resolved,
        "Coverage (Recall)": resolved / total if total else 0.0,
        "Correct among resolved symbols": correct,
        "Precision": correct / resolved if resolved else 0.0,
    }


def write_details_csv(compared: list[ComparedSymbol], output_dir: Path) -> Path:
    path = output_dir / "symbol_comparison_details.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "paper_id",
                "equation_no",
                "symbol",
                "annotated_definition",
                "generated_definition",
                "present_in_output",
                "resolved",
                "correct",
                "match_reason",
            ]
        )
        for item in compared:
            writer.writerow(
                [
                    item.paper_id,
                    item.equation_no,
                    item.symbol,
                    item.annotated_definition,
                    item.generated_definition,
                    item.present_in_output,
                    item.resolved,
                    item.correct,
                    item.match_reason,
                ]
            )
    return path


def write_meaning_details_csv(compared: list[ComparedMeaning], output_dir: Path) -> Path:
    path = output_dir / "meaning_comparison_details.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "paper_id",
                "equation_no",
                "annotated_meaning",
                "generated_meaning",
                "present_in_output",
                "resolved",
                "correct",
                "match_reason",
            ]
        )
        for item in compared:
            writer.writerow(
                [
                    item.paper_id,
                    item.equation_no,
                    item.annotated_meaning,
                    item.generated_meaning,
                    item.present_in_output,
                    item.resolved,
                    item.correct,
                    item.match_reason,
                ]
            )
    return path


def save_metric_charts(
    compared,
    metrics: dict[str, float | int],
    output_dir: Path,
    prefix: str = "definition",
    title: str = "Definition Extraction",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    total = int(metrics["Total symbols examined"])
    resolved = int(metrics["Symbols successfully resolved"])
    correct = int(metrics["Correct among resolved symbols"])
    unresolved = total - resolved
    incorrect_resolved = resolved - correct

    plt.figure(figsize=(7, 4.5))
    labels = ["Coverage / Recall", "Precision"]
    values = [float(metrics["Coverage (Recall)"]), float(metrics["Precision"])]
    bars = plt.bar(labels, values, color=["#00035B", "#FF8C00"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{title} Metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    plt.tight_layout()
    path = output_dir / f"{prefix}_metrics_bar.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(6.2, 4.8))
    pie_values = [correct, incorrect_resolved, unresolved]
    pie_labels = ["Correct resolved", "Incorrect resolved", "Unresolved"]
    pie_colors = ["#00035B", "#FF8C00", "#A5A5A5"]
    plt.pie(
        pie_values,
        labels=pie_labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle=90,
        colors=pie_colors,
    )
    plt.title(f"{title} Outcomes")
    plt.tight_layout()
    path = output_dir / f"{prefix}_outcome_pie.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    by_paper = defaultdict(Counter)
    for item in compared:
        if item.correct:
            by_paper[item.paper_id]["correct"] += 1
        elif item.resolved:
            by_paper[item.paper_id]["incorrect"] += 1
        else:
            by_paper[item.paper_id]["unresolved"] += 1

    if by_paper:
        papers = list(by_paper)
        correct_counts = [by_paper[paper]["correct"] for paper in papers]
        incorrect_counts = [by_paper[paper]["incorrect"] for paper in papers]
        unresolved_counts = [by_paper[paper]["unresolved"] for paper in papers]

        height = max(4.8, 0.45 * len(papers))
        plt.figure(figsize=(9, height))
        left_incorrect = correct_counts
        left_unresolved = [a + b for a, b in zip(correct_counts, incorrect_counts)]
        plt.barh(papers, correct_counts, color="#00035B", label="Correct resolved")
        plt.barh(papers, incorrect_counts, left=left_incorrect, color="#FF8C00", label="Incorrect resolved")
        plt.barh(papers, unresolved_counts, left=left_unresolved, color="#A5A5A5", label="Unresolved")
        plt.xlabel("Item count")
        plt.title(f"{title} Outcomes by Paper")
        plt.legend(loc="upper right")
        plt.tight_layout()
        path = output_dir / f"{prefix}_outcomes_by_paper.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def save_combined_metric_chart(
    metric_sets: list[tuple[str, dict[str, float | int]]],
    output_dir: Path,
) -> Path | None:
    if len(metric_sets) < 2:
        return None

    labels = [label for label, _metrics in metric_sets]
    coverage = [float(metrics["Coverage (Recall)"]) for _label, metrics in metric_sets]
    precision = [float(metrics["Precision"]) for _label, metrics in metric_sets]
    positions = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    plt.bar([pos - width / 2 for pos in positions], coverage, width=width, color="#00035B", label="Coverage / Recall")
    plt.bar([pos + width / 2 for pos in positions], precision, width=width, color="#FF8C00", label="Precision")
    plt.xticks(list(positions), labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Extraction Metrics Summary")
    plt.legend()
    for pos, value in zip(positions, coverage):
        plt.text(pos - width / 2, value + 0.02, f"{value:.1%}", ha="center", fontsize=9)
    for pos, value in zip(positions, precision):
        plt.text(pos + width / 2, value + 0.02, f"{value:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    path = output_dir / "extraction_metrics_summary.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def print_metric_table(metrics: dict[str, float | int], title: str, item_label: str) -> None:
    print(f"\n{title}")
    print("Metric\tValue")
    print(f"Total {item_label} examined\t{metrics['Total symbols examined']}")
    print(f"{item_label.capitalize()} successfully resolved\t{metrics['Symbols successfully resolved']}")
    print(f"Coverage (Recall)\t{metrics['Coverage (Recall)']:.2%}")
    print(f"Correct among resolved {item_label}\t{metrics['Correct among resolved symbols']}")
    print(f"Precision\t{metrics['Precision']:.2%}")


def print_legacy_metric_table(metrics: dict[str, float | int]) -> None:
    print("\nDefinition Extraction")
    print("Metric\tValue")
    print(f"Total symbols examined\t{metrics['Total symbols examined']}")
    print(f"Symbols successfully resolved\t{metrics['Symbols successfully resolved']}")
    print(f"Coverage (Recall)\t{metrics['Coverage (Recall)']:.2%}")
    print(f"Correct among resolved symbols\t{metrics['Correct among resolved symbols']}")
    print(f"Precision\t{metrics['Precision']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare annotated definitions and meanings with final.json pipeline output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Scope options:
              all                 every non-empty Symbol row
              defined             rows with a non-empty Symbol Definition
              annotated-evidence  rows with a definition, definition line, or non-'not found' match type

            Dedupe options:
              none                  count each paper/equation/symbol row
              symbol                count each raw symbol string once
              paper-symbol          count each paper/symbol once
              paper-equation-symbol count each paper/equation/symbol once
            """
        ),
    )
    parser.add_argument("--annotated", default="Annotate symbols.xlsx", type=Path)
    parser.add_argument("--generated", default="final.json", type=Path)
    parser.add_argument("--sheet", default="Symbol Definitions")
    parser.add_argument(
        "--meaning-columns",
        default=",".join(MEANING_COLUMNS),
        help="Comma-separated candidate column names for annotated equation meanings.",
    )
    parser.add_argument("--scope", choices=["all", "defined", "annotated-evidence"], default="all")
    parser.add_argument(
        "--dedupe-by",
        choices=["none", "symbol", "paper-symbol", "paper-equation-symbol"],
        default="none",
    )
    parser.add_argument("--output-dir", default=Path("metric_outputs"), type=Path)
    args = parser.parse_args()

    raw_rows = read_xlsx_sheet(args.annotated, args.sheet)
    definition_rows = filter_rows(raw_rows, args.scope)
    definition_rows = dedupe_rows(definition_rows, args.dedupe_by)

    with args.generated.open(encoding="utf-8") as handle:
        generated = json.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    compared_definitions = compare_symbols(definition_rows, generated)
    definition_metrics = summarize(compared_definitions)
    definition_details_path = write_details_csv(compared_definitions, args.output_dir)
    chart_paths = save_metric_charts(
        compared_definitions,
        definition_metrics,
        args.output_dir,
        prefix="definition",
        title="Definition Extraction",
    )

    meaning_column_candidates = [column.strip() for column in args.meaning_columns.split(",") if column.strip()]
    extracted_meaning_rows, detected_meaning_column = meaning_rows(raw_rows, meaning_column_candidates)

    compared_meanings = []
    meaning_metrics = None
    meaning_details_path = None
    if extracted_meaning_rows:
        compared_meanings = compare_meanings(extracted_meaning_rows, generated)
        meaning_metrics = summarize(compared_meanings)
        meaning_details_path = write_meaning_details_csv(compared_meanings, args.output_dir)
        chart_paths.extend(
            save_metric_charts(
                compared_meanings,
                meaning_metrics,
                args.output_dir,
                prefix="meaning",
                title="Meaning Extraction",
            )
        )

    combined_path = save_combined_metric_chart(
        [("Definition", definition_metrics)]
        + ([("Meaning", meaning_metrics)] if meaning_metrics is not None else []),
        args.output_dir,
    )
    if combined_path is not None:
        chart_paths.append(combined_path)

    print(f"Annotated workbook: {args.annotated}")
    print(f"Generated JSON: {args.generated}")
    print(f"Sheet: {args.sheet}")
    print(f"Scope: {args.scope}")
    print(f"Dedupe: {args.dedupe_by}")
    print_legacy_metric_table(definition_metrics)
    if meaning_metrics is not None:
        print_metric_table(meaning_metrics, "Meaning Extraction", "equations")
        print(f"Meaning ground-truth column: {detected_meaning_column}")
    else:
        print(
            "\nMeaning Extraction\n"
            "Skipped: no annotated meaning column found. Add a column named one of: "
            + ", ".join(meaning_column_candidates)
        )

    print(f"\nDefinition details CSV: {definition_details_path}")
    if meaning_details_path is not None:
        print(f"Meaning details CSV: {meaning_details_path}")
    print("Visualizations:")
    for path in chart_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
