import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

try:
    from type4_agents.equivalence_judge_agent import equivalence_judge_agent
    from type4_agents.type4_transform_agent import type4_transform_agent
except ImportError:
    from equivalence_judge_agent import equivalence_judge_agent
    from type4_transform_agent import type4_transform_agent


SUPPORTED_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".txt",
}


def discover_function_files(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    files = [
        path
        for path in input_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def build_transform_prompt(function_source: str) -> str:
    return (
        "You receive as input a function or code snippet:\n"
        f"{ function_source }\n"
        "Your task is to generate a Type-4 code clone: a rewritten version of the input that preserves the same semantics and"
        "functional behavior while differing significantly in syntax, structure, organization, and implementation details."
        "Valid transformations include (but are not limited to):"
        "• Changing control flow structures"
        "• Restructuring logic and data flow"
        "• Reorganizing statements or functions"
        "• Using different language constructs or APIs"
        "• Renaming variables, functions, and types"
        "• Rewriting the algorithm while preserving behavior"
        "Constraints:"
        "• Do not restate or reference the original code"
        "• Do not copy syntactic structures from the input"
        "• Always output fully functional code"
        "• Focus on structural diversity and semantic equivalence"

    )


def build_equivalence_prompt(original_function: str, transformed_function: str) -> str:
    return (
            "You receive as input two code snippets:\n"
            f"• { original_function }: the original function or code snippet\n"
            f"• { transformed_function }: a rewritten version of the original code\n"
            "Your task is to validate whether the transformed code is a correct Type-4 clone of the original code."
            "Specifically, you must verify that:"
            "• The transformed code is semantically equivalent to the original, producing the same outputs and side effects for"
            "all valid inputs."
            "• The transformed code introduces no new errors, bugs, or unintended behaviors."
            "• Any potential bugs or vulnerabilities present in the original code have not been fixed, mitigated, or altered in"
            "the transformed version."
            "• The transformation preserves the intended behavior and logic, even if the syntax, structure, or control flow differ"
            "substantially."
            "You must carefully analyze control flow, data flow, edge cases, and implicit assumptions to ensure equivalence. If any"
            "behavioral discrepancy, bug fix, or semantic deviation is detected, the transformation must be rejected. Constraints:"
            "• Do not suggest improvements or corrections"
            "• Do not rewrite or modify either code snippet"
            "• Do not assume the availability of tests or execution results"
            "• Base your judgment solely on static reasoning about the code"
    )


async def process_file(file_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "status": "ok",
    }

    try:
        original_function = file_path.read_text(encoding="utf-8", errors="replace")
        if not original_function.strip():
            raise ValueError("File is empty.")

        transform_res = await type4_transform_agent.run(
            build_transform_prompt(original_function)
        )

        transformed_function = transform_res.output.transformed_function
        transformation_description = transform_res.output.transformation_description

        assessments: list[dict[str, Any]] = []
        true_votes = 0
        false_votes = 0

        for run_idx in range(1, 4):
            judge_res = await equivalence_judge_agent.run(
                build_equivalence_prompt(original_function, transformed_function)
            )
            equivalent = bool(judge_res.output.semantically_equivalent)
            justification = judge_res.output.justification

            if equivalent:
                true_votes += 1
            else:
                false_votes += 1

            assessments.append(
                {
                    "run": run_idx,
                    "semantically_equivalent": equivalent,
                    "justification": justification,
                }
            )

        final_equivalence = true_votes >= 2

        result.update(
            {
                "original_function": original_function,
                "transformed_function": transformed_function,
                "transformation_description": transformation_description,
                "equivalence_assessments": assessments,
                "equivalence_quorum": {
                    "final_label": final_equivalence,
                    "true_votes": true_votes,
                    "false_votes": false_votes,
                },
            }
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


async def run_batch(input_folder: Path, output_json: Path) -> dict[str, Any]:
    files = discover_function_files(input_folder)
    results: list[dict[str, Any]] = []

    for idx, file_path in enumerate(files, start=1):
        print(f"[INFO] Processing {idx}/{len(files)}: {file_path}")
        record = await process_file(file_path)
        results.append(record)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "input_folder": str(input_folder),
        "output_json": str(output_json),
        "total_files_found": len(files),
        "results": results,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Type-4 transformation and 3x semantic-equivalence quorum."
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing function files (one function per file).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to the output JSON report.",
    )
    return parser.parse_args()


async def _amain() -> None:
    args = parse_args()
    summary = await run_batch(Path(args.input_folder), Path(args.output_json))
    print(
        "[DONE] "
        f"Processed {summary['total_files_found']} files. "
        f"Output: {summary['output_json']}"
    )


if __name__ == "__main__":
    asyncio.run(_amain())
