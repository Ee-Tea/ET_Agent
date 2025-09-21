import json
import sys
import os
from datetime import datetime


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_all_success(detailed_results: list) -> list:
    filtered = []
    for item in detailed_results:
        metric_scores = item.get('metric_scores', {})
        if not isinstance(metric_scores, dict) or not metric_scores:
            continue
        all_success = True
        for metric in metric_scores.values():
            # Expect a dict with a boolean 'success' field
            if not isinstance(metric, dict) or metric.get('success') is not True:
                all_success = False
                break
        if all_success:
            filtered.append(item)
    return filtered


def deduplicate_by_question(items: list) -> list:
    seen_questions = set()
    deduped = []
    for item in items:
        question = item.get('question')
        if question and question not in seen_questions:
            seen_questions.add(question)
            deduped.append(item)
    return deduped


def main(argv: list) -> int:
    if len(argv) < 3:
        print("Usage: python merge_filter_success.py <file1.json> <file2.json> [more.json...]")
        return 1

    input_files = argv[1:]
    all_filtered = []
    per_file_counts = {}

    for path in input_files:
        try:
            data = read_json_file(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return 2

        detailed_results = data.get('detailed_results', [])
        filtered = filter_all_success(detailed_results)
        all_filtered.extend(filtered)
        per_file_counts[os.path.basename(path)] = len(filtered)

    combined = deduplicate_by_question(all_filtered)

    # Build output object
    out_obj = {
        "source_files": [os.path.basename(p) for p in input_files],
        "total_filtered_per_file": per_file_counts,
        "total_combined": len(combined),
        "detailed_results": combined,
    }

    # Output path in the same directory as the first input file
    out_dir = os.path.dirname(os.path.abspath(input_files[0]))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f"retrieve_deepeval_results_success_merged_{ts}.json")

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 3

    # Print summary for the caller
    print("Merged filtered results written to:", out_path)
    for k, v in per_file_counts.items():
        print(f"  {k}: {v} items")
    print("  total_combined:", len(combined))

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))


