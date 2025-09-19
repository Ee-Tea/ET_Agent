import argparse
import datetime
import glob
import json
import os
from typing import Any, Dict, List, Tuple


def is_all_metrics_success(metric_scores: Dict[str, Any]) -> bool:
    if not isinstance(metric_scores, dict) or not metric_scores:
        return False
    for metric_name, metric in metric_scores.items():
        if not isinstance(metric, dict):
            return False
        if metric.get("success") is not True:
            return False
    return True


def collect_from_file(file_path: str) -> Tuple[List[Dict[str, Any]], int]:
    selected: List[Dict[str, Any]] = []
    total = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read {file_path}: {exc}")
        return selected, total

    detailed_results = data.get("detailed_results", [])
    if not isinstance(detailed_results, list):
        return selected, total

    for item in detailed_results:
        if not isinstance(item, dict):
            continue
        total += 1
        metric_scores = item.get("metric_scores")
        if is_all_metrics_success(metric_scores):
            selected.append({
                "question": item.get("question"),
                "answer": item.get("answer"),
                "reference": item.get("reference"),
                "metric_scores": metric_scores,
                "overall_score": item.get("overall_score"),
                "timestamp": item.get("timestamp"),
                "source_file": os.path.basename(file_path),
            })
    return selected, total


def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("question"), r.get("answer"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect items whose all metric_scores.success are True across JSON result files.")
    default_eval_dir = os.path.join(os.path.dirname(__file__), "eval_goldenset_deepeval")
    parser.add_argument("--dir", default=default_eval_dir, help="Directory to scan (default: eval_goldenset_deepeval under this script)")
    parser.add_argument("--pattern", default="retrieve_deepeval_results_*.json", help="Glob pattern for files (default: retrieve_deepeval_results_*.json)")
    parser.add_argument("--out", default=None, help="Output JSON file path (default: success_questions_<timestamp>.json in --dir)")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    pattern = os.path.join(target_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[INFO] No files matched in {target_dir} (pattern: {args.pattern})")
        return

    print(f"[INFO] Scanning {len(files)} files in {target_dir}")

    aggregated: List[Dict[str, Any]] = []
    total_questions_scanned = 0
    for fp in files:
        selected, total = collect_from_file(fp)
        total_questions_scanned += total
        aggregated.extend(selected)

    aggregated = deduplicate(aggregated)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(target_dir, f"success_questions_{ts}.json")

    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "source_directory": target_dir,
        "file_pattern": args.pattern,
        "source_files": [os.path.basename(f) for f in files],
        "total_files_scanned": len(files),
        "total_questions_scanned": total_questions_scanned,
        "selected_count": len(aggregated),
        "results": aggregated,
    }

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {len(aggregated)} items to {out_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to write output {out_path}: {exc}")


if __name__ == "__main__":
    main()


