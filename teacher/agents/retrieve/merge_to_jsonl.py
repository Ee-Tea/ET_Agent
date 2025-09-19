import argparse
import json
import os
from typing import Dict, Any, List, Tuple


def load_success_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("results", [])
    norm: List[Dict[str, Any]] = []
    for it in items:
        norm.append({
            "question": it.get("question"),
            "answer": it.get("answer"),
            "reference": it.get("reference"),
            "meta": {
                "source": os.path.basename(path),
                "overall_score": it.get("overall_score"),
                "timestamp": it.get("timestamp"),
                "source_file": it.get("source_file"),
            }
        })
    return norm


def load_txt_wrapped_json(path: str) -> List[Dict[str, Any]]:
    # file contains a single big JSON array-like with objects separated; parse via json.loads
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # If file is not a strict JSON array, try to wrap into [ ... ]
    # Many of these files look like JSON objects separated by commas/newlines.
    # Ensure it becomes a valid JSON array before parsing.
    if not text.startswith("["):
        # Attempt to turn into an array: add brackets if appears to be multiple JSON objects
        # Keep existing commas if present
        candidate = text
        # If it starts with { and ends with }, and contains multiple top-level objects separated by '},',
        # wrap into [ ... ] safely.
        if candidate.startswith("{") and candidate.endswith("}"):
            candidate = f"[{candidate}]"
        text = candidate
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to split by lines and parse objects one by one
        items: List[Dict[str, Any]] = []
        buf = ""
        depth = 0
        for ch in text:
            buf += ch
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        items.append(json.loads(buf))
                    except Exception:
                        pass
                    buf = ""
        data = items

    norm: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        data = [data]
    for it in data:
        if not isinstance(it, dict):
            continue
        norm.append({
            "question": it.get("question"),
            "answer": it.get("answer"),
            "reference": it.get("reference"),
            "meta": {
                "source": os.path.basename(path),
                "overall_score": it.get("overall_score"),
                "timestamp": it.get("timestamp"),
            }
        })
    return norm


def dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        key = (q, a)
        if not q or not a:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two datasets into a deduplicated JSONL")
    parser.add_argument("--success_json", default=os.path.join("teacher", "agents", "retrieve", "data", "success_questions_20250918_193640.json"))
    parser.add_argument("--txt_json", default=os.path.join("teacher", "agents", "retrieve", "data", "쓸만한 골든 셋.txt"))
    parser.add_argument("--out", default=os.path.join("teacher", "agents", "retrieve", "data", "merged_success_questions.jsonl"))
    args = parser.parse_args()

    a = load_success_json(args.success_json)
    b = load_txt_wrapped_json(args.txt_json)

    merged = dedup(a + b)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_jsonl(args.out, merged)
    print(f"[OK] Wrote {len(merged)} unique items to {args.out}")


if __name__ == "__main__":
    main()


