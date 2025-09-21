import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Iterable, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def load_json_results(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    detailed = data.get('detailed_results', [])
    items: List[Dict[str, Any]] = []
    for item in detailed:
        # Normalize to a uniform shape similar to JSONL rows
        items.append({
            'question': item.get('question'),
            'answer': item.get('answer'),
            'reference': item.get('reference'),
            'metric_scores': item.get('metric_scores'),
            'overall_score': item.get('overall_score'),
            'timestamp': item.get('timestamp'),
            'meta': {
                'source_file': os.path.basename(path)
            }
        })
    return items


def deduplicate_by_question(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        q = it.get('question')
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(it)
    return out


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write('\n')


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print('Usage: python merge_jsonl_and_json.py <input.jsonl> <input.json>')
        return 1

    jsonl_path = argv[1]
    json_path = argv[2]

    if not os.path.exists(jsonl_path):
        print(f'File not found: {jsonl_path}')
        return 2
    if not os.path.exists(json_path):
        print(f'File not found: {json_path}')
        return 2

    jsonl_items = load_jsonl(jsonl_path)
    json_items = load_json_results(json_path)

    before_counts = {
        'jsonl': len(jsonl_items),
        'json': len(json_items),
    }

    merged = deduplicate_by_question([*jsonl_items, *json_items])

    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'merged_success_combined_{ts}.jsonl')
    write_jsonl(out_path, merged)

    print('Merged output:', out_path)
    print('Counts:')
    print('  input jsonl:', before_counts['jsonl'])
    print('  input json:', before_counts['json'])
    print('  merged unique by question:', len(merged))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))


