import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.eval_poison_metrics import attack_target_match, extract_answer_text, normalize


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(path: str) -> Dict[str, dict]:
    return {str(row["id"]): row for row in load_jsonl(path)}


def iter_poison_targets(row: dict) -> Iterable[str]:
    target = row.get("poison_target")
    if isinstance(target, str) and target.strip():
        yield target
    attack_meta = row.get("attack_meta", {}) or {}
    if isinstance(attack_meta, dict):
        target = attack_meta.get("poison_target") or attack_meta.get("target")
        if isinstance(target, str) and target.strip():
            yield target


def prediction_answers(row: dict, topk: int) -> List[str]:
    preds = row.get("prediction", [])
    if isinstance(preds, str):
        preds = preds.splitlines()
    if topk > 0:
        preds = preds[:topk]
    return [extract_answer_text(p) for p in preds if extract_answer_text(p)]


def prediction_text(row: dict, topk: int) -> str:
    preds = row.get("prediction", [])
    if isinstance(preds, str):
        preds = preds.splitlines()
    if topk > 0:
        preds = preds[:topk]
    return "\n".join(str(p) for p in preds)


def triple_string(tri: List[str]) -> str:
    return f"{tri[0]} -> {tri[1]} -> {tri[2]}"


def parent_front_has_poison(parent_row: dict, front_n: int = 20) -> bool:
    injected = {tuple(tri) for tri in parent_row.get("injected_triples", []) or []}
    graph = parent_row.get("graph", []) or []
    return any(tuple(tri) in injected for tri in graph[:front_n] if isinstance(tri, list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", required=True)
    parser.add_argument("--poison_file", required=True, help="poisoned subquestion jsonl")
    parser.add_argument("--parent_file", default="", help="merged parent poisoned jsonl")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--output_file", default="")
    args = parser.parse_args()

    predictions = load_predictions(args.predict_file)
    poison_rows = load_jsonl(args.poison_file)
    parent_rows = {str(row.get("id")): row for row in load_jsonl(args.parent_file)} if args.parent_file else {}

    stats = collections.Counter()
    by_dep = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)

    for row in poison_rows:
        if not row.get("is_poisoned"):
            stats["not_poisoned"] += 1
            continue

        sid = str(row.get("id", ""))
        pid = str(row.get("parent_id", sid))
        dep_type = str(row.get("dep_type", "none"))
        targets = list(dict.fromkeys(iter_poison_targets(row)))
        if not targets:
            stats["missing_target"] += 1
            continue

        pred_row = predictions.get(sid) or predictions.get(pid)
        if pred_row is None:
            stats["missing_prediction"] += 1
            by_dep[dep_type]["missing_prediction"] += 1
            continue

        target = targets[0]
        answers = prediction_answers(pred_row, args.topk)
        text = prediction_text(pred_row, args.topk)
        gold = [str(x) for x in row.get("answer", [])]
        injected = row.get("injected_triples", []) or []

        target_hit = any(attack_target_match(ans, target, "exact") for ans in answers)
        gold_hit = any(any(attack_target_match(ans, g, "exact") for g in gold) for ans in answers)
        target_in_text = normalize(target) in normalize(text)
        pivot_in_text = "m.piv_" in text
        exact_poison_edge = any(triple_string(tri) in text for tri in injected if isinstance(tri, list) and len(tri) >= 3)
        parent_front = parent_front_has_poison(parent_rows.get(pid, {}))

        stats["total_poisoned"] += 1
        by_dep[dep_type]["total_poisoned"] += 1
        for name, value in [
            ("target_hit", target_hit),
            ("gold_hit", gold_hit),
            ("target_in_text", target_in_text),
            ("pivot_in_text", pivot_in_text),
            ("exact_poison_edge", exact_poison_edge),
            ("parent_front_has_poison", parent_front),
        ]:
            if value:
                stats[name] += 1
                by_dep[dep_type][name] += 1

        bucket = "hit" if target_hit else "miss"
        if len(examples[bucket]) < args.examples:
            examples[bucket].append(
                {
                    "parent_id": pid,
                    "id": sid,
                    "question": row.get("question"),
                    "dep_type": dep_type,
                    "needs_prev_answer": row.get("needs_prev_answer", False),
                    "rule": (row.get("attack_meta") or {}).get("rule"),
                    "poison_target": target,
                    "gold": gold,
                    "prediction_answers": answers[: args.topk],
                    "target_in_text": target_in_text,
                    "pivot_in_text": pivot_in_text,
                    "exact_poison_edge": exact_poison_edge,
                    "parent_front_has_poison": parent_front,
                    "first_injected_triples": injected[:4],
                }
            )

    def pct(count, total):
        return round(count * 100 / total, 4) if total else 0.0

    total = stats["total_poisoned"]
    summary = {
        "counts": dict(stats),
        "rates": {
            "target_hit_rate": pct(stats["target_hit"], total),
            "gold_hit_rate": pct(stats["gold_hit"], total),
            "target_in_text_rate": pct(stats["target_in_text"], total),
            "pivot_in_text_rate": pct(stats["pivot_in_text"], total),
            "exact_poison_edge_rate": pct(stats["exact_poison_edge"], total),
            "parent_front_has_poison_rate": pct(stats["parent_front_has_poison"], total),
        },
        "by_dep_type": {
            dep: {
                **dict(counter),
                "target_hit_rate": pct(counter["target_hit"], counter["total_poisoned"]),
                "exact_poison_edge_rate": pct(counter["exact_poison_edge"], counter["total_poisoned"]),
            }
            for dep, counter in by_dep.items()
        },
        "examples": examples,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
