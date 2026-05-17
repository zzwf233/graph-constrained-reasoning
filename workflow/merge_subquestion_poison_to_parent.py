import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup_injected_triples(base_graph: List[List[str]], injected_groups: List[List[List[str]]]) -> List[List[str]]:
    base_seen = {tuple(tri) for tri in base_graph}
    seen = set()
    uniq = []
    for triples in injected_groups:
        for tri in triples:
            if not isinstance(tri, list) or len(tri) < 3:
                continue
            key = tuple(tri)
            if key in base_seen or key in seen:
                continue
            seen.add(key)
            uniq.append(list(tri))
    return uniq


def normalize_path(path: Any) -> List[List[str]]:
    if not isinstance(path, list) or not path:
        return []
    normalized = []
    for tri in path:
        if not isinstance(tri, (list, tuple)) or len(tri) < 3:
            return []
        normalized.append([str(tri[0]), str(tri[1]), str(tri[2])])
    return normalized


def is_valid_path(path: Any) -> bool:
    return bool(normalize_path(path))


def dedup_paths(path_groups: List[List[List[List[str]]]], max_paths: int = 0) -> List[List[List[str]]]:
    seen = set()
    uniq = []
    for paths in path_groups:
        for path in paths:
            if not is_valid_path(path):
                continue
            normalized_path = normalize_path(path)
            key = tuple(tuple(tri) for tri in normalized_path)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(normalized_path)
            if max_paths > 0 and len(uniq) >= max_paths:
                return uniq
    return uniq


def derive_paths_from_injected_triples(row: Dict[str, Any], max_paths: int = 0) -> List[List[List[str]]]:
    triples = row.get("injected_triples", []) or []
    q_entities = {str(x) for x in row.get("q_entity", []) if str(x).strip()}
    candidates = {str(x) for x in row.get("adversarial_candidates", []) if str(x).strip()}
    if not triples or not q_entities or not candidates:
        return []

    outgoing = defaultdict(list)
    for tri in triples:
        if isinstance(tri, list) and len(tri) >= 3:
            h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
            outgoing[h].append([h, r, t])

    paths = []
    for q in q_entities:
        for first in outgoing.get(q, []):
            if first[2] in candidates:
                paths.append([first])
                if max_paths > 0 and len(paths) >= max_paths:
                    return paths
            for second in outgoing.get(first[2], []):
                if second[2] in candidates:
                    paths.append([first, second])
                    if max_paths > 0 and len(paths) >= max_paths:
                        return paths
    return paths


def dfs_paths_from_triples(
    graph: List[List[str]],
    q_entities: List[str],
    max_length: int,
    max_paths: int,
) -> List[List[List[str]]]:
    outgoing = defaultdict(list)
    for tri in graph:
        if isinstance(tri, (list, tuple)) and len(tri) >= 3:
            h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
            outgoing[h].append([h, r, t])

    paths = []
    seen = set()

    def visit(node: str, path: List[List[str]]):
        if max_paths > 0 and len(paths) >= max_paths:
            return
        if len(path) >= max_length:
            return
        for tri in outgoing.get(node, []):
            next_path = path + [tri]
            key = tuple(tuple(x) for x in next_path)
            if key not in seen:
                seen.add(key)
                paths.append(next_path)
                if max_paths > 0 and len(paths) >= max_paths:
                    return
            visit(tri[2], next_path)

    for entity in q_entities:
        visit(str(entity), [])
        if max_paths > 0 and len(paths) >= max_paths:
            break
    return paths


def build_clean_paths(row: Dict[str, Any], index_path_length: int, max_paths: int) -> List[List[List[str]]]:
    cached_paths = row.get("paths", []) or []
    if isinstance(cached_paths, list) and cached_paths:
        return dedup_paths([cached_paths], max_paths=max_paths)

    graph = row.get("graph", []) or []
    q_entities = row.get("q_entity", []) or []
    if not graph or not q_entities:
        return []

    return dfs_paths_from_triples(graph, q_entities, index_path_length, max_paths)


def count_attack_paths(paths: List[List[List[str]]], targets: List[str]) -> int:
    target_set = {str(x).strip().lower() for x in targets if str(x).strip()}
    if not target_set:
        return 0
    count = 0
    for path in paths:
        tails = {str(tri[2]).strip().lower() for tri in path if len(tri) >= 3}
        if tails & target_set:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file", required=True, help="Original parent-level QA jsonl")
    parser.add_argument("--poisoned_subq_file", required=True, help="Poisoned sub-question jsonl")
    parser.add_argument("--output_file", required=True, help="Merged parent-level poisoned jsonl")
    parser.add_argument("--poison_position", choices=["front", "back"], default="front")
    parser.add_argument(
        "--path_mode",
        choices=["off", "attack_only", "attack_front", "attack_mix"],
        default="off",
        help=(
            "How to populate the optional paths field used by GCR trie construction. "
            "off keeps the attack as graph poisoning only; attack_only constrains "
            "decoding to injected attack paths; attack_front prepends attack paths "
            "before any existing paths; attack_mix combines injected paths with "
            "bounded clean DFS paths so poisoning has competition."
        ),
    )
    parser.add_argument(
        "--max_attack_paths",
        type=int,
        default=256,
        help="Maximum injected attack paths to keep per parent sample. Use 0 for no cap.",
    )
    parser.add_argument(
        "--clean_index_path_length",
        type=int,
        default=2,
        help="DFS path length used to synthesize clean paths when path_mode=attack_mix.",
    )
    parser.add_argument(
        "--max_clean_paths",
        type=int,
        default=256,
        help="Maximum clean paths to include for path_mode=attack_mix. Use 0 for no cap.",
    )
    args = parser.parse_args()

    original_rows = load_jsonl(args.original_file)
    subq_rows = load_jsonl(args.poisoned_subq_file)

    injections = defaultdict(list)
    injected_path_groups = defaultdict(list)
    candidates = defaultdict(list)
    sub_attacks = defaultdict(list)

    for row in subq_rows:
        pid = str(row.get("parent_id", row.get("id", "")))
        if not pid:
            continue
        injected = row.get("injected_triples", []) or []
        if injected:
            injections[pid].append(injected)
        injected_paths = row.get("injected_paths", []) or []
        if not injected_paths:
            injected_paths = derive_paths_from_injected_triples(row, max_paths=args.max_attack_paths)
        if injected_paths:
            injected_path_groups[pid].append(injected_paths)
        for cand in row.get("adversarial_candidates", []) or []:
            if isinstance(cand, str) and cand.strip():
                candidates[pid].append(cand)
        attack_meta = row.get("attack_meta", {}) or {}
        if row.get("is_poisoned") and row.get("poison_target"):
            candidates[pid].append(str(row["poison_target"]))
        if attack_meta or row.get("is_poisoned"):
            sub_attacks[pid].append(
                {
                    "id": row.get("id"),
                    "sub_id": row.get("sub_id"),
                    "question": row.get("question"),
                    "dep_prev_sub_id": row.get("dep_prev_sub_id"),
                    "dep_type": row.get("dep_type"),
                    "needs_prev_answer": row.get("needs_prev_answer", False),
                    "is_poisoned": bool(row.get("is_poisoned", False)),
                    "poison_target": row.get("poison_target"),
                    "poison_pivot": row.get("poison_pivot"),
                    "injected_triples": row.get("injected_triples", []) or [],
                    **attack_meta,
                }
            )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    written = 0
    injected_samples = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in original_rows:
            out = dict(row)
            pid = str(out.get("id", ""))
            original_graph = out.get("graph", []) or []
            injected_triples = dedup_injected_triples(original_graph, injections.get(pid, []))
            if args.poison_position == "front":
                merged_graph = injected_triples + [list(tri) for tri in original_graph]
            else:
                merged_graph = [list(tri) for tri in original_graph] + injected_triples
            if injected_triples:
                injected_samples += 1
            out["graph"] = merged_graph
            out["injected_triples"] = injected_triples
            injected_paths = dedup_paths(injected_path_groups.get(pid, []), max_paths=args.max_attack_paths)
            if args.path_mode != "off" and injected_paths:
                if args.path_mode == "attack_only":
                    out["paths"] = injected_paths
                elif args.path_mode == "attack_mix":
                    clean_paths = build_clean_paths(row, args.clean_index_path_length, args.max_clean_paths)
                    out["paths"] = dedup_paths([injected_paths, clean_paths], max_paths=0)
                else:
                    original_paths = out.get("paths", []) or []
                    if isinstance(original_paths, list):
                        out["paths"] = injected_paths + original_paths
                    else:
                        out["paths"] = injected_paths
            if injected_paths:
                out["injected_paths"] = injected_paths

            cand_out = []
            seen_cands = set()
            for cand in candidates.get(pid, []):
                key = cand.strip().lower()
                if key and key not in seen_cands:
                    seen_cands.add(key)
                    cand_out.append(cand)
            out["adversarial_candidates"] = cand_out
            out["attack_meta"] = {
                "status": "merged_subquestion_poison",
                "num_subquestions": len(sub_attacks.get(pid, [])),
                "num_injected": len(injected_triples),
                "num_injected_paths": len(injected_paths),
                "is_poisoned": bool(injected_triples),
                "poison_targets": [
                    x.get("poison_target")
                    for x in sub_attacks.get(pid, [])
                    if x.get("is_poisoned") and x.get("poison_target")
                ],
                "path_mode": args.path_mode,
                "sub_attacks": sub_attacks.get(pid, []),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved parent-level poisoned dataset to: {args.output_file}")
    print(f"Parent samples: {written}; samples with injected triples: {injected_samples}")


if __name__ == "__main__":
    main()
