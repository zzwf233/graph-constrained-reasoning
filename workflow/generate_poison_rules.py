import argparse
import collections
import json
import os
from typing import List, Tuple


def build_rel_stats(graph: List[List[str]], q_entities: List[str]) -> Tuple[collections.Counter, collections.Counter]:
    one_hop = collections.Counter()
    two_hop = collections.Counter()

    outgoing = collections.defaultdict(list)
    for tri in graph:
        if not isinstance(tri, list) or len(tri) < 3:
            continue
        h, r, t = tri[0], tri[1], tri[2]
        outgoing[h].append((r, t))

    for q in q_entities:
        for r1, mid in outgoing.get(q, []):
            one_hop[r1] += 1
            for r2, _ in outgoing.get(mid, []):
                two_hop[(r1, r2)] += 1

    return one_hop, two_hop


def find_truth_rule(graph: List[List[str]], q_entities: List[str], a_entities: List[str], max_hops: int = 2) -> str:
    if not q_entities or not a_entities:
        return ""

    answers = {str(x) for x in a_entities if str(x).strip()}
    outgoing = collections.defaultdict(list)
    for tri in graph:
        if not isinstance(tri, list) or len(tri) < 3:
            continue
        h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
        outgoing[h].append((r, t))

    queue = collections.deque()
    for q in q_entities:
        queue.append((str(q), []))

    visited = set()
    while queue:
        node, rels = queue.popleft()
        if len(rels) >= max_hops:
            continue
        state = (node, tuple(rels))
        if state in visited:
            continue
        visited.add(state)
        for rel, nxt in outgoing.get(node, []):
            new_rels = rels + [rel]
            if nxt in answers:
                return "->".join(new_rels)
            queue.append((nxt, new_rels))
    return ""


def pick_rule(graph: List[List[str]], q_entities: List[str], a_entities: List[str]) -> Tuple[str, str, bool]:
    truth_rule = find_truth_rule(graph, q_entities, a_entities)
    if truth_rule:
        return truth_rule, "truth_path", True

    one_hop, two_hop = build_rel_stats(graph, q_entities)
    if two_hop:
        (r1, r2), _ = two_hop.most_common(1)[0]
        return f"{r1}->{r2}", "fallback_two_hop_frequency", False
    if one_hop:
        r1, _ = one_hop.most_common(1)[0]
        return r1, "fallback_one_hop_frequency", False

    # Fallback: use first relation in graph
    for tri in graph:
        if isinstance(tri, list) and len(tri) >= 2:
            return str(tri[1]), "fallback_first_relation", False
    return "", "no_rule", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="decomposed jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="rules.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    rows = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in rows:
            sid = str(row.get("id", "")).strip()
            pid = str(row.get("parent_id", sid)).strip()
            sub_id = row.get("sub_id", 0)
            if not sid:
                sid = f"{pid}_{sub_id}"
            graph = row.get("graph", [])
            q_entities = row.get("q_entity", []) or []
            a_entities = row.get("a_entity", row.get("answer", [])) or []
            rule, rule_source, is_reliable = pick_rule(graph, q_entities, a_entities)
            out = {
                "id": sid,
                "parent_id": pid,
                "sub_id": sub_id,
                "question": row.get("question", ""),
                "rules": [rule] if rule else [],
                "rule_source": rule_source,
                "is_reliable": is_reliable,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rules to: {args.output_file}")


if __name__ == "__main__":
    main()
