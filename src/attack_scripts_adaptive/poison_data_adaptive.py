import argparse
import copy
import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    import openai
except ModuleNotFoundError:
    openai = None

DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_API_BASE = "https://api.siliconflow.cn/v1"


def str2bool(x):
    return str(x).lower() in ["1", "true", "yes", "y"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--rule_file", type=str, default="")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["ours", "rand", "rand_local", "paper_rand"], default="ours")
    parser.add_argument(
        "--reuse_attack_file",
        type=str,
        default="",
        help=(
            "Optional poisoned jsonl whose poison_target/adversarial_candidates "
            "should be reused. This fixes attack candidates across budget sweeps."
        ),
    )
    parser.add_argument(
        "--require_reuse_attack",
        type=str2bool,
        default=False,
        help="If true, do not generate new attacks for rows missing from --reuse_attack_file.",
    )

    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--inject_top_k", type=int, default=3)
    parser.add_argument("--api_max_tokens", type=int, default=256)

    parser.add_argument("--hop_repeat", type=int, default=100)
    parser.add_argument("--single_hop_repeat", type=int, default=200)
    parser.add_argument("--front_boost", type=float, default=1.4)
    parser.add_argument("--hop_boost_if_type_match", type=float, default=1.5)
    parser.add_argument("--hop_boost_if_type_mismatch", type=float, default=0.7)
    parser.add_argument("--target_top_k", type=int, default=8)
    parser.add_argument("--budget_k", type=int, default=0)
    parser.add_argument(
        "--per_answer_budget_k",
        type=int,
        default=0,
        help="If > 0, keep at most this many injected triples for each selected adversarial answer.",
    )
    parser.add_argument(
        "--strict_target_filter",
        type=str2bool,
        default=True,
        help="Reject poison targets that look like aliases, broader forms, or near-overlaps of gold answers.",
    )
    parser.add_argument(
        "--gold_overlap_threshold",
        type=float,
        default=0.5,
        help="Token-overlap threshold for strict poison-target filtering against gold answers.",
    )
    parser.add_argument("--expand_single_hop", type=str2bool, default=True)
    parser.add_argument("--include_direct_single_hop", type=str2bool, default=True)
    parser.add_argument("--single_hop_bridge_relation", type=str, default="poison.answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dependency_aware",
        type=str2bool,
        default=True,
        help="Use the previous poisoned sub-question target as the anchor for dependent sub-questions.",
    )
    return parser.parse_args()


def canonical_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    return "paper_rand" if mode == "rand" else mode


def normalize_rule(rule_text: str) -> Optional[Tuple[str, Optional[str]]]:
    if not rule_text:
        return None
    s = str(rule_text).strip().strip("()")
    if not s:
        return None
    for sep in ["->", "=>", ",", "|", "\t"]:
        if sep in s:
            parts = [x.strip() for x in s.split(sep) if x.strip()]
            if parts:
                return parts[0], parts[1] if len(parts) > 1 else None
    return s, None


def load_rog_rules(rule_file: str) -> Dict[str, Dict[str, Any]]:
    qid2rule: Dict[str, Dict[str, Any]] = {}
    if not rule_file:
        return qid2rule
    if not os.path.exists(rule_file):
        print(f"[Warning] rule_file not found: {rule_file}. Fallback to default relation.attack.")
        return qid2rule
    with open(rule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item.get("id", "")).strip()
            if not qid:
                continue
            raw = item.get("rules", item.get("prediction", []))
            if isinstance(raw, list):
                if not raw:
                    continue
                raw = raw[0]
            if not isinstance(raw, str):
                continue
            parsed = normalize_rule(raw)
            if parsed:
                qid2rule[qid] = {
                    "rels": parsed,
                    "rule_source": item.get("rule_source", "unknown"),
                    "is_reliable": bool(item.get("is_reliable", True)),
                }
    return qid2rule


def load_reuse_attacks(reuse_attack_file: str) -> Dict[str, Dict[str, Any]]:
    attacks: Dict[str, Dict[str, Any]] = {}
    if not reuse_attack_file:
        return attacks
    if not os.path.exists(reuse_attack_file):
        raise FileNotFoundError(f"reuse_attack_file not found: {reuse_attack_file}")
    with open(reuse_attack_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("id", row.get("parent_id", "")))
            if not sid:
                continue
            meta = row.get("attack_meta", {}) or {}
            target = row.get("poison_target") or meta.get("poison_target") or meta.get("target")
            candidates = row.get("adversarial_candidates", []) or []
            if target:
                merged = [str(target)] + [str(c) for c in candidates if str(c).strip()]
                deduped = []
                seen = set()
                for cand in merged:
                    key = normalize_text(cand)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    deduped.append(cand)
                attacks[sid] = {
                    "is_poisoned": bool(row.get("is_poisoned", False)),
                    "poison_target": str(target),
                    "adversarial_candidates": deduped,
                    "attack_meta": meta,
                }
            elif row.get("is_poisoned") is False or meta:
                attacks[sid] = {
                    "is_poisoned": False,
                    "poison_target": None,
                    "adversarial_candidates": [],
                    "attack_meta": meta,
                }
    return attacks


def normalize_text(s: Any) -> str:
    return str(s).strip().lower()


def canonical_text(s: Any) -> str:
    text = normalize_text(s)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def content_tokens(s: Any) -> set:
    stop = {
        "of",
        "and",
        "in",
        "on",
        "at",
        "for",
        "to",
        "by",
        "from",
        "with",
        "language",
        "city",
        "country",
        "state",
        "province",
        "county",
        "person",
        "people",
        "united",
        "america",
    }
    return {tok for tok in canonical_text(s).split() if tok and tok not in stop}


def infer_question_answer_type(question: str) -> str:
    q = normalize_text(question)
    if any(k in q for k in ["language", "speak", "spoken"]):
        return "language"
    if any(k in q for k in ["profession", "occupation", "job", "office", "position"]) or re.search(r"\bwhat did .+ do\b", q):
        return "profession"
    if any(k in q for k in ["where", "country", "city", "located", "location", "state"]):
        return "location"
    if any(k in q for k in ["when", "year", "date", "born", "died"]):
        return "time"
    if any(k in q for k in ["who", "president", "actor", "singer", "player", "person"]):
        return "person"
    if any(k in q for k in ["team", "club", "university", "organization", "company"]):
        return "organization"
    return "entity"


def infer_entity_type_hint(name: str) -> str:
    n = normalize_text(name)
    if "language" in n or n in ["english", "french", "spanish", "german", "arabic"]:
        return "language"
    if any(k in n for k in ["representative", "governor", "president", "minister", "speaker", "senator", "office", "title"]):
        return "profession"
    if any(k in n for k in ["city", "country", "river", "island", "sea", "state", "province"]):
        return "location"
    if any(k in n for k in ["fc", "club", "company", "inc", "ltd", "university", "college"]):
        return "organization"
    if any(k in n for k in ["president", "mr", "mrs", "dr", "king", "queen"]):
        return "person"
    if any(ch.isdigit() for ch in n):
        return "time"
    return "entity"


def extract_answer_texts(item: Dict[str, Any]) -> List[str]:
    ans = item.get("answer", item.get("answers", ""))
    if isinstance(ans, list):
        return [str(x) for x in ans if str(x).strip()] or ["UnknownAnswer"]
    if ans is None:
        return ["UnknownAnswer"]
    return [str(ans)]


def extract_answer_text(item: Dict[str, Any]) -> str:
    return extract_answer_texts(item)[0]


def answer_key_set(original_answer: Any) -> set:
    if isinstance(original_answer, list):
        return {normalize_text(x) for x in original_answer if str(x).strip()}
    return {normalize_text(original_answer)}


def too_close_to_gold(
    candidate: str,
    item: Dict[str, Any],
    rels: Tuple[str, Optional[str]],
    overlap_threshold: float = 0.5,
) -> bool:
    cand = canonical_text(candidate)
    if not cand:
        return True
    cand_tokens = content_tokens(candidate)
    ctype = infer_entity_type_hint(candidate)
    qtype = infer_question_answer_type(item.get("question", ""))

    for gold in extract_answer_texts(item):
        gold_norm = canonical_text(gold)
        if not gold_norm:
            continue
        if cand == gold_norm or cand in gold_norm or gold_norm in cand:
            return True

        gold_tokens = content_tokens(gold)
        if not cand_tokens or not gold_tokens:
            continue
        overlap = cand_tokens & gold_tokens
        if not overlap:
            continue
        containment = max(
            len(overlap) / max(1, len(cand_tokens)),
            len(overlap) / max(1, len(gold_tokens)),
        )
        if containment >= overlap_threshold:
            return True

        # Same-type answers that share a content token are often aliases or
        # broader/narrower variants, e.g. English Language vs Jamaican English.
        if qtype != "entity" and ctype == qtype:
            return True

        # Relation hints can make a broad answer look plausible but still too
        # close to the gold answer for a clean adversarial target.
        if score_by_relation_hint(candidate, rels, qtype) > 0 and containment >= 0.34:
            return True
    return False


def is_machine_id(value: str) -> bool:
    return str(value).startswith(("m.", "g."))


def is_noisy_surface(value: str) -> bool:
    v = normalize_text(value)
    if not v or len(v) < 2 or len(v) > 60:
        return True
    noisy_bits = [
        " - speaker",
        " - topic",
        "official site",
        "homepage",
        "www.",
        "http",
        "isbn",
    ]
    return any(bit in v for bit in noisy_bits)


def is_quality_target(
    candidate: str,
    item: Dict[str, Any],
    rels: Tuple[str, Optional[str]],
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> bool:
    cand = str(candidate).strip()
    key = normalize_text(cand)
    if not key or is_machine_id(cand) or is_noisy_surface(cand):
        return False
    if key in answer_key_set(item.get("answer", item.get("answers", []))):
        return False
    if strict_target_filter and too_close_to_gold(cand, item, rels, gold_overlap_threshold):
        return False

    topic_keys = {normalize_text(x) for x in item.get("q_entity", []) if str(x).strip()}
    if any(key == t or key in t or t in key for t in topic_keys if t):
        return False

    qtype = infer_question_answer_type(item.get("question", ""))
    ctype = infer_entity_type_hint(cand)
    if qtype != "entity" and ctype != qtype and score_by_relation_hint(cand, rels, qtype) <= 0:
        return False
    return True


def score_by_relation_hint(candidate: str, rels: Tuple[str, Optional[str]], qtype: str) -> float:
    rel_text = " ".join([r for r in rels if r]).lower()
    ctype = infer_entity_type_hint(candidate)
    score = 0.0
    if qtype != "entity" and ctype == qtype:
        score += 3.0
    if any(x in rel_text for x in ["location", "country", "place", "city"]) and ctype == "location":
        score += 2.5
    if any(x in rel_text for x in ["language"]) and ctype == "language":
        score += 2.5
    if any(x in rel_text for x in ["profession", "position", "title", "office"]) and ctype == "profession":
        score += 2.5
    if any(x in rel_text for x in ["date", "year", "time", "born", "died"]) and ctype == "time":
        score += 2.5
    if any(x in rel_text for x in ["person", "spouse", "parent", "child"]) and ctype == "person":
        score += 2.5
    return score


def rank_target_candidates(
    cands: List[str],
    question: str,
    original_answer: Any,
    rels: Tuple[str, Optional[str]],
    strict_item: Optional[Dict[str, Any]] = None,
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> List[str]:
    gold_keys = answer_key_set(original_answer)
    qtype = infer_question_answer_type(question)

    uniq = []
    seen = set()
    for c in cands:
        key = normalize_text(c)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    def _score(c: str) -> float:
        key = normalize_text(c)
        if key in gold_keys:
            return -1e9
        if strict_item is not None and strict_target_filter and too_close_to_gold(c, strict_item, rels, gold_overlap_threshold):
            return -1e9
        score = 0.0
        ctype = infer_entity_type_hint(c)
        if qtype != "entity" and ctype == qtype:
            score += 4.0
        elif qtype != "entity" and ctype != "entity" and ctype != qtype:
            score -= 1.5
        score += score_by_relation_hint(c, rels, qtype)
        if 2 <= len(c.strip()) <= 40:
            score += 0.5
        return score

    return sorted(uniq, key=_score, reverse=True)


def filter_quality_targets(
    cands: List[str],
    item: Dict[str, Any],
    rels: Tuple[str, Optional[str]],
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> List[str]:
    out = []
    seen = set()
    for cand in cands:
        key = normalize_text(cand)
        if key in seen or not is_quality_target(cand, item, rels, strict_target_filter, gold_overlap_threshold):
            continue
        seen.add(key)
        out.append(cand)
    return out


def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_array_from_text(text: str):
    data = safe_json_loads(text)
    if isinstance(data, list):
        return data
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return None
    return safe_json_loads(m.group(0))


def build_readable_entity_pool(item: Dict[str, Any], blocked_values: Optional[List[str]] = None) -> List[str]:
    readable = []
    fallback = []
    blocked = {normalize_text(x) for x in (blocked_values or []) if str(x).strip()}
    for tri in item.get("graph", []):
        if not isinstance(tri, list) or len(tri) < 3:
            continue
        h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
        if r == "type.object.name" and t:
            readable.append(t)
        for x in (h, t):
            if x and not x.startswith("m.") and not x.startswith("g."):
                fallback.append(x)

    def dedup(xs):
        out, seen = [], set()
        for x in xs:
            key = normalize_text(x)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    readable = dedup(readable)
    fallback = dedup(fallback)
    pool = readable if readable else fallback
    filtered = [x for x in pool if normalize_text(x) not in blocked]
    if filtered:
        return filtered
    return [x for x in fallback if normalize_text(x) not in blocked]


def build_rule_candidate_pool(
    item: Dict[str, Any],
    rels: Tuple[str, Optional[str]],
    blocked_values: Optional[List[str]] = None,
    head_entities_override: Optional[List[str]] = None,
) -> List[str]:
    rel1, rel2 = rels
    blocked = {normalize_text(x) for x in (blocked_values or []) if str(x).strip()}
    head_entities = head_entities_override if head_entities_override is not None else item.get("q_entity", [])
    q_entities = {str(x) for x in head_entities if str(x).strip()}
    graph = [tri for tri in item.get("graph", []) if isinstance(tri, list) and len(tri) >= 3]

    primary = []
    fallback = []
    if rel2:
        mids = {str(t) for h, r, t in graph if str(h) in q_entities and str(r) == rel1}
        primary.extend(str(t) for h, r, t in graph if str(h) in mids and str(r) == rel2)
        fallback.extend(str(t) for _, r, t in graph if str(r) == rel2)
    else:
        primary.extend(str(t) for h, r, t in graph if str(h) in q_entities and str(r) == rel1)
        fallback.extend(str(t) for _, r, t in graph if str(r) == rel1)

    def clean(candidates: List[str]) -> List[str]:
        out, seen = [], set()
        for cand in candidates:
            key = normalize_text(cand)
            if not key or key in blocked or key in seen or cand.startswith(("m.", "g.")):
                continue
            seen.add(key)
            out.append(cand)
        return out

    return clean(primary) or clean(fallback)


def find_natural_pivots(item: Dict[str, Any], rel1: str, head_entities_override: Optional[List[str]] = None) -> List[str]:
    head_entities = head_entities_override if head_entities_override is not None else item.get("q_entity", [])
    q_entities = {str(x) for x in head_entities if str(x).strip()}
    pivots, seen = [], set()
    for tri in item.get("graph", []):
        if not isinstance(tri, list) or len(tri) < 3:
            continue
        h, r, t = str(tri[0]), str(tri[1]), str(tri[2])
        if h in q_entities and r == rel1 and t not in seen:
            seen.add(t)
            pivots.append(t)
    return pivots


def pick_non_gold(cands: List[str], gold: Any, default_val: str = "Unknown Entity") -> str:
    gold_keys = answer_key_set(gold)
    pool = [c for c in cands if normalize_text(c) not in gold_keys]
    if pool:
        return random.choice(pool)
    if cands:
        return random.choice(cands)
    return default_val


def build_adversarial_answer_set(
    candidate_entities: List[str],
    question: str,
    original_answer: Any,
    selected_target: str,
    rels: Tuple[str, Optional[str]],
    top_k: int = 5,
    strict_item: Optional[Dict[str, Any]] = None,
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> List[str]:
    ranked = rank_target_candidates(
        candidate_entities,
        question,
        original_answer,
        rels,
        strict_item=strict_item,
        strict_target_filter=strict_target_filter,
        gold_overlap_threshold=gold_overlap_threshold,
    )
    gold_keys = answer_key_set(original_answer)
    out, seen = [], set()
    for c in [selected_target] + ranked:
        k = normalize_text(c)
        if not k or k in seen or k in gold_keys:
            continue
        if strict_item is not None and strict_target_filter and too_close_to_gold(c, strict_item, rels, gold_overlap_threshold):
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max(1, top_k):
            break
    return out


def llm_plan_target(
    client,
    model_name: str,
    question: str,
    original_answer: str,
    candidate_entities: List[str],
    rels: Tuple[str, Optional[str]],
    temperature: float,
    api_max_tokens: int,
    strict_item: Optional[Dict[str, Any]] = None,
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> str:
    ranked = rank_target_candidates(
        candidate_entities,
        question,
        original_answer,
        rels,
        strict_item=strict_item,
        strict_target_filter=strict_target_filter,
        gold_overlap_threshold=gold_overlap_threshold,
    )
    if not ranked:
        return "FallbackEntity"
    if client is None:
        return ranked[0]

    system_prompt = (
        "You are helping to construct adversarial KGQA poisoning. "
        "Pick one plausible but wrong target answer from candidate list. "
        "Never return the gold answer. Return exactly one string."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Gold answer: {original_answer}\n"
        f"Rule: {rels}\n"
        f"Candidates: {ranked[:12]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=api_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        picked = (resp.choices[0].message.content or "").strip()
        for cand in ranked:
            if normalize_text(cand) == normalize_text(picked):
                return cand
    except Exception:
        pass
    return ranked[0]


def llm_generate_candidates(
    client,
    model_name: str,
    question: str,
    original_answer: str,
    candidate_entities: List[str],
    rels: Tuple[str, Optional[str]],
    temperature: float,
    api_max_tokens: int,
    num_candidates: int,
    strict_item: Optional[Dict[str, Any]] = None,
    strict_target_filter: bool = True,
    gold_overlap_threshold: float = 0.5,
) -> List[str]:
    ranked = rank_target_candidates(
        candidate_entities,
        question,
        original_answer,
        rels,
        strict_item=strict_item,
        strict_target_filter=strict_target_filter,
        gold_overlap_threshold=gold_overlap_threshold,
    )
    if client is None:
        return ranked[: max(1, num_candidates)]

    system_prompt = (
        "Generate a JSON array of plausible but wrong answer entities. "
        "Do not include the gold answer. Output JSON array only."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Gold answer: {original_answer}\n"
        f"Rule: {rels}\n"
        f"Candidate pool: {ranked[:30]}\n"
        f"Need {num_candidates} candidates."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=api_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        arr = extract_json_array_from_text(text)
        if isinstance(arr, list):
            out = [str(x).strip() for x in arr if str(x).strip()]
            if out:
                return out[: max(1, num_candidates)]
    except Exception:
        pass
    return ranked[: max(1, num_candidates)]


def instantiate_cand(
    head_entity: str,
    rel1: str,
    rel2: Optional[str],
    cand: str,
    instance_tag: Optional[str] = None,
    pivot_node: Optional[str] = None,
    expand_single_hop: bool = True,
    include_direct_single_hop: bool = True,
    single_hop_bridge_relation: str = "poison.answer",
) -> List[List[str]]:
    def _pivot_id():
        if instance_tag is not None:
            return f"m.piv_{instance_tag}"
        return "m.piv_" + hashlib.md5(str(cand).encode("utf-8")).hexdigest()[:12]

    if rel2:
        if pivot_node:
            return [[pivot_node, rel2, cand]]
        pivot = _pivot_id()
        return [[head_entity, rel1, pivot], [pivot, rel2, cand]]

    triples = []
    if include_direct_single_hop:
        triples.append([head_entity, rel1, cand])
    if expand_single_hop:
        pivot = _pivot_id()
        triples.extend(
            [
                [head_entity, rel1, pivot],
                [pivot, single_hop_bridge_relation, cand],
            ]
        )
    if triples:
        return triples
    return [[head_entity, rel1, cand]]


def instantiate_cand_paths(
    head_entity: str,
    rel1: str,
    rel2: Optional[str],
    cand: str,
    instance_tag: Optional[str] = None,
    pivot_node: Optional[str] = None,
    expand_single_hop: bool = True,
    include_direct_single_hop: bool = True,
    single_hop_bridge_relation: str = "poison.answer",
) -> List[List[List[str]]]:
    def _pivot_id():
        if instance_tag is not None:
            return f"m.piv_{instance_tag}"
        return "m.piv_" + hashlib.md5(str(cand).encode("utf-8")).hexdigest()[:12]

    if rel2:
        if pivot_node:
            return [[[head_entity, rel1, pivot_node], [pivot_node, rel2, cand]]]
        pivot = _pivot_id()
        return [[[head_entity, rel1, pivot], [pivot, rel2, cand]]]

    paths = []
    if include_direct_single_hop:
        paths.append([[head_entity, rel1, cand]])
    if expand_single_hop:
        pivot = _pivot_id()
        paths.append([[head_entity, rel1, pivot], [pivot, single_hop_bridge_relation, cand]])
    if paths:
        return paths
    return [[[head_entity, rel1, cand]]]


def extract_head_entity_mentions(item: Dict[str, Any]) -> List[str]:
    qes = [str(x) for x in item.get("q_entity", []) if str(x).strip()]
    if qes:
        return qes
    graph = item.get("graph", [])
    if graph and isinstance(graph[0], list) and len(graph[0]) >= 1:
        return [str(graph[0][0])]
    return ["m.anchor"]


def get_rule_info_for_item(item: Dict[str, Any], mode: str, qid2rule: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if mode not in ["ours", "rand_local"]:
        return {"rels": ("relation.attack", None), "rule_source": "paper_rand", "is_reliable": True}

    sid = str(item.get("id", item.get("parent_id", "")))
    pid = str(item.get("parent_id", sid))
    return qid2rule.get(
        sid,
        qid2rule.get(
            pid,
            {"rels": (None, None), "rule_source": "missing", "is_reliable": False},
        ),
    )


def dependency_anchor_for_item(item: Dict[str, Any], parent_state: Dict[int, Dict[str, Any]]) -> Optional[List[str]]:
    if not item.get("needs_prev_answer"):
        return None
    dep_prev = item.get("dep_prev_sub_id")
    if dep_prev is None:
        dep_prev = int(item.get("sub_id", 0) or 0) - 1
    try:
        prev_state = parent_state.get(int(dep_prev), {})
    except (TypeError, ValueError):
        prev_state = {}
    anchor = prev_state.get("poison_target") or prev_state.get("poison_pivot")
    if isinstance(anchor, str) and anchor.strip():
        return [anchor.strip()]
    return None


def apply_attack_for_item(
    item: Dict[str, Any],
    mode: str,
    client,
    args,
    rule_info: Dict[str, Any],
    head_entities_override: Optional[List[str]] = None,
    reuse_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = copy.deepcopy(item)
    graph = [list(x) for x in item.get("graph", [])]
    question = item.get("question", "")
    gold_answers = extract_answer_texts(item)
    rels = rule_info.get("rels", (None, None))
    rel1, rel2 = rels
    head_entities = head_entities_override or extract_head_entity_mentions(item)
    if not rule_info.get("is_reliable", False):
        out["is_poisoned"] = False
        out["poison_target"] = None
        out["poison_pivot"] = None
        out["attack_meta"] = {
            "mode": mode,
            "status": "unreliable_rule",
            "rule": [rel1, rel2],
            "rule_source": rule_info.get("rule_source", "missing"),
        }
        return out

    if not rel1 or not head_entities:
        out["is_poisoned"] = False
        out["poison_target"] = None
        out["poison_pivot"] = None
        out["attack_meta"] = {
            "mode": mode,
            "status": "missing_rule_or_start",
            "rule": [rel1, rel2],
            "rule_source": rule_info.get("rule_source", "missing"),
        }
        return out

    strict_target_filter = bool(getattr(args, "strict_target_filter", True))
    gold_overlap_threshold = float(getattr(args, "gold_overlap_threshold", 0.5))

    if reuse_plan is not None:
        if not reuse_plan.get("is_poisoned", False) or not reuse_plan.get("poison_target"):
            out["is_poisoned"] = False
            out["poison_target"] = None
            out["poison_pivot"] = None
            out["attack_meta"] = {
                "mode": mode,
                "status": "reuse_not_poisoned",
                "rule": [rel1, rel2],
                "rule_source": rule_info.get("rule_source", "missing"),
                "reuse_attack": True,
            }
            return out
        target = str(reuse_plan["poison_target"])
        candidates = [str(x) for x in reuse_plan.get("adversarial_candidates", []) if str(x).strip()]
        if not candidates:
            candidates = [target]
    else:
        blocked_values = gold_answers + [str(x) for x in item.get("q_entity", [])] + [str(x) for x in item.get("a_entity", [])]
        entity_pool = build_rule_candidate_pool(
            item,
            rels,
            blocked_values=blocked_values,
            head_entities_override=head_entities,
        )
        entity_pool = filter_quality_targets(
            entity_pool,
            item,
            rels,
            strict_target_filter=strict_target_filter,
            gold_overlap_threshold=gold_overlap_threshold,
        )

        if not entity_pool:
            out["is_poisoned"] = False
            out["poison_target"] = None
            out["poison_pivot"] = None
            out["attack_meta"] = {
                "mode": mode,
                "status": "empty_rule_candidate_pool",
                "rule": [rel1, rel2],
                "rule_source": rule_info.get("rule_source", "missing"),
            }
            return out

        if mode in ["paper_rand", "rand_local"]:
            target = pick_non_gold(entity_pool, gold_answers)
            candidates = build_adversarial_answer_set(
                entity_pool,
                question,
                gold_answers,
                target,
                rels,
                top_k=args.num_candidates,
                strict_item=item,
                strict_target_filter=strict_target_filter,
                gold_overlap_threshold=gold_overlap_threshold,
            )
        else:
            target = llm_plan_target(
                client,
                args.model_name,
                question,
                gold_answers,
                entity_pool,
                rels,
                args.temperature,
                args.api_max_tokens,
                strict_item=item,
                strict_target_filter=strict_target_filter,
                gold_overlap_threshold=gold_overlap_threshold,
            )
            generated = llm_generate_candidates(
                client,
                args.model_name,
                question,
                gold_answers,
                entity_pool,
                rels,
                args.temperature,
                args.api_max_tokens,
                args.num_candidates,
                strict_item=item,
                strict_target_filter=strict_target_filter,
                gold_overlap_threshold=gold_overlap_threshold,
            )
            candidates = build_adversarial_answer_set(
                filter_quality_targets(
                    generated + entity_pool,
                    item,
                    rels,
                    strict_target_filter=strict_target_filter,
                    gold_overlap_threshold=gold_overlap_threshold,
                ),
                question,
                gold_answers,
                target,
                rels,
                top_k=args.num_candidates,
                strict_item=item,
                strict_target_filter=strict_target_filter,
                gold_overlap_threshold=gold_overlap_threshold,
            )
        candidates = filter_quality_targets(
            candidates,
            item,
            rels,
            strict_target_filter=strict_target_filter,
            gold_overlap_threshold=gold_overlap_threshold,
        )

        if not is_quality_target(target, item, rels, strict_target_filter, gold_overlap_threshold):
            if candidates:
                target = candidates[0]

        if not target or not is_quality_target(target, item, rels, strict_target_filter, gold_overlap_threshold):
            out["is_poisoned"] = False
            out["poison_target"] = None
            out["poison_pivot"] = None
            out["attack_meta"] = {
                "mode": mode,
                "status": "invalid_target",
                "rule": [rel1, rel2],
                "rule_source": rule_info.get("rule_source", "missing"),
            }
            return out

    candidates = [
        cand
        for cand in candidates
        if normalize_text(cand) and normalize_text(cand) != normalize_text(target)
    ]
    selected_candidates = [target] + candidates
    selected_candidates = [
        cand
        for i, cand in enumerate(selected_candidates)
        if normalize_text(cand)
        and normalize_text(cand) not in {normalize_text(x) for x in selected_candidates[:i]}
    ]

    natural_pivots = find_natural_pivots(item, rel1, head_entities_override=head_entities) if rel2 else []
    if rel2 and not natural_pivots and not head_entities_override:
        out["is_poisoned"] = False
        out["poison_target"] = None
        out["poison_pivot"] = None
        out["attack_meta"] = {
            "mode": mode,
            "status": "missing_natural_pivot",
            "rule": [rel1, rel2],
            "rule_source": rule_info.get("rule_source", "missing"),
        }
        return out

    base_repeat = args.single_hop_repeat if rel2 is None else args.hop_repeat

    injected = []
    injected_paths = []
    perturbations_by_answer = []
    qtype = infer_question_answer_type(question)
    poison_pivot = None
    base_seen = {tuple(x) for x in graph}
    seen = set()
    seen_paths = set()
    per_answer_budget_k = int(getattr(args, "per_answer_budget_k", 0) or 0)
    for i, cand in enumerate(selected_candidates[: args.inject_top_k]):
        ctype = infer_entity_type_hint(cand)
        boost = args.hop_boost_if_type_match if ctype == qtype else args.hop_boost_if_type_mismatch
        if i < 2:
            boost *= args.front_boost
        repeat = max(1, int(base_repeat * boost))
        raw_triples = []
        raw_paths = []
        for j in range(repeat):
            h = random.choice(head_entities)
            instance_tag = f"{i}_{j}"
            pivot_node = random.choice(natural_pivots) if rel2 and natural_pivots else None
            if i == 0 and j == 0:
                poison_pivot = pivot_node or (f"m.piv_{instance_tag}" if rel2 or args.expand_single_hop else None)
            raw_paths.extend(
                instantiate_cand_paths(
                    h,
                    rel1,
                    rel2,
                    cand,
                    instance_tag=instance_tag,
                    pivot_node=pivot_node,
                    expand_single_hop=args.expand_single_hop,
                    include_direct_single_hop=args.include_direct_single_hop,
                    single_hop_bridge_relation=args.single_hop_bridge_relation,
                )
            )
            raw_triples.extend(
                instantiate_cand(
                    h,
                    rel1,
                    rel2,
                    cand,
                    instance_tag=instance_tag,
                    pivot_node=pivot_node,
                    expand_single_hop=args.expand_single_hop,
                    include_direct_single_hop=args.include_direct_single_hop,
                    single_hop_bridge_relation=args.single_hop_bridge_relation,
                )
            )

        cand_uniq = []
        for tri in raw_triples:
            tp = tuple(tri)
            if tp in base_seen or tp in seen:
                continue
            if per_answer_budget_k > 0 and len(cand_uniq) >= per_answer_budget_k:
                break
            seen.add(tp)
            cand_uniq.append(tri)
            injected.append(tri)

        allowed = base_seen | {tuple(tri) for tri in injected}
        for path in raw_paths:
            path_key = tuple(tuple(tri) for tri in path)
            if path_key in seen_paths:
                continue
            if not all(tuple(tri) in allowed for tri in path):
                continue
            seen_paths.add(path_key)
            injected_paths.append(path)

        perturbations_by_answer.append(
            {
                "answer": cand,
                "num_injected": len(cand_uniq),
                "injected_triples": cand_uniq,
            }
        )

    uniq = []
    seen_uniq = set()
    for tri in injected:
        tp = tuple(tri)
        if tp in base_seen or tp in seen_uniq:
            continue
        seen_uniq.add(tp)
        uniq.append(tri)

    if args.budget_k > 0:
        uniq = uniq[: args.budget_k]
        allowed_triples = base_seen | {tuple(tri) for tri in uniq}
        injected_paths = [
            path
            for path in injected_paths
            if all(tuple(tri) in allowed_triples for tri in path)
        ]

    out["graph"] = uniq + graph
    out["injected_triples"] = uniq
    out["injected_paths"] = injected_paths
    out["perturbations_by_answer"] = perturbations_by_answer
    out["adversarial_candidates"] = selected_candidates[: args.num_candidates]
    out["is_poisoned"] = bool(uniq)
    out["poison_target"] = target if uniq else None
    out["poison_pivot"] = poison_pivot if uniq else None
    out["attack_meta"] = {
        "mode": mode,
        "status": "poisoned" if uniq else "no_new_triples",
        "rule": [rel1, rel2],
        "rule_source": rule_info.get("rule_source", "missing"),
        "target": target,
        "poison_target": target if uniq else None,
        "poison_pivot": poison_pivot if uniq else None,
        "gold_answers": gold_answers,
        "num_injected": len(uniq),
        "inject_top_k": args.inject_top_k,
        "per_answer_budget_k": per_answer_budget_k,
        "reuse_attack": bool(reuse_plan is not None),
        "expand_single_hop": args.expand_single_hop,
        "strict_target_filter": strict_target_filter,
        "gold_overlap_threshold": gold_overlap_threshold,
        "dependency_anchor": head_entities_override[0] if head_entities_override else None,
    }
    return out


def main():
    args = parse_args()
    mode = canonical_mode(args.mode)
    random.seed(args.seed)
    reuse_attacks = load_reuse_attacks(args.reuse_attack_file)
    reuse_only = bool(reuse_attacks) and bool(args.require_reuse_attack)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if mode == "ours" and not api_key and not reuse_only:
        raise ValueError(
            "mode=ours requires API key. Please set --api_key or OPENAI_API_KEY."
        )
    if mode == "ours" and openai is None and not reuse_only:
        raise ValueError("mode=ours requires the openai package. Install it or use MODE=rand_local / MODE=paper_rand.")
    client = openai.OpenAI(api_key=api_key, base_url=args.api_base) if mode == "ours" and api_key and openai is not None else None

    qid2rule = load_rog_rules(args.rule_file)

    with open(args.input_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    indexed_rows = list(enumerate(rows))
    if args.dependency_aware:
        indexed_rows = sorted(
            indexed_rows,
            key=lambda x: (
                str(x[1].get("parent_id", x[1].get("id", ""))),
                int(x[1].get("sub_id", 0) or 0),
                x[0],
            ),
        )

    poisoned_by_index = {}
    parent_state: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for original_index, item in tqdm(indexed_rows, desc=f"Poisoning ({mode})"):
        sid = str(item.get("id", item.get("parent_id", "")))
        pid = str(item.get("parent_id", sid))
        sample_seed = int(hashlib.md5(sid.encode("utf-8")).hexdigest(), 16) % 10_000_000
        random.seed(args.seed + sample_seed)

        rule_info = get_rule_info_for_item(item, mode, qid2rule)
        state = parent_state.setdefault(pid, {})
        head_override = dependency_anchor_for_item(item, state) if args.dependency_aware else None
        reuse_plan = reuse_attacks.get(sid)
        if args.require_reuse_attack and reuse_attacks and reuse_plan is None:
            poisoned = copy.deepcopy(item)
            poisoned["injected_triples"] = []
            poisoned["injected_paths"] = []
            poisoned["adversarial_candidates"] = []
            poisoned["is_poisoned"] = False
            poisoned["poison_target"] = None
            poisoned["poison_pivot"] = None
            poisoned["attack_meta"] = {
                "mode": mode,
                "status": "missing_reuse_attack",
                "reuse_attack": True,
            }
        else:
            poisoned = apply_attack_for_item(
                item,
                mode,
                client,
                args,
                rule_info,
                head_entities_override=head_override,
                reuse_plan=reuse_plan,
            )
        poisoned_by_index[original_index] = poisoned

        try:
            sub_id = int(item.get("sub_id", 0) or 0)
        except (TypeError, ValueError):
            sub_id = 0
        if poisoned.get("is_poisoned"):
            state[sub_id] = {
                "poison_target": poisoned.get("poison_target"),
                "poison_pivot": poisoned.get("poison_pivot"),
            }

    out_rows = [poisoned_by_index[i] for i in range(len(rows))]

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved poisoned dataset to: {args.output_file}")


if __name__ == "__main__":
    main()
