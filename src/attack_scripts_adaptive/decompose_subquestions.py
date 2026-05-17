from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

MULTIHOP_CUES = [
    r"\bwho\b.*\b(whose|that|which)\b",
    r"\bwhat\b.*\b(whose|that|which)\b",
    r"\bwhere\b.*\b(whose|that|which)\b",
    r"\bwhich\b.*\b(whose|that|which)\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bthen\b",
    r"\bfirst\b",
    r"\band\b",
    r"\bof\b.*\bof\b",
    r"\bfrom\b.*\bto\b",
]

SPLIT_PATTERNS = [
    r",\s*and\s+",
    r"\s+and\s+(?=(who|what|where|when|which|how)\b)",
    r"\sand\s+then\s+",
    r"\sthen\s+",
    r"\safter\s+",
    r"\sbefore\s+",
    r"\swhich\s+",
    r"\sthat\s+",
    r"\swhose\s+",
]

SEMANTIC_TEMPLATE_PATTERNS = {
    "x_y_born_city_country": re.compile(
        r"""(?ix)
        ^\s*(?:what|which)\s+country\s+
        (?:is|was)?\s*
        (?:the\s+)?(?:city|place|location)\s+
        (?:where|in\s+which)\s+
        (?P<x>.+?)\s*'s\s*(?P<y>.+?)\s*
        (?:was\s+)?born
        (?:\s+located\s+in)?\s*\??\s*$
        """
    ),
    "x_y_country": re.compile(
        r"""(?ix)
        ^\s*(?:what|which)\s+country\s+
        (?:is|was)?\s*
        (?P<x>.+?)\s*'s\s*(?P<y>.+?)\s*
        (?:in|located\s+in)?\s*\??\s*$
        """
    ),
}


def _to_full_question(text: str) -> str:
    q = (text or "").strip(" ,.;，。；")
    if not q:
        return q
    return q if q.endswith("?") else f"{q}?"


def _is_multihop_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    lower = q.lower()
    if any(x in lower for x in ["what's your name", "who are you", "tell me about yourself", " your ", " my "]):
        return False
    return any(re.search(cue, q, flags=re.IGNORECASE) for cue in MULTIHOP_CUES)


def _is_low_information_subquestion(text: str) -> bool:
    q = (text or "").strip(" ?？,.;，。；")
    if not q:
        return True
    tokens = [t for t in re.split(r"\s+", q) if t]
    if len(tokens) <= 1:
        return True
    lower = q.lower()
    if lower.startswith("what is "):
        tail = lower[len("what is ") :].strip()
        if len([t for t in tail.split() if t]) <= 1:
            return True
    return False


def _has_dependency_signal(text: str) -> bool:
    q = (text or "").lower()
    if re.search(r"\[[A-Z]\]", text or ""):
        return True
    return any(x in f" {q} " for x in [" it ", " its ", " they ", " them ", " he ", " she "])


def _upgrade_low_information_subquestion(text: str, prev_placeholder: Optional[str] = None) -> str:
    q = (text or "").strip(" ?？,.;，。；")
    if not q:
        return text
    lower = q.lower()
    if prev_placeholder:
        if any(k in lower for k in ["college", "university", "school"]):
            return f"Which college did {prev_placeholder} attend?"
        if any(k in lower for k in ["city", "place", "location", "country"]):
            return f"Where is {prev_placeholder} located?"
        if any(k in lower for k in ["born", "birth"]):
            return f"Where was {prev_placeholder} born?"
        return f"What is the {q} of {prev_placeholder}?"
    return f"Which entity is related to {q}?"


def _ensure_dependency(candidate: str, raw_fragment: str, sub_id: int) -> str:
    if sub_id <= 0:
        return candidate
    if _has_dependency_signal(candidate):
        return candidate
    prev_placeholder = f"[{chr(ord('B') + sub_id - 1)}]"
    return _upgrade_low_information_subquestion(raw_fragment, prev_placeholder=prev_placeholder)


def _to_entity_question(text: str) -> str:
    q = _to_full_question(text)
    if q.lower().startswith(("who ", "what ", "which ", "where ", "when ", "how ")):
        return q
    return f"Who is {q.strip('?')}?"


def _semantic_template_decompose(question: str):
    q = (question or "").strip()
    if not q:
        return None
    m = SEMANTIC_TEMPLATE_PATTERNS["x_y_born_city_country"].match(q)
    if m:
        x = m.group("x").strip(" ,.;")
        y = m.group("y").strip(" ,.;")
        return [
            _to_entity_question(f"{x}'s {y}"),
            "Where was [B] born?",
            "Which country is [C] located in?",
        ]

    m = SEMANTIC_TEMPLATE_PATTERNS["x_y_country"].match(q)
    if m:
        x = m.group("x").strip(" ,.;")
        y = m.group("y").strip(" ,.;")
        return [
            _to_entity_question(f"{x}'s {y}"),
            "Which country is [B] located in?",
        ]

    return None


def _canonicalize_dependency_placeholder(candidate: str, sub_id: int) -> str:
    if sub_id <= 0:
        return candidate
    expected = f"[{chr(ord('B') + sub_id - 1)}]"
    q = str(candidate or "")
    if re.search(r"\[[A-Z]\]", q):
        q = re.sub(r"\[[A-Z]\]", expected, q)
    else:
        coref_patterns = [
            r"\bit\b",
            r"\bits\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\bhe\b",
            r"\bshe\b",
            r"\bthat one\b",
            r"\bthis one\b",
            r"\bthat country\b",
            r"\bthat city\b",
            r"\bthat person\b",
        ]
        replaced = False
        for pat in coref_patterns:
            if re.search(pat, q, flags=re.IGNORECASE):
                q = re.sub(pat, expected, q, flags=re.IGNORECASE)
                replaced = True
                break
        if not replaced and expected not in q:
            if q.endswith("?"):
                q = q[:-1].strip()
            q = f"{q} of {expected}?"
    return q


def split_question(question: str):
    text = (question or "").strip()
    if not text:
        return []

    templated = _semantic_template_decompose(text)
    if templated:
        return templated

    if not _is_multihop_question(text):
        return [text]

    merged = text
    for pattern in SPLIT_PATTERNS:
        merged = re.sub(pattern, " [SPLIT] ", merged, flags=re.IGNORECASE)

    parts = [p.strip(" ,.;，。；") for p in merged.split("[SPLIT]")]
    parts = [p for p in parts if len(p) > 3]
    if not parts:
        parts = [text]

    uniq = []
    seen = set()
    for idx, p in enumerate(parts):
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)

        candidate = _to_full_question(p)
        if _is_low_information_subquestion(candidate):
            prev_placeholder = f"[{chr(ord('B') + idx - 1)}]" if idx > 0 else None
            candidate = _upgrade_low_information_subquestion(p, prev_placeholder=prev_placeholder)

        candidate = _ensure_dependency(candidate, p, idx)
        candidate = _canonicalize_dependency_placeholder(candidate, idx)
        uniq.append(candidate)

    return uniq


def _infer_dep_type(subq: str, sub_id: int) -> str:
    if sub_id == 0:
        return "none"
    q = f" {(subq or '').lower()} "
    if any(x in q for x in ["[b]", "[c]", " it ", " its ", " they ", " them ", " he ", " she ", " that one ", " this one "]):
        return "coref"
    if any(x in q for x in [" which ", " what ", " where ", " when ", " who ", " whose "]):
        return "filter"
    return "bridge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--max_subquestions", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    out_rows = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            item = json.loads(line)
            q = item.get("question", "")
            subquestions = split_question(q)[: args.max_subquestions]

            if len(subquestions) == 1 and subquestions[0].strip("?").lower() == q.strip("?").lower():
                row = dict(item)
                row["parent_id"] = str(row.get("id", total - 1))
                row["sub_id"] = 0
                row["dep_prev_sub_id"] = None
                row["dep_type"] = "none"
                row["needs_prev_answer"] = False
                row["id"] = str(row.get("id", total - 1))
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1
                continue

            parent_id = str(item.get("id", total - 1))
            for i, sq in enumerate(subquestions):
                row = dict(item)
                row["parent_id"] = parent_id
                row["sub_id"] = i
                row["id"] = f"{parent_id}_{i}"
                row["question"] = sq
                row["dep_prev_sub_id"] = i - 1 if i > 0 else None
                row["dep_type"] = _infer_dep_type(sq, i)
                row["needs_prev_answer"] = i > 0
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1

    print(f"Input samples: {total}")
    print(f"Output sub-question samples: {out_rows}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
