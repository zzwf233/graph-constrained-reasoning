import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llms import get_registed_model
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder
from workflow.eval_poison_metrics import extract_answer_text


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replace_dependency_placeholder(question: str, prev_answer: str) -> str:
    if not prev_answer:
        return question
    q = str(question or "")
    if any(token in q for token in ["[B]", "[C]", "[D]", "[E]"]):
        return q.replace("[B]", prev_answer).replace("[C]", prev_answer).replace("[D]", prev_answer).replace("[E]", prev_answer)
    return q


def top_answer(prediction) -> str:
    if isinstance(prediction, list):
        for item in prediction:
            ans = extract_answer_text(item)
            if ans:
                return ans
        return ""
    return extract_answer_text(prediction)


def group_by_parent(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for row in rows:
        pid = str(row.get("parent_id", row.get("id", "")))
        grouped[pid].append(row)
    for pid in grouped:
        grouped[pid].sort(key=lambda x: int(x.get("sub_id", 0) or 0))
    return grouped


def predict_one(data: Dict[str, Any], input_builder, model) -> Optional[Dict[str, Any]]:
    input_query, ground_paths, trie = input_builder.process_input(data)
    if trie is None:
        return None
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    model_input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        model_input,
        trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    if prediction is None:
        return None
    return {
        "id": data.get("id"),
        "parent_id": data.get("parent_id", data.get("id")),
        "sub_id": data.get("sub_id", 0),
        "question": data.get("question", ""),
        "prediction": prediction,
        "ground_truth": data.get("answer", []),
        "ground_truth_paths": ground_paths,
        "input": model_input,
        "prev_answer_used": data.get("_prev_answer_used", ""),
    }


def main(args, LLM):
    rows = load_jsonl(args.data_file)
    grouped = group_by_parent(rows)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(args.output_file) and not args.force:
        raise FileExistsError(f"Output exists, pass --force to overwrite: {args.output_file}")

    model = LLM(args)
    model.prepare_for_inference()
    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=False,
    )

    total_rows = sum(len(parent_rows) for parent_rows in grouped.values())
    with open(args.output_file, "w", encoding="utf-8") as fout:
        progress = tqdm(total=total_rows, desc="Chain prediction")
        for _, parent_rows in grouped.items():
            prev_answer = ""
            prev_by_sub_id = {}
            for row in parent_rows:
                data = dict(row)
                if data.get("needs_prev_answer"):
                    dep_prev = data.get("dep_prev_sub_id")
                    if dep_prev is None:
                        dep_prev = int(data.get("sub_id", 0) or 0) - 1
                    prev_answer = prev_by_sub_id.get(int(dep_prev), prev_answer)
                    if prev_answer:
                        data["question"] = replace_dependency_placeholder(data.get("question", ""), prev_answer)
                        data["q_entity"] = [prev_answer]
                        data["_prev_answer_used"] = prev_answer

                res = predict_one(data, input_builder, model)
                if res is None:
                    res = {
                        "id": data.get("id"),
                        "parent_id": data.get("parent_id", data.get("id")),
                        "sub_id": data.get("sub_id", 0),
                        "question": data.get("question", ""),
                        "prediction": [],
                        "ground_truth": data.get("answer", []),
                        "ground_truth_paths": [],
                        "input": "",
                        "prev_answer_used": data.get("_prev_answer_used", ""),
                    }
                answer = top_answer(res["prediction"])
                try:
                    sub_id = int(data.get("sub_id", 0) or 0)
                except (TypeError, ValueError):
                    sub_id = 0
                if answer:
                    prev_by_sub_id[sub_id] = answer
                    prev_answer = answer
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                progress.update(1)
        progress.close()

    print(f"Saved chain predictions to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="Poisoned decomposed sub-question JSONL")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", default="GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--index_path_length", type=int, default=2)
    parser.add_argument("--prompt_mode", default="zero-shot", choices=["zero-shot", "mcq-zero-shot", "few-shot"])
    parser.add_argument("--undirected", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--force", action="store_true")

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()
    main(args, LLM)
