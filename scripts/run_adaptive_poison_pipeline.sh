#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_adaptive_poison_pipeline.sh <MODEL_NAME> [DATASETS]
#
# Example:
#   export OPENAI_API_KEY=...
#   bash scripts/run_adaptive_poison_pipeline.sh GCR-Meta-Llama-3.1-8B-Instruct "RoG-webqsp RoG-cwq"
#
# This script runs Ours only. Clean and Rand should be run by separate scripts.

MODEL_NAME=${1:-""}
DATASETS=${2:-"RoG-webqsp RoG-cwq"}

if [[ -z "${MODEL_NAME}" || "${MODEL_NAME}" == "<model_name>" || "${MODEL_NAME}" == "<MODEL_NAME>" ]]; then
  echo "Usage: bash scripts/run_adaptive_poison_pipeline.sh MODEL_NAME [DATASETS]"
  echo "Example: bash scripts/run_adaptive_poison_pipeline.sh GCR-Meta-Llama-3.1-8B-Instruct \"RoG-webqsp RoG-cwq\""
  exit 1
fi

REUSE_ATTACK_FILE=${REUSE_ATTACK_FILE:-}
REQUIRE_REUSE_ATTACK=${REQUIRE_REUSE_ATTACK:-false}

if [[ -z "${OPENAI_API_KEY:-}" && ! ( -n "${REUSE_ATTACK_FILE}" && "${REQUIRE_REUSE_ATTACK}" == "true" ) ]]; then
  echo "Error: Ours requires OPENAI_API_KEY."
  echo "Please export OPENAI_API_KEY before running this script."
  exit 1
fi

TOPK=${TOPK:-5}
INDEX_LEN=${INDEX_LEN:-2}
PROMPT_MODE=${PROMPT_MODE:-zero-shot}
GENERATION_MODE=${GENERATION_MODE:-group-beam}
PREDICT_SPLIT=${PREDICT_SPLIT:-test}
PREDICT_PATH=${PREDICT_PATH:-results/GenPaths}
RUN_INFERENCE=${RUN_INFERENCE:-1}
RUN_CHAIN_INFERENCE=${RUN_CHAIN_INFERENCE:-0}
FORCE_INFERENCE=${FORCE_INFERENCE:-1}
FORCE_RULES=${FORCE_RULES:-1}
MODEL_PATH=${MODEL_PATH:-"rmanluo/${MODEL_NAME}"}
DTYPE=${DTYPE:-bf16}
QUANT=${QUANT:-none}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
CHAT_MODEL=${CHAT_MODEL:-true}

MAX_SUBQUESTIONS=${MAX_SUBQUESTIONS:-3}
INJECT_TOP_K=${INJECT_TOP_K:-3}
NUM_CANDIDATES=${NUM_CANDIDATES:-12}
SEED=${SEED:-42}
HOP_REPEAT=${HOP_REPEAT:-100}
SINGLE_HOP_REPEAT=${SINGLE_HOP_REPEAT:-200}
FRONT_BOOST=${FRONT_BOOST:-1.4}
HOP_BOOST_IF_TYPE_MATCH=${HOP_BOOST_IF_TYPE_MATCH:-1.5}
HOP_BOOST_IF_TYPE_MISMATCH=${HOP_BOOST_IF_TYPE_MISMATCH:-0.7}
EXPAND_SINGLE_HOP=${EXPAND_SINGLE_HOP:-true}
INCLUDE_DIRECT_SINGLE_HOP=${INCLUDE_DIRECT_SINGLE_HOP:-true}
SINGLE_HOP_BRIDGE_RELATION=${SINGLE_HOP_BRIDGE_RELATION:-poison.answer}
POISON_POSITION=${POISON_POSITION:-front}
# GCR decodes paths through a trie. `attack_front` can be useful for a strong
# attack run, but when parent rows have no precomputed clean `paths` it becomes
# effectively attack-only and can saturate A-* metrics. Keep the default graph
# poisoning only; opt in to path poisoning explicitly for ablations.
POISON_PATH_MODE=${POISON_PATH_MODE:-off}
MAX_ATTACK_PATHS=${MAX_ATTACK_PATHS:-256}
MAX_CLEAN_PATHS=${MAX_CLEAN_PATHS:-256}
CLEAN_INDEX_LEN=${CLEAN_INDEX_LEN:-${INDEX_LEN}}
# RoG-main-poisoning's cascade evaluator treats adversarial-answer matches as
# substring containment after normalization. Use the same default for A-* unless
# an exact-match ablation is explicitly requested.
ATTACK_MATCH=${ATTACK_MATCH:-substring}
BUDGET_K=${BUDGET_K:-0}
PER_ANSWER_BUDGET_K=${PER_ANSWER_BUDGET_K:-0}
STRICT_TARGET_FILTER=${STRICT_TARGET_FILTER:-true}
GOLD_OVERLAP_THRESHOLD=${GOLD_OVERLAP_THRESHOLD:-0.5}
SUMMARY_FILE=${SUMMARY_FILE:-results/adaptive_poison_ours_summary.tsv}

mkdir -p "$(dirname "${SUMMARY_FILE}")"
printf "dataset\tHit\tF1\tPrecision\tRecall\tHits@1\tEM\tEM@1\tA-Precision\tA-H@1\tA-MRR\tshared_parent_spread_rate\toverall_parent_spread_rate\tchain_success@k\tdependency_ASR\n" > "${SUMMARY_FILE}"

append_ours_summary() {
  local dataset="$1"
  local metrics_file="$2"
  python - "${SUMMARY_FILE}" "${dataset}" "${metrics_file}" <<'PY'
import json
import sys

summary_file, dataset, metrics_file = sys.argv[1:4]
with open(metrics_file, "r", encoding="utf-8") as f:
    metrics = json.load(f)

std = metrics.get("standard", {})
atk = metrics.get("attack", {})
cascade = metrics.get("cascade", {})
cols = [
    dataset,
    std.get("Hit", ""),
    std.get("F1", ""),
    std.get("Precision", ""),
    std.get("Recall", ""),
    std.get("Hits@1", ""),
    std.get("EM", ""),
    std.get("EM@1", ""),
    atk.get("A-Precision", ""),
    atk.get("A-H@1", ""),
    atk.get("A-MRR", ""),
    cascade.get("shared_parent_spread_rate", ""),
    cascade.get("overall_parent_spread_rate", ""),
    cascade.get("chain_success@k", ""),
    cascade.get("dependency_ASR", ""),
]

def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

with open(summary_file, "a", encoding="utf-8") as f:
    f.write("\t".join(fmt(x) for x in cols) + "\n")
PY
}

run_ours_prediction() {
  local data_path="$1"
  local data_file="$2"
  local output_d="$3"
  local predict_file="$4"

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    FORCE_ARGS=()
    if [[ "${FORCE_INFERENCE}" == "1" ]]; then
      FORCE_ARGS+=(--force)
    fi

    python workflow/predict_paths_and_answers.py \
      --data_path "${data_path}" \
      --d "${data_file}" \
      --output_d "${output_d}" \
      --split "${PREDICT_SPLIT}" \
      --index_path_length "${INDEX_LEN}" \
      --predict_path "${PREDICT_PATH}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --generation_mode "${GENERATION_MODE}" \
      --prompt_mode "${PROMPT_MODE}" \
      --k "${TOPK}" \
      --dtype "${DTYPE}" \
      --quant "${QUANT}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --chat_model "${CHAT_MODEL}" \
      "${FORCE_ARGS[@]}"
  elif [[ ! -f "${predict_file}" ]]; then
    echo "[Warning] RUN_INFERENCE=0 and prediction file not found: ${predict_file}"
  fi
}

for D in ${DATASETS}; do
  echo "============================================================"
  echo "[${D}] Ours adaptive poison pipeline"

  INPUT_JSONL="datasets/${D}_test.jsonl"
  if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "[${D}] Input file ${INPUT_JSONL} not found. Trying local parquet first, then HuggingFace fallback..."
    python - "${D}" "${INPUT_JSONL}" <<'PY'
import glob
import json
import os
import sys
from datasets import load_dataset

dataset_name = sys.argv[1]
output_file = sys.argv[2]
os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

local_dir_candidates = {
    "RoG-webqsp": ["datasets/webqsp", "datasets/RoG-webqsp"],
    "RoG-cwq": ["datasets/cwq", "datasets/RoG-cwq"],
}

ds = None
for local_dir in local_dir_candidates.get(dataset_name, []):
    files = sorted(glob.glob(os.path.join(local_dir, "test-*.parquet")))
    if files:
        print(f"[{dataset_name}] Found local parquet files: {files}")
        ds = load_dataset("parquet", data_files=files, split="train")
        break

if ds is None:
    print(f"[{dataset_name}] No local parquet found. Downloading from HuggingFace rmanluo/{dataset_name} ...")
    ds = load_dataset(f"rmanluo/{dataset_name}", split="test")

with open(output_file, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[{dataset_name}] Exported {len(ds)} rows to: {output_file}")
PY
  fi

  DECOMPOSED_FILE="datasets/${D}_test_decomposed.jsonl"
  POISON_SUBQ_FILE="datasets/${D}_test_adaptive_poisoned_ours.jsonl"
  PARENT_POISON_FILE="datasets/${D}_test_adaptive_poisoned_parent_ours.jsonl"
  RULE_FILE="results/GenRule/${D}/test/rules.jsonl"

  echo "[${D}] Step 1/5: Decompose subquestions"
  python src/attack_scripts_adaptive/decompose_subquestions.py \
    --input_file "${INPUT_JSONL}" \
    --output_file "${DECOMPOSED_FILE}" \
    --max_subquestions "${MAX_SUBQUESTIONS}"

  echo "[${D}] Step 2/5: Generate poison rules"
  if [[ "${FORCE_RULES}" == "1" || ! -f "${RULE_FILE}" ]]; then
    python workflow/generate_poison_rules.py \
      --input_file "${DECOMPOSED_FILE}" \
      --output_file "${RULE_FILE}"
  else
    echo "[${D}] Reusing rule file: ${RULE_FILE}"
  fi

  echo "[${D}] Step 3/5: Ours adaptive poisoning"
  POISON_ARGS=(
    --input_file "${DECOMPOSED_FILE}"
    --output_file "${POISON_SUBQ_FILE}"
    --mode "ours"
    --inject_top_k "${INJECT_TOP_K}"
    --num_candidates "${NUM_CANDIDATES}"
    --seed "${SEED}"
    --hop_repeat "${HOP_REPEAT}"
    --single_hop_repeat "${SINGLE_HOP_REPEAT}"
    --front_boost "${FRONT_BOOST}"
    --hop_boost_if_type_match "${HOP_BOOST_IF_TYPE_MATCH}"
    --hop_boost_if_type_mismatch "${HOP_BOOST_IF_TYPE_MISMATCH}"
    --expand_single_hop "${EXPAND_SINGLE_HOP}"
    --include_direct_single_hop "${INCLUDE_DIRECT_SINGLE_HOP}"
    --single_hop_bridge_relation "${SINGLE_HOP_BRIDGE_RELATION}"
    --budget_k "${BUDGET_K}"
    --per_answer_budget_k "${PER_ANSWER_BUDGET_K}"
    --require_reuse_attack "${REQUIRE_REUSE_ATTACK}"
    --strict_target_filter "${STRICT_TARGET_FILTER}"
    --gold_overlap_threshold "${GOLD_OVERLAP_THRESHOLD}"
  )
  if [[ -n "${REUSE_ATTACK_FILE}" ]]; then
    POISON_ARGS+=(--reuse_attack_file "${REUSE_ATTACK_FILE}")
  fi
  if [[ -f "${RULE_FILE}" ]]; then
    POISON_ARGS+=(--rule_file "${RULE_FILE}")
  fi
  python src/attack_scripts_adaptive/poison_data_adaptive.py "${POISON_ARGS[@]}"

  echo "[${D}] Step 4/5: Merge subquestion poison back to parent questions"
  python workflow/merge_subquestion_poison_to_parent.py \
    --original_file "${INPUT_JSONL}" \
    --poisoned_subq_file "${POISON_SUBQ_FILE}" \
    --output_file "${PARENT_POISON_FILE}" \
    --poison_position "${POISON_POSITION}" \
    --path_mode "${POISON_PATH_MODE}" \
    --max_attack_paths "${MAX_ATTACK_PATHS}" \
    --clean_index_path_length "${CLEAN_INDEX_LEN}" \
    --max_clean_paths "${MAX_CLEAN_PATHS}"

  echo "[${D}] Step 5/5: Run poisoned inference and evaluate ours metrics"
  POSTFIX="${PROMPT_MODE}-${GENERATION_MODE}-k${TOPK}-index_len${INDEX_LEN}"
  POISON_OUTPUT_D="${D}_adaptive_ours"
  PREDICT_FILE="${PREDICT_PATH}/${POISON_OUTPUT_D}/${MODEL_NAME}/${PREDICT_SPLIT}/${POSTFIX}/predictions.jsonl"
  run_ours_prediction "$(dirname "${PARENT_POISON_FILE}")" "$(basename "${PARENT_POISON_FILE}")" "${POISON_OUTPUT_D}" "${PREDICT_FILE}"

  if [[ ! -f "${PREDICT_FILE}" ]]; then
    echo "[${D}] Error: poisoned predictions.jsonl not found: ${PREDICT_FILE}"
    echo "Set RUN_INFERENCE=1 to generate it, or place predictions at the path above."
    exit 1
  fi

  OUTPUT_PREFIX="$(dirname "${PREDICT_FILE}")/${D}_ours_poison_eval"
  python workflow/eval_poison_metrics.py \
    --predict_file "${PREDICT_FILE}" \
    --poison_file "${POISON_SUBQ_FILE}" \
    --topk "${TOPK}" \
    --output_prefix "${OUTPUT_PREFIX}" \
    --attack_match "${ATTACK_MATCH}"

  python workflow/diagnose_poison_failures.py \
    --predict_file "${PREDICT_FILE}" \
    --poison_file "${POISON_SUBQ_FILE}" \
    --parent_file "${PARENT_POISON_FILE}" \
    --topk "${TOPK}" \
    --output_file "${OUTPUT_PREFIX}_failure_diagnosis.json"

  append_ours_summary "${D}" "${OUTPUT_PREFIX}_all_metrics.json"

  if [[ "${RUN_CHAIN_INFERENCE}" == "1" ]]; then
    CHAIN_PREDICT_FILE="$(dirname "${PREDICT_FILE}")/subquestion_chain_predictions.jsonl"
    python workflow/predict_subquestion_chain.py \
      --data_file "${POISON_SUBQ_FILE}" \
      --output_file "${CHAIN_PREDICT_FILE}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --index_path_length "${INDEX_LEN}" \
      --prompt_mode "${PROMPT_MODE}" \
      --dtype "${DTYPE}" \
      --quant "${QUANT}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}" \
      --generation_mode "${GENERATION_MODE}" \
      --k "${TOPK}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --chat_model "${CHAT_MODEL}" \
      --force

    python workflow/eval_poison_metrics.py \
      --predict_file "${CHAIN_PREDICT_FILE}" \
      --poison_file "${POISON_SUBQ_FILE}" \
      --topk "${TOPK}" \
      --output_prefix "$(dirname "${PREDICT_FILE}")/${D}_ours_chain_eval" \
      --attack_match "${ATTACK_MATCH}" \
      --target_id_field id
  fi

  echo "[${D}] Done. Ours outputs:"
  echo "  - ${POISON_SUBQ_FILE}"
  echo "  - ${PARENT_POISON_FILE}"
  echo "  - ${PREDICT_FILE}"
  echo "  - ${OUTPUT_PREFIX}_std_metrics.txt"
  echo "  - ${OUTPUT_PREFIX}_attack_metrics.txt"
  echo "  - ${OUTPUT_PREFIX}_cascade_metrics.txt"
  echo "  - ${OUTPUT_PREFIX}_failure_diagnosis.json"
  echo "  - ${OUTPUT_PREFIX}_all_metrics.json"
  if [[ "${RUN_CHAIN_INFERENCE}" == "1" ]]; then
    echo "  - $(dirname "${PREDICT_FILE}")/subquestion_chain_predictions.jsonl"
    echo "  - $(dirname "${PREDICT_FILE}")/${D}_ours_chain_eval_all_metrics.json"
  fi
  echo
done

echo "============================================================"
echo "Ours summary:"
cat "${SUMMARY_FILE}"
