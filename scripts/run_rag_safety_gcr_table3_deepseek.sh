set -euo pipefail

DATA_PATH=data/poisoned
DATASETS="RoG-webqsp RoG-cwq"
ATTACKERS="clean rand"
SPLIT=test
INDEX_LEN=2
ATTN_IMP=sdpa
GCR_MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
GCR_MODEL_NAME=$(basename "$GCR_MODEL_PATH")
ANSWER_MODEL_NAME=deepseek-ai/DeepSeek-V3.2
N_THREAD=1
K=10

for DATA in ${DATASETS}; do
  for ATTACKER in ${ATTACKERS}; do
    POISONED_DATA="${DATA}_${ATTACKER}_test"

    python workflow/predict_paths_and_answers.py \
      --data_path ${DATA_PATH} \
      --d ${POISONED_DATA} \
      --split ${SPLIT} \
      --index_path_length ${INDEX_LEN} \
      --model_name ${GCR_MODEL_NAME} \
      --model_path ${GCR_MODEL_PATH} \
      --k ${K} \
      --prompt_mode zero-shot \
      --generation_mode group-beam \
      --attn_implementation ${ATTN_IMP}

    REASONING_PATH="results/GenPaths/${POISONED_DATA}/${GCR_MODEL_NAME}/${SPLIT}/zero-shot-group-beam-k${K}-index_len${INDEX_LEN}/predictions.jsonl"

    python workflow/predict_final_answer.py \
      --data_path ${DATA_PATH} \
      --d ${POISONED_DATA} \
      --split ${SPLIT} \
      --model_name ${ANSWER_MODEL_NAME} \
      --reasoning_path ${REASONING_PATH} \
      --add_path True \
      -n ${N_THREAD}
  done
done
