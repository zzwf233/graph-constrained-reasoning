DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp RoG-cwq"
SPLIT="test"

MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
# Use 1 thread by default to avoid provider TPM rate limit (HTTP 429).
N_THREAD=1

# MODEL_NAME=gpt-4o-mini
# N_THREAD=10

for DATA in ${DATA_LIST}; do
  REASONING_PATH="results/GenPaths/${DATA}/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl"

  python workflow/predict_final_answer.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --model_name ${MODEL_NAME} --reasoning_path ${REASONING_PATH} --add_path True -n ${N_THREAD}
done
