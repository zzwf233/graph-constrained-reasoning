DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp RoG-cwq"
SPLIT="test"
INDEX_LEN=2
#ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

K="10" # 3 5 10 20
for DATA in ${DATA_LIST}; do
  for k in $K; do
    python workflow/predict_paths_and_answers.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --index_path_length ${INDEX_LEN} --model_name ${MODEL_NAME} --model_path ${MODEL_PATH} --k ${k} --prompt_mode zero-shot --generation_mode group-beam --attn_implementation ${ATTN_IMP}
  done
done
