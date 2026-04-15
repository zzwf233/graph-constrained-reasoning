# For training
DATA_PATH="RoG-webqsp RoG-cwq"
SPLIT=train
N_PROCESS=8
for DATA_PATH in ${DATA_PATH}; do
  python workflow/build_shortest_path_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS}
done

# For evaluation

# DATA_PATH="RoG-webqsp RoG-cwq"
# SPLIT=test
# N_PROCESS=8
# HOP=2 # 3
# for DATA_PATH in ${DATA_PATH}; do
#     python workflow/build_graph_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --K ${HOP}
# done