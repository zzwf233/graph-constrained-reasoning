import os
from datasets import load_dataset, load_from_disk

def load_qa_dataset(data_path: str, dataset_name: str, split: str):
    """
    统一支持两种输入：
    1. Hugging Face 数据集名路径：如 rmanluo/RoG-webqsp
    2. 本地保存的数据集目录：如 data/poisoned/RoG-webqsp_clean
    """
    input_file = os.path.join(data_path, dataset_name)

    if os.path.isfile(input_file):
        ext = os.path.splitext(input_file)[1].lower()
        split_name = str(split or "train").split("[", 1)[0] or "train"
        if ext in [".json", ".jsonl"]:
            return load_dataset("json", data_files={split_name: input_file}, split=split)
        if ext == ".parquet":
            return load_dataset("parquet", data_files={split_name: input_file}, split=split)

    # 优先判断是否为本地 save_to_disk 的目录
    if os.path.isdir(input_file):
        try:
            ds = load_from_disk(input_file)
            # DatasetDict
            if hasattr(ds, "keys"):
                return ds[split]
            return ds
        except Exception:
            pass

    # 否则按原来的 HF 方式加载
    return load_dataset(input_file, split=split)
