import os
import random
import argparse
from copy import deepcopy
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

def load_source_dataset(data_path: str, dataset_name: str, split: str):
    input_file = os.path.join(data_path, dataset_name)
    if os.path.isdir(input_file):
        ds = load_from_disk(input_file)
        if hasattr(ds, "keys"):
            return ds[split]
        return ds
    return load_dataset(input_file, split=split)

def collect_global_schema(dataset):
    entities = set()
    relations = set()
    for sample in dataset:
        for h, r, t in sample["graph"]:
            entities.add(h)
            entities.add(t)
            relations.add(r)
    return sorted(entities), sorted(relations)

def rand_poison_graph(sample, global_entities, global_relations, n_insert=20, seed=42):
    rnd = random.Random(seed + hash(str(sample["id"])) % 10_000_000)

    graph = [tuple(x) for x in sample["graph"]]
    existing = set(graph)

    # 题目相关实体优先作为起点，增强“可被检索到”的概率
    q_entities = sample.get("q_entity", [])
    if not q_entities:
        q_entities = []

    inserted = []
    max_trials = n_insert * 50

    for _ in range(max_trials):
        if len(inserted) >= n_insert:
            break

        # 50% 概率从 question entity 出发，50% 概率完全随机
        if q_entities and rnd.random() < 0.5:
            h = rnd.choice(q_entities)
        else:
            h = rnd.choice(global_entities)

        r = rnd.choice(global_relations)
        t = rnd.choice(global_entities)

        triple = (h, r, t)

        # 避免重复，避免自环可以按需关掉
        if triple in existing:
            continue
        if h == t:
            continue

        existing.add(triple)
        inserted.append([h, r, t])

    poisoned = deepcopy(sample)
    poisoned["graph"] = sample["graph"] + inserted
    poisoned["rand_inserted_triples"] = inserted
    return poisoned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="data/poisoned")
    parser.add_argument("--n_insert", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_source_dataset(args.data_path, args.d, args.split)
    global_entities, global_relations = collect_global_schema(ds)

    # clean
    clean_list = []
    # rand
    rand_list = []

    for sample in ds:
        clean_sample = deepcopy(sample)
        clean_sample["rand_inserted_triples"] = []
        clean_list.append(clean_sample)

        rand_sample = rand_poison_graph(
            sample,
            global_entities=global_entities,
            global_relations=global_relations,
            n_insert=args.n_insert,
            seed=args.seed,
        )
        rand_list.append(rand_sample)

    clean_ds = Dataset.from_list(clean_list)
    rand_ds = Dataset.from_list(rand_list)

    clean_path = os.path.join(args.output_dir, f"{args.d}_clean_{args.split}")
    rand_path = os.path.join(args.output_dir, f"{args.d}_rand_{args.split}")

    os.makedirs(args.output_dir, exist_ok=True)
    clean_ds.save_to_disk(clean_path)
    rand_ds.save_to_disk(rand_path)

    print(f"Saved clean dataset to: {clean_path}")
    print(f"Saved rand dataset to: {rand_path}")

if __name__ == "__main__":
    main()s
