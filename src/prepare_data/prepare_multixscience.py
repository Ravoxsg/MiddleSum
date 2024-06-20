import pickle
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("multi_x_science_sum")
print(dataset.keys())
split_symbol = " ||||| "

splits = ["train", "validation", "test"]
for split in splits:
    clean_split = split
    if split == "validation":
        clean_split = "val"
    data = dataset[split]
    abstracts = data["abstract"]
    labels = data["related_work"]
    ref_abstracts = data["ref_abstract"]
    texts, counts = [], []
    for i in tqdm(range(len(abstracts))):
        text = abstracts[i] + split_symbol + split_symbol.join(ref_abstracts[i]["abstract"])
        text = text.replace("||||| |||||", "|||||")
        texts.append(text)
        counts.append(text.count("|||||"))
    print(len(texts), len(labels), np.mean(counts))
    folder = "../../raw_summaries/MultiXScience/{clean_split}/"
    os.makedirs(folder, exist_ok=True)
    pickle.dump(texts, open(f"{folder}/{clean_split}_texts_{len(texts)}.pkl", "wb"))
    pickle.dump(labels, open(f"{folder}/{clean_split}_labels_{len(texts)}.pkl", "wb"))
