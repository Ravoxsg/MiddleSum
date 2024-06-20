import json
import os
import pickle
from tqdm import tqdm


val_data = []
with open("summscreen/SummScreen/ForeverDreaming/fd_dev.json", "r") as f:
    for line in f:
        val_data.append(json.loads(line))

test_data = []
with open("summscreen/SummScreen/ForeverDreaming/fd_test.json", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print(len(val_data), len(test_data))

val_ids = [x["filename"] for x in val_data]
test_ids = [x["filename"] for x in test_data]

val_texts, val_labels = [], []
test_texts, test_labels = [], []
folders = ["fd"]
for folder in folders:
    for f in tqdm(os.listdir(f"summscreen/SummScreen_raw/{folder}/")):
        filename = f"summscreen/SummScreen_raw/{folder}/" + f
        data = json.load(open(filename, 'r'))
        text = data["Transcript"]
        text = "\n".join(text)
        label = data["Recap"]
        label = "\n".join(label)
        if f in val_ids:
            val_texts.append(text)
            val_labels.append(label)
        elif f in test_ids:
            test_texts.append(text)
            test_labels.append(label)
print(len(val_texts), len(test_texts))

sets = ["val", "test"]
texts = [val_texts, test_texts]
labels = [val_labels, test_labels]
for i in range(len(sets)):
    subset = sets[i]
    folder =  f"../../raw_summaries/SummScreen/{subset}/"
    os.makedirs(folder, exist_ok=True)
    texts_path = f"{folder}/{subset}_texts_{len(texts[i])}.pkl"
    pickle.dump(texts[i], open(texts_path, "wb"))
    labels_path = f"{folder}/{subset}_labels_{len(labels[i])}.pkl"
    pickle.dump(labels[i], open(labels_path, "wb"))