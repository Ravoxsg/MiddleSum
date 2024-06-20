import pickle
import json

texts = pickle.load(open("summaries/MiddleSum/test/test_texts_225.pkl", "rb"))
labels = pickle.load(open("summaries/MiddleSum/test/test_labels_225.pkl", "rb"))
queries = pickle.load(open("summaries/MiddleSum/test/test_queries_225.pkl", "rb"))
print(len(texts), len(labels), len(queries))

export_path = "MiddleSum/middlesum.jsonl"
data_points = []
for i in range(len(texts)):
    d = {}
    d["source"] = texts[i]
    d["label"] = labels[i]
    d["dataset"] = queries[i]
    data_points.append(d)
with open(export_path, 'w') as out:
    for ddict in data_points:
        jout = json.dumps(ddict) + '\n'
        out.write(jout)