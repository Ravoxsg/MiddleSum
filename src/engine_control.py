import torch
import numpy as np
from tqdm import tqdm
from bert_score import score
from rouge_score import rouge_scorer


def filter_n_docs(texts, queries, labels, args):
    idx = [i for i in tqdm(range(len(texts))) if len(texts[i].split("|||||")) == args.min_n_docs]
    print(f"Keeping {len(idx)} MDS data points with enough docs")
    texts = [texts[x] for x in idx]
    queries = [queries[x] for x in idx]
    labels = [labels[x] for x in idx]

    return texts, queries, labels

def prepare_control_texts(texts, queries, labels, tokenizer, args):
    trunc_texts, just_relevant_texts, n_docs = [], [], []
    if args.control == "default":
        trunc_texts, n_docs = prepare_control_default(texts, tokenizer, args)
    elif args.control == "shuffle":
        trunc_texts, n_docs = prepare_control_shuffle(texts, tokenizer, args)
    elif args.control == "position":
        trunc_texts, just_relevant_texts, n_docs = prepare_control_position(texts, queries, labels, tokenizer, args)
    elif args.control == "filling":
        trunc_texts, just_relevant_texts, n_docs = prepare_control_filling(texts, queries, labels, tokenizer, args)
    n_docs = np.mean(np.array(n_docs))
    print(f"Mean # docs / data point: {n_docs:.2f}")

    return trunc_texts, just_relevant_texts

def prepare_control_default(texts, tokenizer, args):
    trunc_texts, n_docs = [], []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        docs = text.split(args.multi_doc_split)
        n_docs.append(len(docs))
        tokens_per_doc = int((args.context_length - 64 - args.max_summary_length - 4 * len(docs)) / len(docs))
        trunc_text = ""
        for doc in docs:
            if not(args.model.startswith("gpt-3.5")):
                doc_ids = tokenizer(doc)["input_ids"]
                doc_ids = doc_ids[:tokens_per_doc]
                decoded_doc = " [DOCUMENT] " + tokenizer.decode(doc_ids, skip_special_tokens=True)
            else:
                doc_ids = tokenizer.encode(doc)
                doc_ids = doc_ids[:tokens_per_doc]
                decoded_doc = " [DOCUMENT] " + tokenizer.decode(doc_ids)
            trunc_text += decoded_doc
        trunc_text = trunc_text[1:]
        trunc_texts.append(trunc_text)

    return trunc_texts, n_docs

def prepare_control_shuffle(texts, tokenizer, args):
    trunc_texts, n_docs = [], []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        docs = text.split(args.multi_doc_split)
        n_docs.append(len(docs))
        if args.dataset == "multixscience":
            temp_docs = docs[1:]
            for _ in range(args.n_shuffle_perm):
                p = np.random.permutation(len(temp_docs))
            docs = [docs[0]] + [temp_docs[x] for x in p]
            # p = np.random.permutation(len(docs))
            # docs = [docs[x] for x in p]
        else:
            for _ in range(args.n_shuffle_perm):
                p = np.random.permutation(len(docs))
            docs = [docs[x] for x in p]
        tokens_per_doc = int((args.context_length - 64 - args.max_summary_length - 4 * len(docs)) / len(docs))
        trunc_text = ""
        for doc in docs:
            if not(args.model.startswith("gpt-3.5")):
                doc_ids = tokenizer(doc)["input_ids"]
                doc_ids = doc_ids[:tokens_per_doc]
                decoded_doc = " [DOCUMENT] " + tokenizer.decode(doc_ids, skip_special_tokens=True)
            else:
                doc_ids = tokenizer.encode(doc)
                doc_ids = doc_ids[:tokens_per_doc]
                decoded_doc = " [DOCUMENT] " + tokenizer.decode(doc_ids)
            trunc_text += decoded_doc
        trunc_text = trunc_text[1:]
        trunc_texts.append(trunc_text)

    return trunc_texts, n_docs

def prepare_control_position(texts, queries, labels, tokenizer, args):
    trunc_texts, just_relevant_texts, n_docs = [], [], []
    flat_docs, flat_labels = [], []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        query = queries[i]
        docs = text.split(args.multi_doc_split)
        n_docs.append(len(docs))
        if args.dataset == "multixscience":
            flat_docs += docs[1:]
            flat_labels += [label] * (len(docs) - 1)
            # flat_docs += docs
            # flat_labels += [label] * len(docs)
        else:
            flat_docs += docs
            if args.control_label == "label":
                flat_labels += [label] * len(docs)
            else:
                flat_labels += [query] * len(docs)
    print(len(flat_docs), len(flat_labels))
    if args.control_metric == "rouge-1":
        print(f"Ranking docs on {args.control_label} R-1...")
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = [100 * scorer.score(flat_docs[i], flat_labels[i])["rouge1"].fmeasure for i in range(len(flat_docs))]
    if args.control_metric == "bertscore":
        print(f"Ranking docs on {args.control_label} BERTScore...")
        p, r, f1 = score(flat_docs, flat_labels, lang='en', verbose=True)
        scores = 100 * f1.detach().cpu().numpy()
    count = 0
    for i in tqdm(range(len(texts))):
        text = texts[i]
        docs = text.split(args.multi_doc_split)
        tokens_per_doc = int((args.context_length - 64 - args.max_summary_length - 4 * len(docs)) / len(docs))
        if args.dataset == "multixscience":
            top_doc = docs[0]
            count += len(docs) - 1
        else:
            doc_scores = scores[count:(count + len(docs))]
            count += len(docs)
            sort_idx = np.argsort(doc_scores)[::-1]
            top_doc = docs[sort_idx[0]]
        if args.swap_docs:
            p_j = np.random.permutation(len(texts) - 1)
            other_texts = texts[:i] + texts[(i + 1):]
            text_j = other_texts[p_j[0]]
            docs_j = text_j.split(args.multi_doc_split)
            p_j = np.random.permutation(len(docs_j))
            docs_j = [docs_j[x] for x in p_j]
            other_docs = docs_j[:(len(docs) - 1)]
            docs = other_docs[:args.control_doc_pos] + [top_doc] + other_docs[args.control_doc_pos:]
        else:  # just use the most relevant doc
            docs = [top_doc]
        trunc_text = ""
        for j in range(len(docs)):
            doc = docs[j]
            if not(args.model.startswith("gpt-3.5")):
                doc_ids = tokenizer(doc)["input_ids"]
                doc_ids = doc_ids[:tokens_per_doc]
                prefix = " [DOCUMENT] "
                if args.swap_docs:
                    if j == args.control_doc_pos:
                        prefix = " [RELEVANT DOCUMENT] "
                    else:
                        prefix = " [IRRELEVANT DOCUMENT] "
                decoded_doc = prefix + tokenizer.decode(doc_ids, skip_special_tokens=True)
            else:
                doc_ids = tokenizer.encode(doc)
                doc_ids = doc_ids[:tokens_per_doc]
                prefix = " [DOCUMENT] "
                if args.swap_docs:
                    if j == args.control_doc_pos:
                        prefix = " [RELEVANT DOCUMENT] "
                    else:
                        prefix = " [IRRELEVANT DOCUMENT] "
                decoded_doc = prefix + tokenizer.decode(doc_ids)
            trunc_text += decoded_doc
            if j == args.control_doc_pos:
                just_relevant_texts.append(doc)
        trunc_text = trunc_text[1:]
        trunc_texts.append(trunc_text)

    return trunc_texts, just_relevant_texts, n_docs

def prepare_control_filling(texts, queries, labels, tokenizer, args):
    trunc_texts, just_relevant_texts, n_docs = [], [], []
    flat_docs, flat_labels = [], []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        query = queries[i]
        docs = text.split(args.multi_doc_split)
        n_docs.append(len(docs))
        if args.dataset == "multixscience":
            flat_docs += docs[1:]
            flat_labels += [label] * (len(docs) - 1)
        else:
            flat_docs += docs
            if args.control_label == "label":
                flat_labels += [label] * len(docs)
            else:
                flat_labels += [query] * len(docs)
    print(len(flat_docs), len(flat_labels))
    for i in tqdm(range(len(texts))):
        text = texts[i]
        docs = text.split(args.multi_doc_split)
        tokens_per_doc = int((args.context_length - 64 - args.max_summary_length - 4 * len(docs)) / len(docs))
        first_doc = docs[0]
        last_doc = docs[-1]
        if args.swap_docs:
            p_j = np.random.permutation(len(texts) - 1)
            other_texts = texts[:i] + texts[(i + 1):]
            text_j = other_texts[p_j[0]]
            docs_j = text_j.split(args.multi_doc_split)
            p_j = np.random.permutation(len(docs_j))
            docs_j = [docs_j[x] for x in p_j]
            other_docs = docs_j[:(len(docs) - 2)]
            docs = [first_doc] + other_docs + [last_doc]
        else:
            docs = [first_doc, last_doc]
        trunc_text = ""
        for j in range(len(docs)):
            doc = docs[j]
            if not(args.model.startswith("gpt-3.5")):
                doc_ids = tokenizer(doc)["input_ids"]
                doc_ids = doc_ids[:tokens_per_doc]
                prefix = " [DOCUMENT] "
                if args.swap_docs:
                    if j in [0, len(docs)-1]:
                        prefix = " [RELEVANT DOCUMENT] "
                    else:
                        prefix = " [IRRELEVANT DOCUMENT] "
                decoded_doc = prefix + tokenizer.decode(doc_ids, skip_special_tokens=True)
            else:
                doc_ids = tokenizer.encode(doc)
                doc_ids = doc_ids[:tokens_per_doc]
                prefix = " [DOCUMENT] "
                if args.swap_docs:
                    if j in [0, len(docs) - 1]:
                        prefix = " [RELEVANT DOCUMENT] "
                    else:
                        prefix = " [IRRELEVANT DOCUMENT] "
                decoded_doc = prefix + tokenizer.decode(doc_ids)
            trunc_text += decoded_doc
        just_relevant_texts.append("[DOCUMENT] " + first_doc + " [DOCUMENT] " + last_doc)
        trunc_text = trunc_text[1:]
        trunc_texts.append(trunc_text)

    return trunc_texts, just_relevant_texts, n_docs
