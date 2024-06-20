# RQ2: relative alignment between visible source sentences and generated summary sentences

import numpy as np
import argparse
import pickle
import os
import tiktoken
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from keys import root, hf_token
from utils import boolean_string, seed_everything, settle_args, load_data, load_pred, get_clean_model_name
from engine import prepare_texts, prepare_texts_gptlikert


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = root)
parser.add_argument('--dataset', type=str, default = "xsum",
                    choices=["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience"])
parser.add_argument('--subset', type=str, default = "test")
parser.add_argument('--instruction_position', type=str, default = "post",
                    choices=["pre", "post"])
parser.add_argument('--focus_prompt', type=boolean_string, default = False)
parser.add_argument('--max_size', type=int, default = 1000)
parser.add_argument('--multi_doc_split', type=str, default = "|||||")
parser.add_argument('--use_control', type=boolean_string, default = False)
parser.add_argument('--swap_docs', type=boolean_string, default = False)
parser.add_argument('--oracle_n_sents', type=boolean_string, default = False)
parser.add_argument('--oracle_n_words', type=boolean_string, default = False)
parser.add_argument('--check_stats', type=boolean_string, default = False)
parser.add_argument('--analysis_size', type=int, default = 1000)
parser.add_argument('--clean_model_name', type=str, default = "llama_2_7b",
                    choices=["llama_2_7b_base", "llama_2_13b_base",
                             "flan_ul2", "llama_2_7b", "llama_2_13b", "xgen_7b", "mistral_7b",
                             "vicuna_7b_16k", "llama_2_7b_32k",
                             "gpt-3.5-turbo-0125"])
parser.add_argument('--inference_method', type=str, default = "normal",
                    choices = ["normal", "pyramidal", "incremental"])
parser.add_argument('--decoding_method', type=str, default = "top_k",
                    choices = ["greedy", "beam_search", "top_k", "top_p", "temperature"])
parser.add_argument('--enforced_max_length', type=int, default=-1)  # [-1, 512, 1024, 2048, 4096, 6144, 8192, 10240]
parser.add_argument('--alignment_metric', type=str, default = "rouge-1",
                    choices=["rouge-1", "rouge-2", "rouge-l"])
parser.add_argument('--compute_alignment', type=boolean_string, default = True)
parser.add_argument('--summaries_alignment', type=boolean_string, default = True)
parser.add_argument('--labels_alignment', type=boolean_string, default = False)
parser.add_argument('--n_bins', type=int, default = 10)

args = parser.parse_args()

settle_args(args)


def main(args):
    print(args)

    seed_everything(args)

    # load the data
    texts, labels = load_data(args)
    queries = [""] * len(texts)
    if args.dataset == "middlesum":
        (texts, queries) = texts
    texts = texts[:args.analysis_size]
    queries = queries[:args.analysis_size]
    labels = labels[:args.analysis_size]
    print(len(texts), len(queries), len(labels))
    print("*" * 50 + " First source:")
    print(texts[0][:500])
    print("*" * 50 + " First label:")
    print(" ".join(labels[0].split()))

    # tokenizer
    tokenizer = None
    if not(args.model.startswith("gpt-3.5")):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=hf_token,
            cache_dir=f"{args.root}/hf_models/{args.model}",
            trust_remote_code=True
        )

    # truncate texts
    if args.model.startswith("gpt-3.5"):
        encoding = tiktoken.encoding_for_model(args.model)
        args.gpt_model_max_length = args.context_length
        trunc_texts = prepare_texts_gptlikert(encoding, texts, queries, args)
    else:
        trunc_texts = prepare_texts(texts, queries, tokenizer, args)

    if args.summaries_alignment:
        # load summaries
        summaries = load_pred(args)
        summaries = summaries[:args.analysis_size]
        print(len(summaries))
        print("*" * 50 + " First prediction:")
        print(" ".join(summaries[0].split()))

        # compute alignments
        model_name = get_clean_model_name(args)
        folder = f"alignments/{args.dataset_name}/{args.subset}"
        os.makedirs(folder, exist_ok=True)
        alignment_path = f"{folder}/{args.subset}_summaries_relative_alignment_{model_name}_{args.decoding_method}_{len(trunc_texts)}.pkl"
        if args.alignment_metric == "rouge-2":
            alignment_path = f"{folder}/{args.subset}_summaries_relative_alignment_r2_{model_name}_{args.decoding_method}_{len(trunc_texts)}.pkl"
        elif args.alignment_metric == "rouge-l":
            alignment_path = f"{folder}/{args.subset}_summaries_relative_alignment_rl_{model_name}_{args.decoding_method}_{len(trunc_texts)}.pkl"

        if args.compute_alignment:
            aligned_summaries = align_source_preds(trunc_texts, summaries, args)
            pickle.dump(aligned_summaries, open(alignment_path, "wb"))
            print(f"\nWrote alignments to: {alignment_path}")
        else:
            aligned_summaries = pickle.load(open(alignment_path, "rb"))
            print(f"\nLoaded alignments from: {alignment_path}")

        # stats on alignment
        compute_stats(trunc_texts, summaries, aligned_summaries)
        # histogram
        compute_histogram(aligned_summaries, args)

    if args.labels_alignment:
        print("\nAlignment for labels...")

        # compute alignments
        folder = f"alignments/{args.dataset_name}/{args.subset}"
        os.makedirs(folder, exist_ok=True)
        alignment_path = f"{folder}/{args.subset}_labels_alignment_{len(trunc_texts)}.pkl"
        if args.alignment_metric == "rouge-2":
            alignment_path = f"{folder}/{args.subset}_labels_alignment_r2_{len(trunc_texts)}.pkl"
        elif args.alignment_metric == "rouge-l":
            alignment_path = f"{folder}/{args.subset}_labels_alignment_rl_{len(trunc_texts)}.pkl"

        if args.compute_alignment:
            print("\nComputing labels alignment...")
            aligned_labels = align_source_preds(texts, labels, args)
            pickle.dump(aligned_labels, open(alignment_path, "wb"))
            print(f"\nWrote labels alignments to: {alignment_path}")
        else:
            aligned_labels = pickle.load(open(alignment_path, "rb"))
            print(f"\nLoaded labels alignments from: {alignment_path}")

        # stats on alignment
        compute_stats(texts, labels, aligned_labels)
        # histogram
        compute_histogram(aligned_labels, args)

def align_source_preds(texts, summaries, args):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    if args.alignment_metric == "rouge-2":
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    elif args.alignment_metric == "rouge-l":
        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    aligned_summaries = []
    for i in tqdm(range(len(texts))):
        text_sents = sent_tokenize(texts[i])
        if len(text_sents) == 0:
            aligned_summaries.append((0, texts[i]))
        else:
            summary = summaries[i]
            summary = "\n".join(sent_tokenize(summary))
            sents_scores = np.array([scorer.score(summary, x)["rouge1"].fmeasure for x in text_sents])
            idx = np.argmax(sents_scores)
            score = sents_scores[idx]
            used_idx = [idx]
            align_sents = [text_sents[idx]]
            improve = True
            while improve and len(used_idx) < len(text_sents):
                sents = [text_sents[j] for j in range(len(text_sents)) if not(j in used_idx)]
                sents_scores = np.array([scorer.score(summary, "\n".join(align_sents + [x]))["rouge1"].fmeasure for x in sents])
                idx = np.argmax(sents_scores)
                max_score = sents_scores[idx]
                if max_score > score:
                    score = max_score
                    real_idx = 0
                    while (text_sents[real_idx] != sents[idx]) or (real_idx in used_idx):
                        real_idx += 1
                    used_idx.append(real_idx)
                    align_sents.append(sents[idx])
                    sort_idx = np.argsort(np.array(used_idx))
                    used_idx = [used_idx[x] for x in sort_idx]
                    align_sents = [align_sents[x] for x in sort_idx]
                else:
                    improve = False
            used_idx = np.array(used_idx)
            assert len(used_idx) == len(np.unique(used_idx))
            used_idx = [x for x in used_idx]
            aligned_summaries.append((used_idx, align_sents))

    return aligned_summaries

def compute_stats(texts, summaries, aligned_summaries):
    n_sents = np.mean([len(x[0]) for x in aligned_summaries])
    print(f"\nThere are {n_sents:.4f} sentences found after alignment on average")
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    r1_scores = [100 * scorer.score("\n".join(sent_tokenize(summaries[i])), "\n".join(aligned_summaries[i][1]))["rouge1"].fmeasure for i in range(len(summaries))]
    print(f"Avg R-1 scores is: {np.mean(r1_scores):.4f}")
    for i in tqdm(range(len(texts))):
        text = texts[i]
        n_sents = len(sent_tokenize(text))
        for j in range(len(aligned_summaries[i][0])):
            aligned_summaries[i][0][j] = 100 * (aligned_summaries[i][0][j] + 1) / n_sents
    mean_ranks = [np.mean(np.array(x[0])) for x in aligned_summaries]
    low = np.percentile(mean_ranks, 5)
    q1 = np.percentile(mean_ranks, 25)
    median = np.percentile(mean_ranks, 50)
    mean = np.mean(mean_ranks)
    q3 = np.percentile(mean_ranks, 75)
    high = np.percentile(mean_ranks, 95)
    print(f"Sentence ranks: Low: {low:.4f}, Q1: {q1:.4f}, median: {median:.4f}, mean: {mean:.4f}, Q3: {q3:.4f}, high: {high:.4f}")

def compute_histogram(aligned_summaries, args):
    all_ranks = []
    for x in aligned_summaries:
        all_ranks += x[0]
    all_ranks = np.array(all_ranks)
    print(all_ranks.shape)
    bins = 100 * np.arange(args.n_bins+1) / args.n_bins
    hist = np.histogram(all_ranks, bins=bins)
    (values, bins) = hist
    values = 100 * values / all_ranks.shape[0]
    values = [np.round(x, 2) for x in values]
    print(f"Histogram with {args.n_bins}:")
    print(values)


if __name__ == '__main__':
    main(args)

