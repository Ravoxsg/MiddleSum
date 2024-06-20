# RQ3: dependency between performance and position of salient info

import numpy as np
import argparse
import pickle
import os
import tiktoken
from tqdm import tqdm
from scipy.stats import spearmanr
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer

from keys import root, hf_token
from utils import boolean_string, seed_everything, settle_args, load_data, load_pred
from rq2_alignment_sentences import align_source_preds
from evaluation import compute_scores


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = root)
parser.add_argument('--dataset', type=str, default = "samsum",
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
parser.add_argument('--enforced_max_length', type=float, default = -1)  # [-1, 512, 1024, 2048, 4096, 6144, 8192, 10240]
parser.add_argument('--compute_alignment', type=boolean_string, default = False)
parser.add_argument('--metric', type=str, default = "rouge-2",
                    choices=["rouge-1", "rouge-2", "rouge-l", "mean-rouge", "bertscore", "moverscore", "bartscore", "a3cu",
                             "supert", "bartscore-source", "summac", "gptscore", "gpt-likert"])
parser.add_argument('--openai_model', type=str, default = "gpt-3.5-turbo-0125",
                    choices=["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview"])
parser.add_argument('--process_empty_summaries', type=boolean_string, default = False)
parser.add_argument('--compute_scores', type=boolean_string, default = False)
parser.add_argument('--keep_within_window', type=boolean_string, default = True)
parser.add_argument('--save_scores', type=boolean_string, default = False)
parser.add_argument('--bin_size', type=int, default = 20)

args = parser.parse_args()

settle_args(args)
args.max_tokens_count = args.context_length - 64 - args.max_summary_length


def main(args):
    print(args)

    seed_everything(args)

    # load the data
    texts, labels = load_data(args)
    queries = [""] * len(texts)
    if args.dataset == "middlesum":
        (texts, queries) = texts
    summaries = load_pred(args)
    texts = texts[:args.analysis_size]
    queries = queries[:args.analysis_size]
    labels = labels[:args.analysis_size]
    summaries = summaries[:args.analysis_size]
    print(len(texts), len(queries), len(labels), len(summaries))
    print("*" * 50 + " First source:")
    print(texts[0][:500])
    print("*" * 50 + " First label:")
    print(" ".join(labels[0].split()))
    print("*" * 50 + " First prediction:")
    print(" ".join(summaries[0].split()))

    # tokenizer
    tokenizer = None
    if not(args.model.startswith("gpt-3.5")):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=hf_token,
            cache_dir=f"{args.root}/hf_models/{args.model}",
            trust_remote_code=True
        )
    else:
        encoding = tiktoken.encoding_for_model(args.model)
    n_sents = np.array([len(sent_tokenize(x)) for x in tqdm(texts)])
    print(f"\nAvg # sents in source: {np.mean(n_sents):.4f}")

    # compute labels alignments
    folder = f"alignments/{args.dataset_name}/{args.subset}"
    aligned_labels_path = f"{folder}/{args.subset}_labels_alignment_{args.max_size}.pkl"
    if args.compute_alignment:
        aligned_labels = align_source_preds(texts, labels, args)
        pickle.dump(aligned_labels, open(aligned_labels_path, "wb"))
        print(f"Wrote labels alignment to: {aligned_labels_path}")
    else:
        aligned_labels = pickle.load(open(aligned_labels_path, "rb"))
        print(f"Loaded labels alignments from {aligned_labels_path}")
        aligned_labels = aligned_labels[:args.analysis_size]

    mean_idx = np.array([np.mean(x[0]) for x in aligned_labels])
    p10 = np.percentile(mean_idx, 10)
    q1 = np.percentile(mean_idx, 25)
    median = np.median(mean_idx)
    mean = np.mean(mean_idx)
    q3 = np.percentile(mean_idx, 75)
    p90 = np.percentile(mean_idx, 90)
    print("\nStats on average positions of aligned label sentences:")
    print(f"10th percentile: {p10:.4f}")
    print(f"Q1: {q1:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Mean: {mean:.4f}")
    print(f"Q3: {q3:.4f}")
    print(f"90th percentile: {p90:.4f}")

    mean_pos, std_pos, max_pos = [], [], []
    for i in tqdm(range(len(texts))):
        sents = sent_tokenize(texts[i])
        sent_idx_to_words = {}
        words_count = 0
        for j in range(len(sents)):
            #words = word_tokenize(sents[j])
            if not(args.model.startswith("gpt-3.5")):
                words = tokenizer(sents[j])["input_ids"]
            else:
                words = encoding.encode(sents[j])
            words_count += len(words)
            sent_idx_to_words[j] = words_count
        sents_idx = aligned_labels[i][0]
        word_pos = [sent_idx_to_words[x] for x in sents_idx]
        word_pos = np.array(word_pos)
        pos = np.mean(word_pos)
        mean_pos.append(pos)
        spread = np.std(word_pos)
        std_pos.append(spread)
        m = np.max(word_pos)
        max_pos.append(m)
    mean_pos = np.array(mean_pos)
    std_pos = np.array(std_pos)
    max_pos = np.array(max_pos)

    scores = compute_scores(texts, queries, summaries, labels, args, control=args.use_control)
    print(scores[:10])
    print(mean_idx.shape, mean_pos.shape, scores.shape)
    if args.metric == "gpt-likert":
        idx = scores != -1
        scores = scores[idx]
        mean_idx = mean_idx[idx]
        mean_pos = mean_pos[idx]
        std_pos = std_pos[idx]
        max_pos = max_pos[idx]
        print(mean_idx.shape, mean_pos.shape, scores.shape)

    if args.keep_within_window:
        idx = mean_pos <= args.max_tokens_count
        mean_idx = mean_idx[idx]
        mean_pos = mean_pos[idx]
        scores = scores[idx]
        print(mean_idx.shape, mean_pos.shape, scores.shape)

    threshs = [0]
    print(f"\nMean {args.metric} for each 10% bin of mean pos of aligned sentences")
    n_bins = int(100 / args.bin_size)
    for j in range(n_bins):
        low = np.percentile(mean_pos, args.bin_size*j)
        high = np.percentile(mean_pos, args.bin_size*(j+1))
        threshs.append(high)
        idx_j = np.arange(len(mean_pos))[(mean_pos >= low) * (mean_pos < high)]
        count = len(idx_j)
        mean_pos_j = np.mean(mean_pos[idx_j])
        mean_scores_j = np.mean(scores[idx_j])
        std_scores_j = np.std(scores[idx_j])
        print(f"Bin {j+1}, mean pos between {low:.2f} and {high:.2f} (avg: {mean_pos_j:.2f}), count: {count}, avg {args.metric} is {mean_scores_j:.4f} (std: {std_scores_j:.4f})")

    idx = 0
    while idx < len(threshs) and threshs[idx] <= args.max_tokens_count:
        idx += 1
    idx -= 1
    thresh = threshs[idx]
    mean_idx_first_bins = mean_idx[mean_pos <= thresh]
    scores_first_bins = scores[mean_pos <= thresh]
    print("Data points kept within visible window", mean_idx_first_bins.shape, scores_first_bins.shape)
    spearman_corr, spearman_p = spearmanr(mean_idx_first_bins, scores_first_bins)
    print(f"Spearman corr coefficient: {spearman_corr:.6f}, p-value: {spearman_p:.6f}")


if __name__ == '__main__':
    main(args)

