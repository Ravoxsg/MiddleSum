# RQ1: distribution of generated summary bigrams within the source

import argparse
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

from keys import root
from utils import boolean_string, seed_everything, settle_args, load_data, load_pred


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = root)
parser.add_argument('--dataset', type=str, default = "xsum",
                    choices=["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience", "middlesum"])
parser.add_argument('--subset', type=str, default = "test")
parser.add_argument('--subset_size', type=int, default = -1)
parser.add_argument('--instruction_position', type=str, default = "post",
                    choices=["pre", "post"])
parser.add_argument('--focus_prompt', type=boolean_string, default = False)
parser.add_argument('--max_size', type=int, default = 1000)
parser.add_argument('--multi_doc_split', type=str, default = "|||||")
parser.add_argument('--use_control', type=boolean_string, default = False)
parser.add_argument('--control', type=str, default = "default",
                    choices = ["default", "shuffle", "position", "filling"])
parser.add_argument('--random_baseline', type=boolean_string, default = False) # only for control=default
parser.add_argument('--n_shuffle_perm', type=int, default = 1) # only for control=shuffle
parser.add_argument('--control_doc_pos', type=int, default = 0) # only for control=pos
parser.add_argument('--control_label', type=str, default = "label")
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
parser.add_argument('--enforced_max_length', type=float, default = -1) # [-1, 512, 1024, 2048, 4096, 6144, 8192, 10240]
parser.add_argument('--n_bins', type=int, default = 20)
parser.add_argument('--statistical_significance', type=boolean_string, default = True)

args = parser.parse_args()

settle_args(args)


def main(args):
    print(args)

    seed_everything(args)

    texts, labels = load_data(args)
    queries = [""] * len(texts)
    if args.summarization_type == "query":
        (texts, queries) = texts
    summaries = load_pred(args, control=args.use_control)
    texts = texts[:args.analysis_size]
    queries = queries[:args.analysis_size]
    labels = labels[:args.analysis_size]
    summaries = summaries[:args.analysis_size]
    print(len(texts), len(queries), len(labels), len(summaries))
    print("*"*50 + " First source:")
    print(texts[0][:500])
    print("*"*50 + " First label:")
    print(" ".join(labels[0].split()))
    print("*"*50 + " First prediction:")
    print(" ".join(summaries[0].split()))
    print(len(texts), len(queries), len(summaries), len(labels))

    all_n_labels_words, all_n_labels_bigrams = 0, 0
    all_n_preds_words, all_n_preds_bigrams = 0, 0
    label_saliency, preds_saliency = [], []
    count = 0
    label_failed, preds_failed = 0, 0
    labels_obs, preds_obs = [], []
    for i in tqdm(range(len(texts))):
        if len(word_tokenize(texts[i])) <= args.n_bins:
            continue
        
        n_labels_words, n_labels_bigrams, bins, label_failed, obs = get_bigrams_position(texts[i], labels[i], args.n_bins, label_failed)
        all_n_labels_words += n_labels_words
        all_n_labels_bigrams += n_labels_bigrams
        label_saliency.append(np.expand_dims(bins, 0))
        labels_obs += obs

        n_preds_words, n_preds_bigrams, bins, preds_failed, obs = get_bigrams_position(texts[i], summaries[i], args.n_bins, preds_failed)
        all_n_preds_words += n_preds_words
        all_n_preds_bigrams += n_preds_bigrams
        preds_saliency.append(np.expand_dims(bins, 0))
        preds_obs += obs
        count += 1

    n_labels_words = all_n_labels_words / count
    n_labels_bigrams = all_n_labels_bigrams / count
    print(f"\nLabels: # words: {n_labels_words:.2f}, unique bigrams: {n_labels_bigrams:.2f}")
    
    n_preds_words = all_n_preds_words / count
    n_preds_bigrams = all_n_preds_bigrams / count
    print(f"\nPredictions: # words: {n_preds_words:.2f}, unique bigrams: {n_preds_bigrams:.2f}")
    
    label_saliency = np.concatenate(label_saliency, 0)
    preds_saliency = np.concatenate(preds_saliency, 0)
    print(label_saliency.shape, preds_saliency.shape)
    mean_label_saliency = np.mean(label_saliency, 0)
    mean_preds_saliency = np.mean(preds_saliency, 0)
    
    print(f"\n# Data points where cannot map the source bigrams distribution to labels: {label_failed} or {100*label_failed/len(texts):.4f}%")
    print("Position of label bigrams in text:")
    clean_label_saliency = np.round(mean_label_saliency, 2)
    print(clean_label_saliency)
    print(f"Sums up to {np.sum(mean_label_saliency):.4f}")
    
    print(f"\n# Data points where cannot map the source bigrams distribution to preds: {preds_failed} or {100*preds_failed/len(texts):.4f}%")
    print("Position of predicted bigrams in text:")
    clean_preds_saliency = np.round(mean_preds_saliency, 2)
    print(clean_preds_saliency)
    print(f"Sums up to {np.sum(mean_preds_saliency):.4f}")

    if args.dataset == "middlesum":
        for dataset in ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]:
            idx = [i for i in range(len(queries)) if queries[i] == dataset]
            label_saliency_dataset = label_saliency[idx]
            preds_saliency_dataset = preds_saliency[idx]
            mean_label_saliency_dataset = np.mean(label_saliency_dataset, 0)
            mean_preds_saliency_dataset = np.mean(preds_saliency_dataset, 0)
            print(f"\n{dataset} subset: Position of label bigrams in text:")
            clean_label_saliency_dataset = np.round(mean_label_saliency_dataset, 2)
            print(clean_label_saliency_dataset)
            print(f"{dataset} subset: Position of predicted bigrams in text:")
            clean_preds_saliency_dataset = np.round(mean_preds_saliency_dataset, 2)
            print(clean_preds_saliency_dataset)

    if args.statistical_significance:
        # KS 2-sample test
        labels_obs = np.array(labels_obs)
        preds_obs = np.array(preds_obs)
        l = min(len(labels_obs), len(preds_obs))
        uniform_obs = np.random.uniform(0, 1, size=l)
        stat, p = stats.ks_2samp(labels_obs, uniform_obs, alternative='two-sided')
        print(f"\ntwo-sample Kolmogorov-Smirnov between label distribution and uniform distribution: {p:.20f}")
        stat, p = stats.ks_2samp(preds_obs, uniform_obs, alternative='two-sided')
        print(f"two-sample Kolmogorov-Smirnov between LLM distribution and uniform distribution: {p:.20f}")
        stat, p = stats.ks_2samp(preds_obs, labels_obs, alternative='two-sided')
        print(f"two-sample Kolmogorov-Smirnov between LLM distribution and label distribution: {p:.20f}")

def get_bigrams_position(text, summary, n_bins, failed):
    trunc_text = text

    bigrams = {}
    summary_words = word_tokenize(summary.lower())
    n_summary_words = len(summary_words)
    for i in range(len(summary_words)-1):
        bigram = summary_words[i] + " " + summary_words[i+1]
        bigrams[bigram] = 0
    n_summary_bigrams = len(bigrams.keys())

    text_words = word_tokenize(trunc_text.lower())
    bin_size = int(len(text_words) / n_bins)
    all_obs = []
    if bin_size > 0:
        bins = np.zeros(n_bins)
        n_found = 0
        for i in range(len(text_words)-1):
            bigram = text_words[i] + " " + text_words[i+1]
            if bigram in bigrams.keys():
                obs = i / max(1, len(text_words)-1)
                bin = int(i / bin_size)
                if bin <= n_bins-1:
                    bins[bin] += 1
                    n_found += 1
                    all_obs.append(obs)
        if n_found > 0:
            bins /= n_found
            bins *= 100
        else:
            failed += 1
            bins = 100 * np.ones(n_bins)/n_bins
            all_obs += list(np.random.uniform(0,1,n_bins))
    else:
        failed += 1
        bins = 100 * np.ones(n_bins)/n_bins
        all_obs += list(np.random.uniform(0,1,n_bins))

    return n_summary_words, n_summary_bigrams, bins, failed, all_obs


if __name__ == '__main__':
    main(args)

