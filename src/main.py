# This script controls evaluation: running any metric on any LLM in any setup on any dataset

import argparse
import numpy as np
from tqdm import tqdm
import time

from keys import mathieu_key
from utils import seed_everything, settle_args, load_data,  load_pred
from evaluation import compute_scores


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = "/data/mathieu")
parser.add_argument('--dataset', type=str, default = "samsum",
                    choices = ["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience", "middlesum"])
parser.add_argument('--subset', type=str, default = "test")
parser.add_argument('--instruction_position', type=str, default = "post",
                    choices=["pre", "post"])
parser.add_argument('--focus_prompt', type=bool, default = False)
parser.add_argument('--max_size', type=int, default = 1000)
parser.add_argument('--multi_doc_split', type=str, default = "|||||")
parser.add_argument('--use_control', type=bool, default = False)
parser.add_argument('--control', type=str, default = "default",
                    choices = ["default", "shuffle", "position", "filling"])
parser.add_argument('--random_baseline', type=bool, default = False) # only for control=default
parser.add_argument('--n_shuffle_perm', type=int, default = 0) # only for control=shuffle
parser.add_argument('--control_doc_pos', type=int, default = 0) # only for control=pos
parser.add_argument('--control_label', type=str, default = "label")
parser.add_argument('--swap_docs', type=bool, default = False)
parser.add_argument('--oracle_n_sents', type=bool, default = False)
parser.add_argument('--oracle_n_words', type=bool, default = False)
parser.add_argument('--analysis_size', type=int, default = 1000)
parser.add_argument('--clean_model_name', type=str, default = "llama_2_7b",
                    choices = ["llama_2_7b_base", "llama_2_13b_base",
                             "flan_ul2", "llama_2_7b", "llama_2_13b", "xgen_7b", "mistral_7b",
                             "vicuna_7b_16k", "llama_2_7b_32k",
                             "gpt-3.5-turbo-0125"])
parser.add_argument('--inference_method', type=str, default = "normal",
                    choices = ["normal", "pyramidal", "incremental"])
parser.add_argument('--decoding_method', type=str, default = "top_k",
                    choices = ["greedy", "beam_search", "top_k", "top_p", "temperature"])
parser.add_argument('--enforced_max_length', type=int, default = -1) # [-1, 512, 1024, 2048, 4096, 6144, 8192, 10240]
parser.add_argument('--metric', type=str, default = "rouge-2",
                    choices = ["sents", "words", "abstractiveness",
                    "rouge-1", "rouge-2", "rouge-l", "mean-rouge", "bertscore", "a2cu", "a3cu",
                    "summac", "gpt-likert",
                    "gpt-likert-informativeness", "gpt-likert-quality", "gpt-likert-coherence", "gpt-likert-attributable"])
parser.add_argument('--openai_model', type=str, default = "gpt-3.5-turbo-0125",
                    choices = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview"])
parser.add_argument('--openai_key', type=str, default = mathieu_key)
parser.add_argument('--process_empty_summaries', type=bool, default = False)
parser.add_argument('--compute_scores', type=bool, default = True)
parser.add_argument('--save_scores', type=bool, default = True)
parser.add_argument('--bootstrap', type=bool, default = True)
parser.add_argument('--bootstrap_size', type=int, default = 100)
parser.add_argument('--bootstrap_rounds', type=int, default = 1000)
# general stuff
parser.add_argument('--n_bins', type=int, default = 5)

args = parser.parse_args()

settle_args(args)
args.bin_size = int(100 / args.n_bins)
args.evaluate = True


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

    # score summaries
    if args.random_baseline:
        print("\nShuffling the predictions")
        p = np.random.permutation(len(summaries))
        summaries = [summaries[x] for x in p]

    print(len(texts), len(queries), len(summaries), len(labels))
    scores = compute_scores(texts, queries, summaries, labels, args, control=args.use_control)

    if args.dataset == "middlesum":
        for dataset in ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]:
            idx = [i for i in range(len(scores)) if queries[i] == dataset]
            scores_dataset = scores[idx]
            print(f"On dataset {dataset} (size: {len(idx)}), performance is {np.mean(scores_dataset):.4f}")
    if args.bootstrap:
        vals = []
        for i in tqdm(range(args.bootstrap_rounds)):
            p = np.random.permutation(len(scores))
            p = p[:args.bootstrap_size]
            scores_p = scores[p]
            val = np.mean(scores_p)
            vals.append(val)
        vals = np.array(vals)
        m, s = np.mean(vals), np.std(vals)
        print(f"Mean score after {args.bootstrap_rounds} bootstrapping of size {args.bootstrap_size}: {m:.4f}, std: {s:.4f}, lower: {(m-s):.4f}, higher: {(m+s):.4f}")


if __name__ == '__main__':
    main(args)

