# Run inference with the LLMs on the 10 datasets while controlling some aspect: order of docs, etc
# Used in the following analysis:
# Controlling the position of an important document (args.control = "position")
# Filling the middle of the prompt with noisy document (args.control = "filling")

import argparse
import numpy as np
import pickle
import os
import time
import torch
import tiktoken

from keys import mathieu_key
from utils import seed_everything, settle_args, save_pred, get_clean_control_name
from engine import prepare_model, load_raw_data, prepare_prompts, run_inference
from engine_control import filter_n_docs, prepare_control_texts
from evaluation import complete_evaluation


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = "/data/mathieu")
parser.add_argument('--dataset', type=str, default = "multinews",
                    choices = ["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience"])
parser.add_argument('--subset', type=str, default = "test")
parser.add_argument('--subset_size', type=int, default = -1)
parser.add_argument('--random_baseline', type=bool, default = False)
parser.add_argument('--control_n_docs', type=bool, default = True)
parser.add_argument('--min_n_docs', type=int, default = 5)
parser.add_argument('--max_size', type=int, default = 100000) # set it > subset_size to keep the whole subset, which we subsample later
parser.add_argument('--multi_doc_split', type=str, default = "|||||")
parser.add_argument('--instruction_position', type=str, default = "post",
                    choices=["pre", "post"])
parser.add_argument('--focus_prompt', type=bool, default = False)
parser.add_argument('--max_control_size', type=int, default = 500)
parser.add_argument('--control', type=str, default = "filling",
                    choices = ["default", "shuffle", "position", "filling"])
parser.add_argument('--n_shuffle_perm', type=int, default = 1) # only for control=shuffle
parser.add_argument('--control_doc_pos', type=int, default = 0) # only for control=pos
parser.add_argument('--swap_docs', type=bool, default = True) # only for control=pos or control=filling
parser.add_argument('--control_metric', type=str, default = "bertscore")
parser.add_argument('--control_label', type=str, default = "label", choices=["label", "query"])
parser.add_argument('--oracle_n_sents', type=bool, default = False)
parser.add_argument('--oracle_n_words', type=bool, default = False)
parser.add_argument('--model', type=str, default = "gpt-3.5-turbo-0125",
                    choices=["flan-ul2", "llama-2-7b-chat", "llama-2-13b-chat", "xgen-7b", "mistral-7b",
                            "gpt-3.5-turbo-0125"])
parser.add_argument('--openai_key', type=str, default = mathieu_key)
parser.add_argument('--torch_dtype', default = torch.bfloat16, 
                    choices = [torch.float16, torch.bfloat16, torch.float32])
parser.add_argument('--inference_method', type=str, default = "normal",
                    choices = ["normal", "pyramidal"])
parser.add_argument('--decoding_method', type=str, default = "top_k",
                    choices = ["greedy", "beam_search", "top_k", "top_p", "temperature"])
parser.add_argument('--enforced_max_length', type=float, default = -1)
parser.add_argument('--temperature', type=float, default = 0.3)
parser.add_argument('--top_k', type=int, default = 50)
parser.add_argument('--top_p', type=float, default = 0.9)
parser.add_argument('--print_input', type=bool, default = False)
parser.add_argument('--print_output', type=bool, default = False)
parser.add_argument('--check_output', type=bool, default = False)
parser.add_argument('--metrics', type=list, default = ["rouge-1", "rouge-2", "rouge-l"])
parser.add_argument('--save', type=bool, default = True)

args = parser.parse_args()

settle_args(args)
args.evaluate = False


def main(args):
    # check args
    print(args)

    # seed
    seed_everything(args)

    # model and tokenizer
    tokenizer, model = None, None
    if not(args.model.startswith("gpt-3.5")):
        tokenizer, model = prepare_model(args)
        n_params = sum(p.numel() for p in model.parameters())
        print("\nThe model has {} parameters".format(n_params))
        params = list(model.parameters())[0]
        print(params.dtype)
    else:
        tokenizer = tiktoken.encoding_for_model(args.model)
        args.gpt_model_max_length = args.context_length

    # data
    texts, labels = load_raw_data(args)
    queries = [""] * len(labels)
    if args.summarization_type == "query":
        (texts, queries) = texts
    print(len(texts), len(queries), len(labels))
    print("\nFirst data point:")
    print(texts[0][:500] + "\n\n")
    print(labels[0])

    # count # docs (for MDS)
    if args.summarization_input == "multi":
        n_docs_distr = {}
        for text in texts:
            n_docs = len(text.split(args.multi_doc_split))
            if not(n_docs in n_docs_distr.keys()):
                n_docs_distr[n_docs] = 0
            n_docs_distr[n_docs] += 1
        n_docs = list(n_docs_distr.keys())
        n_docs.sort()
        for k in n_docs:
            print(f"There are {n_docs_distr[k]} data points with {k} docs")

    # filter data points, shuffle & subsample
    if args.summarization_input == "multi" and args.control_n_docs:
        print("\nControlling the number of documents...")
        texts, queries, labels = filter_n_docs(texts, queries, labels, args)
    p = np.random.permutation(len(texts))
    texts = [texts[x] for x in p[:args.max_control_size]]
    queries = [queries[x] for x in p[:args.max_control_size]]
    labels = [labels[x] for x in p[:args.max_control_size]]
    print(len(texts), len(queries), len(labels))
    print("\nFirst data point for CONTROL experiment:")
    print(texts[0][:500] + "\n\n")
    print(labels[0])

    # prepare text for LLM controlled inference
    trunc_texts, just_relevant_texts = prepare_control_texts(texts, queries, labels, tokenizer, args)
    prompts = prepare_prompts(trunc_texts, queries, labels, args)

    # inference
    new_summaries = run_inference(prompts, tokenizer, model, args)

    if args.check_output:
        for i in range(5):
            print("*"*10)
            print(new_summaries[i])

    # evaluation
    scores = complete_evaluation(new_summaries, labels, args)
    print(f"R-1: {np.mean(scores[:, 0]):.2f} | R-2: {np.mean(scores[:, 1]):.2f} | R-L: {np.mean(scores[:, 2]):.2f}")

    # export
    if args.save:
        size = len(texts)
        folder = f"summaries/{args.dataset_name}/{args.subset}"
        os.makedirs(folder, exist_ok=True)
        control = get_clean_control_name(args)

        texts_path = f"{folder}/{args.subset}_texts_{control}_{size}.pkl"
        texts_to_save = trunc_texts
        if args.control in ["position", "filling"]:
            print(f"\nSaving only relevant docs from the input...")
            texts_to_save = just_relevant_texts
        pickle.dump(texts_to_save, open(texts_path, "wb"))
        if args.summarization_type == "query":
            queries_path = f"{folder}/{args.subset}_queries_{control}_{size}.pkl"
            pickle.dump(queries, open(queries_path, "wb"))
        labels_path = f"{folder}/{args.subset}_labels_{control}_{size}.pkl"
        pickle.dump(labels, open(labels_path, "wb"))
        print(f"Saved labels to: {labels_path}")
        save_pred(new_summaries, args, control=True)


if __name__ == '__main__':
    main(args)
