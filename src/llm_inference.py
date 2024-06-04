# Run inference with the LLMs on the 10 datasets.

import argparse
import numpy as np 
import pickle
import os
import time
import torch
import tiktoken
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from keys import root, openai_key
from utils import seed_everything, settle_args, save_pred
from engine import prepare_model, load_raw_data, prepare_texts, prepare_prompts, prepare_texts_gptlikert
from engine import run_inference, run_pyramidal_inference, run_incremental_inference
from evaluation import complete_evaluation


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--root', type=str, default = root)
parser.add_argument('--dataset', type=str, default = "govreport",
                    choices = ["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience", "middlesum"])
parser.add_argument('--subset', type=str, default = "test")
parser.add_argument('--subset_size', type=int, default = -1)
parser.add_argument('--max_size', type=int, default = 1000) # cap subset size to 1000
parser.add_argument('--multi_doc_split', type=str, default = "|||||")
parser.add_argument('--instruction_position', type=str, default = "post",
                    choices=["pre", "post"])
parser.add_argument('--focus_prompt', type=bool, default = False)
parser.add_argument('--use_control', type=bool, default = False)
parser.add_argument('--control', type=str, default = "") # controlling the input (e.g, keeping a single salient document) happens in llm_control_inference.py
parser.add_argument('--swap_docs', type=bool, default = False)
parser.add_argument('--oracle_n_sents', type=bool, default = False)
parser.add_argument('--oracle_n_words', type=bool, default = False)
parser.add_argument('--clean_model_name', type=str, default = "llama_2_7b",
                    choices = ["llama_2_7b_base", "llama_2_13b_base",
                               "flan_ul2", "llama_2_7b", "llama_2_13b", "xgen_7b", "mistral_7b",
                               "vicuna_7b_16k", "llama_2_7b_32k",
                               "gpt-3.5-turbo-0125"])
parser.add_argument('--openai_key', type=str, default = openai_key)
parser.add_argument('--torch_dtype', default = torch.bfloat16, 
                    choices = [torch.float16, torch.bfloat16, torch.float32])
parser.add_argument('--inference_method', type=str, default = "normal",
                    choices = ["normal", "pyramidal", "incremental"])
parser.add_argument('--n_words_per_block', type=int, default = 1500)
parser.add_argument('--max_n_blocks', type=int, default = 8)
parser.add_argument('--decoding_method', type=str, default = "top_k",
                    choices = ["greedy", "beam_search", "top_k", "top_p", "temperature"])
parser.add_argument('--enforced_max_length', type=float, default = -1)
parser.add_argument('--check_trunc_fraction', type=bool, default = False)
parser.add_argument('--temperature', type=float, default = 0.3)
parser.add_argument('--top_k', type=int, default = 50)
parser.add_argument('--top_p', type=float, default = 0.95)
parser.add_argument('--print_input', type=bool, default = False)
parser.add_argument('--print_output', type=bool, default = False)
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

    # data
    texts, labels = load_raw_data(args)
    queries = [""] * len(labels)
    if args.summarization_type == "query":
        (texts, queries) = texts
    if args.model.startswith("gpt-3.5"):
        encoding = tiktoken.encoding_for_model(args.model)
        args.gpt_model_max_length = args.context_length
        trunc_texts = prepare_texts_gptlikert(encoding, texts, queries, args)
    else:
        trunc_texts = prepare_texts(texts, queries, tokenizer, args)
    mean_n_words = np.mean(np.array([len(word_tokenize(x)) for x in tqdm(trunc_texts)]))
    print(f"Mean # words / truncated text: {mean_n_words:.4f}")
    prompts = prepare_prompts(trunc_texts, queries, labels, args)
    if args.check_trunc_fraction:
        trunc_sizes = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
        lengths = []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            text_ids = tokenizer(text)["input_ids"]
            lengths.append(len(text_ids))
        for trunc_size in trunc_sizes:
            max_length = trunc_size - 64 - args.max_summary_length
            fracs = []
            for i in tqdm(range(len(texts))):
                length = lengths[i]
                frac = 100 * min(max_length, length) / max(1, length)
                fracs.append(frac)
            fracs = np.array(fracs)
            print(f"At trunc length {trunc_size}, we get {np.mean(fracs):.4f}% of docs")

    # inference
    if args.inference_method == "normal":
        summaries = run_inference(prompts, tokenizer, model, args)
    elif args.inference_method == "pyramidal":
        summaries = run_pyramidal_inference(texts, queries, tokenizer, model, args)
    elif args.inference_method == "incremental":
        summaries = run_incremental_inference(texts, queries, tokenizer, model, args)

    # eval
    scores = complete_evaluation(summaries, labels, args)
    print(f"R-1: {np.mean(scores[:, 0]):.2f} | R-2: {np.mean(scores[:, 1]):.2f} | R-L: {np.mean(scores[:, 2]):.2f}")
    if args.dataset == "middlesum":
        for dataset in ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]:
            idx = [i for i in range(len(queries)) if queries[i] == dataset]
            r1_dataset = scores[idx, 0]
            r2_dataset = scores[idx, 1]
            rl_dataset = scores[idx, 2]
            print(f"For {dataset} ({len(idx)} data points): R-1: {np.mean(r1_dataset):.2f} | R-2: {np.mean(r2_dataset):.2f} | R-L: {np.mean(rl_dataset):.2f}")

    # export
    if args.save:
        size = len(texts)
        folder = f"summaries/{args.dataset_name}/{args.subset}"
        os.makedirs(folder, exist_ok=True)
        texts_path = f"{folder}/{args.subset}_texts_{size}.pkl"
        pickle.dump(texts, open(texts_path, "wb"))
        if args.summarization_type == "query":
            queries_path = f"{folder}/{args.subset}_queries_{size}.pkl"
            pickle.dump(queries, open(queries_path, "wb"))
        labels_path = f"{folder}/{args.subset}_labels_{size}.pkl"
        pickle.dump(labels, open(labels_path, "wb"))
        save_pred(summaries, args)


if __name__ == '__main__':
    main(args)
