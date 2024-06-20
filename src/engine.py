import torch
import numpy as np
import pickle
import gc
import os 
import tiktoken
import time
import openai
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from datasets import load_dataset
from bert_score import score
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from keys import hf_token


def prepare_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token = hf_token,
        cache_dir = f"{args.root}/hf_models/{args.model}",
        trust_remote_code = True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map = "auto",
        torch_dtype = args.torch_dtype,
        token = hf_token,
        cache_dir = f"{args.root}/hf_models/{args.model}"
    )

    return tokenizer, model

def load_raw_data(args):
    # HuggingFace datasets
    if args.dataset_proper_name != "_":
        dataset_args = [args.dataset_proper_name, args.dataset_version]
        data = load_dataset(*dataset_args)
        # datasets with an existing split
        if not(args.dataset in ["reddit"]):
            val_data = data[args.subset_key]
            texts = [val_data[i][args.text_key] for i in range(len(val_data))]
            labels = [val_data[i][args.summary_key] for i in range(len(val_data))]
        # datasets without a train/val/test split: ["reddit"]
        else:
            data = data["train"]
            n_docs = len(data[args.text_key])
            texts = data[args.text_key]
            summaries = data[args.summary_key]
            texts = [texts[i] for i in tqdm(range(n_docs))]
            labels = [summaries[i] for i in tqdm(range(n_docs))]
            p = np.random.permutation(len(texts))
            texts = [texts[x] for x in p]
            labels = [labels[x] for x in p]
            thresh1 = int(0.8 * len(texts))
            thresh2 = int(0.9 * len(texts))
            if args.subset == "train":
                begin = 0
                end = thresh1
            elif args.subset == "val":
                begin = thresh1
                end = thresh2
            else:
                begin = thresh2
                end = len(labels)
            texts = texts[begin:end]
            labels = labels[begin:end]
    # datasets not in HuggingFace:
    else:
        if args.dataset in ["summscreen", "multixscience"]:
            texts_path = f"raw_summaries/{args.dataset_name}/{args.subset}/{args.subset}_texts_{args.subset_size}.pkl"
            texts = pickle.load(open(texts_path, "rb"))
            labels_path = f"raw_summaries/{args.dataset_name}/{args.subset}/{args.subset}_labels_{args.subset_size}.pkl"
            labels = pickle.load(open(labels_path, "rb"))
        elif args.dataset == "middlesum":
            path = "MiddleSum/middlesum.jsonl"
            data_points = [json.loads(file) for file in open(path, 'r')]
            texts = [x["source"]  for x in data_points]
            labels = [x["label"] for x in data_points]
            queries = [x["dataset"] for x in data_points]
    p = np.random.permutation(len(labels))
    texts = [texts[x] for x in p[:args.max_size]]
    labels = [labels[x] for x in p[:args.max_size]]
    print("\nFirst data point:")
    print(texts[0][:500] + "\n\n")
    if args.summarization_type == "query":
        queries = [queries[x] for x in p[:args.max_size]]
        texts = (texts, queries)

    return texts, labels

def prepare_texts(texts, queries, tokenizer, args):
    margin = 64
    max_length = args.context_length - margin
    if args.enforced_max_length > 0:
        max_length = min(max_length, args.enforced_max_length)

    trunc_texts = []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        if args.dataset == "middlesum":
            query = queries[i]
            dataset_name = query
            dataset_names = ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]
            index = dataset_names.index(dataset_name)
            max_summary_lengths = [512, 512, 768, 512, 512]
            summarization_inputs = ["single", "single", "single", "single", "multi"]
            args.max_summary_length = max_summary_lengths[index]
            args.summarization_input = summarization_inputs[index]

        if args.summarization_input == "single":
            text_ids = tokenizer(text)["input_ids"]
            text_ids = text_ids[:(max_length - args.max_summary_length)]
            trunc_text = tokenizer.decode(text_ids, skip_special_tokens=True)
        elif args.summarization_input == "multi":
            docs = text.split(args.multi_doc_split)
            tokens_per_doc = int((max_length - args.max_summary_length - 4 * len(docs)) / len(docs))
            trunc_text = ""
            for doc in docs:
                doc_ids = tokenizer(doc)["input_ids"]
                doc_ids = doc_ids[:tokens_per_doc]
                decoded_doc = " [DOCUMENT] " + tokenizer.decode(doc_ids, skip_special_tokens=True)
                trunc_text += decoded_doc
            trunc_text = trunc_text[1:]
        trunc_texts.append(trunc_text)

    return trunc_texts

def prepare_texts_gptlikert(encoding, texts, queries, args):
    margin = 64 #384
    if args.focus_prompt:
        margin = 384
    max_length = args.gpt_model_max_length - margin
    if not(args.evaluate) and args.enforced_max_length > 0:
        max_length = min(max_length, args.enforced_max_length)
    max_length = int(max_length)
    print(max_length)

    trunc_texts = []
    n_tokens, n_tokens_kept = [], []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        if args.dataset == "middlesum":
            dataset = queries[i]
            datasets = ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]
            index = datasets.index(dataset)
            max_summary_lengths = [512, 512, 768, 512, 512]
            summarization_inputs = ["single", "single", "single", "single", "multi"]
            args.max_summary_length = max_summary_lengths[index]
            args.summarization_input = summarization_inputs[index]

        if args.summarization_input == "single":
            text_ids = encoding.encode(text)
            n_tokens.append(len(text_ids))
            text_ids = text_ids[:(max_length - args.max_summary_length)]
            n_tokens_kept.append(len(text_ids))
            trunc_text = encoding.decode(text_ids)
        elif args.summarization_input == "multi":
            if args.use_control and args.control != "position":
                docs = text.split("[DOCUMENT]")
            else:
                docs = text.split(args.multi_doc_split)
            tokens_per_doc = int((max_length - args.max_summary_length - 4 * len(docs)) / len(docs))
            trunc_text = ""
            temp_n_tokens, temp_n_tokens_kept = 0, 0
            for doc in docs:
                doc_ids = encoding.encode(doc)
                temp_n_tokens += len(doc_ids)
                doc_ids = doc_ids[:tokens_per_doc]
                temp_n_tokens_kept += len(doc_ids)
                decoded_doc = " [DOCUMENT] " + encoding.decode(doc_ids)
                trunc_text += decoded_doc
            trunc_text = trunc_text[1:]
            n_tokens.append(temp_n_tokens)
            n_tokens_kept.append(temp_n_tokens_kept)
        trunc_texts.append(trunc_text)
    n_tokens = np.array(n_tokens)
    n_tokens_kept = np.array(n_tokens_kept)
    print(len(n_tokens), len(n_tokens_kept))
    print(f"\nAvg # tokens: {np.mean(n_tokens):.2f} (max: {np.max(n_tokens_kept)}), avg # tokens kept: {np.mean(n_tokens_kept):.2f}")

    return trunc_texts

def prepare_prompts(trunc_texts, queries, labels, args):
    n = args.n
    length = f"{n} sentences"
    prompts = []
    for i in tqdm(range(len(trunc_texts))):
        trunc_text = trunc_texts[i]
        query = queries[i]

        # set the length
        if args.oracle_n_sents:
            n = len(sent_tokenize(labels[i]))
            length = f"{n} sentences"
        if args.oracle_n_words:
            n = len(word_tokenize(labels[i]))
            length = f"{n} words"
        if args.dataset == "middlesum":
            dataset_name = query
            dataset_names = ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]
            index = dataset_names.index(dataset_name)
            ns = [6, 7, 22, 5, 10]
            n = ns[index]
            length = f"{n} sentences"

        # build the instruction
        if args.instruction_position == "post":
            instruction_position = "above"
            if args.summarization_type == "generic" or args.dataset == "middlesum":
                instruction = f"Summarize the {instruction_position} text in {length}."
                if args.swap_docs:
                    instruction = f"Summarize the {instruction_position} text in {length}. Only pay attention to the relevant document."
                if args.focus_prompt:
                    instruction = f"Summarize the {instruction_position} text in {length}. Please also pay attention to the middle section of the input when constructing the summary."
            else:
                instruction = f"Query:\n{query}.\nSummarize the {instruction_position} text with regards to the query in {length}."
                if args.swap_docs:
                    instruction = f"Query:\n{query}.\nSummarize the {instruction_position} text with regards to the query in {length}. Only pay attention to the relevant document."
            prompt = f"Read the following text and then summarize it:\n{trunc_text}\n{instruction}\nSummary:"
        else:
            instruction_position = "following"
            if args.summarization_type == "generic" or args.dataset == "middlesum":
                instruction = f"Summarize the {instruction_position} text in {length}."
                if args.swap_docs:
                    instruction = f"Summarize the {instruction_position} text in {length}. Only pay attention to the relevant document."
                if args.focus_prompt:
                    instruction = f"Summarize the {instruction_position} text in {length}. Please also pay attention to the middle section of the input when constructing the summary."
            else:
                instruction = f"Summarize the {instruction_position} text with regards to the query in {length}.\nQuery:\n{query}."
                if args.swap_docs:
                    instruction = f"Summarize the {instruction_position} text with regards to the query in {length}. Only pay attention to the relevant document.\nQuery:\n{query}."
            prompt = f"{instruction}\nText:\n{trunc_text}\nSummary:"
        prompts.append(prompt)

    return prompts

def run_inference(prompts, tokenizer, model, args):
    print("\nRunning inference..")

    all_lengths, all_summaries = [], []
    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        if args.print_input:
            print("*" * 50)
            print(prompt)

        if args.model.startswith("gpt-3.5"):
            all_lengths.append(len(prompt.split()))
        else:
            tok = tokenizer(prompt, return_tensors="pt")
            length = tok["input_ids"].shape[1]
            all_lengths.append(length)

        # inference
        response = inference(prompt, tokenizer, model, args)
        all_summaries.append(response)

        if args.print_output:
            print("*"*50)
            print(" ".join(response.split()))
    print(f"Avg # tokens in input: {np.mean(np.array(all_lengths)):.2f}")

    return all_summaries

def run_pyramidal_inference(texts, queries, tokenizer, model, args):
    print("\nRunning pyramidal inference..")
    n_words_per_block = args.n_words_per_block
    max_n_blocks = args.max_n_blocks
    n = args.n
    length = f"{n} sentences"

    all_n_blocks, all_summaries = [], []
    for i in tqdm(range(len(texts))):
        query = queries[i]
        if args.dataset == "middlesum":
            dataset_name = query
            dataset_names = ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]
            index = dataset_names.index(dataset_name)
            ns = [6, 7, 22, 5, 10]
            n = ns[index]
            length = f"{n} sentences"

        text = texts[i]
        if i in [221]: # data point not fitting in my GPU RAM :/
            text = " ".join(sent_tokenize(text)[:300])
        sents = sent_tokenize(text)
        sent_to_word_count = {}
        word_count = 0
        for j in range(len(sents)):
            n_words = len(word_tokenize(sents[j]))
            word_count += n_words
            sent_to_word_count[j] = word_count

        n_blocks = int(np.ceil(word_count / n_words_per_block))
        n_blocks = min(max_n_blocks, n_blocks)
        all_n_blocks.append(n_blocks)
        sent_start_thresh, sent_end_thresh = 0, 0
        block_summaries = []
        for j in range(n_blocks):
            block_end_thresh = (j+1) * n_words_per_block
            while (sent_end_thresh < len(sents)) and (sent_to_word_count[sent_end_thresh] < block_end_thresh):
                sent_end_thresh += 1
            sents_block = sents[sent_start_thresh:(sent_end_thresh+1)]
            text_block = " ".join(sents_block)

            # build the prompt
            if args.instruction_position == "post":
                instruction_position = "above"
                instruction = f"Summarize the {instruction_position} text in {length}."
                prompt = f"Read the following text and then summarize it:\n{text_block}\n{instruction}\nSummary:"
            else:
                instruction_position = "following"
                instruction = f"Summarize the {instruction_position} text in {length}."
                prompt = f"{instruction}\nText:\n{text_block}\nSummary:"
            
            # inference
            response = inference(prompt, tokenizer, model, args)
            block_summaries.append(response)
            sent_start_thresh = sent_end_thresh + 1

        # build prompt for overall summary
        prompt = "Here is a list of summaries from consecutive text blocks:"
        for j in range(len(block_summaries)):
            prompt += f"\nSummary {j}:\n{block_summaries[j]}"
        prompt += "\nPlease summarize them into an overall summary.\nSummary:"

        # overall summary inference
        response = inference(prompt, tokenizer, model, args)
        all_summaries.append(response)

        if args.print_output:
            print("*" * 10)
            for j in range(len(block_summaries)):
                print(f"Summary {j}")
                print(block_summaries[j])
            print("Overall summary:")
            print(" ".join(response.split()))
    print(f"Avg # blocks: {np.mean(np.array(all_n_blocks)):.2f}")

    return all_summaries

def run_incremental_inference(texts, queries, tokenizer, model, args):
    print("\nRunning incremental inference..")
    n_words_per_block = args.n_words_per_block
    max_n_blocks = args.max_n_blocks
    n = args.n
    length = f"{n} sentences"

    all_n_blocks, all_summaries = [], []
    for i in tqdm(range(len(texts))):
        query = queries[i]
        if args.dataset == "middlesum":
            dataset_name = query
            dataset_names = ["Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews"]
            index = dataset_names.index(dataset_name)
            ns = [6, 7, 22, 5, 10]
            n = ns[index]
            length = f"{n} sentences"

        text = texts[i]
        if i in [221]: # data point not fitting in my GPU RAM :/
            text = " ".join(sent_tokenize(text)[:300])
        sents = sent_tokenize(text)
        sent_to_word_count = {}
        word_count = 0
        for j in range(len(sents)):
            n_words = len(word_tokenize(sents[j]))
            word_count += n_words
            sent_to_word_count[j] = word_count

        n_blocks = int(np.ceil(word_count / n_words_per_block))
        n_blocks = min(max_n_blocks, n_blocks)
        all_n_blocks.append(n_blocks)

        sent_start_thresh, sent_end_thresh = 0, 0
        block_summaries = []
        for j in range(n_blocks):
            block_end_thresh = (j + 1) * n_words_per_block
            while (sent_end_thresh < len(sents)) and (sent_to_word_count[sent_end_thresh] < block_end_thresh):
                sent_end_thresh += 1
            sents_block = sents[sent_start_thresh:(sent_end_thresh + 1)]
            text_block = " ".join(sents_block)
            text_block = " ".join(word_tokenize(text_block)[:(2*n_words_per_block)])

            # build the prompt
            if j == 0:
                instruction = f"Summarize the above text. The total summary should be around {length}."
                prompt = f"Read the following text and then summarize it:\n{text_block}\n{instruction}\nSummary:"
            else:
                instruction = f"Update the summary so far based on the above text. The total summary should be around {length}."
                summary_so_far = block_summaries[-1]
                prompt = f"Here is a summary of the text so far:\n{summary_so_far}.\nNow read the following text and then update the summary. You may rewrite the summary so far. Current text block:\n{text_block}\n{instruction}\nSummary:"

            # inference
            response = inference(prompt, tokenizer, model, args)
            block_summaries.append(response)
            sent_start_thresh = sent_end_thresh + 1

        # get the last summary
        response = block_summaries[-1]
        all_summaries.append(response)

        if args.print_output:
            print("*" * 10)
            for j in range(len(block_summaries)):
                print(f"Summary {j}")
                print(block_summaries[j])
            print("Overall summary:")
            print(" ".join(response.split()))
    print(f"Avg # blocks: {np.mean(np.array(all_n_blocks)):.2f}")

    return all_summaries

def inference(prompt, tokenizer, model, args):
    if args.model.startswith("gpt-3.5"):
        openai.api_key = args.openai_key
        received = False
        response = ""
        while not received:
            try:
                response = openai.ChatCompletion.create(
                    model=f"{args.model}",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=args.max_summary_length,
                    temperature=args.temperature,
                    # top_k=50,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                received = True
            except Exception as exc:
                print(exc)
                time.sleep(1.0)
                continue
        response = response["choices"][0]["message"]["content"]
        time.sleep(1.0)
    else:
        tok = tokenizer(prompt, return_tensors="pt")
        if args.decoding_method == "greedy":
            outputs = model.generate(
                input_ids = tok["input_ids"].cuda(),
                attention_mask = tok["attention_mask"].cuda(),
                do_sample = False,
                num_beams = 1,
                temperature = 0,
                min_new_tokens = 4,
                max_new_tokens = args.max_summary_length,
                pad_token_id = tokenizer.eos_token_id,
                return_dict_in_generate = True
            )
        elif args.decoding_method == "beam_search":
            outputs = model.generate(
                input_ids = tok["input_ids"].cuda(),
                attention_mask = tok["attention_mask"].cuda(),
                num_beams = 4,
                min_new_tokens = 4,
                max_new_tokens = args.max_summary_length,
                pad_token_id = tokenizer.eos_token_id,
                return_dict_in_generate = True
            )
        elif args.decoding_method == "top_k":
            outputs = model.generate(
                input_ids = tok["input_ids"].cuda(),
                attention_mask = tok["attention_mask"].cuda(),
                do_sample = True,
                top_k = args.top_k,
                temperature = args.temperature,
                min_new_tokens = 4,
                max_new_tokens = args.max_summary_length,
                pad_token_id = tokenizer.eos_token_id,
                return_dict_in_generate = True
            )
        elif args.decoding_method == "top_p":
            outputs = model.generate(
                input_ids = tok["input_ids"].cuda(),
                attention_mask = tok["attention_mask"].cuda(),
                do_sample = True,
                top_p = args.top_p,
                temperature = args.temperature,
                min_new_tokens = 4,
                max_new_tokens = args.max_summary_length,
                pad_token_id = tokenizer.eos_token_id,
                return_dict_in_generate = True
            )
        elif args.decoding_method == "temperature":
            outputs = model.generate(
                input_ids = tok["input_ids"].cuda(),
                attention_mask = tok["attention_mask"].cuda(),
                do_sample = True,
                temperature = args.temperature,
                min_new_tokens = 4,
                max_new_tokens = args.max_summary_length,
                pad_token_id = tokenizer.eos_token_id,
                return_dict_in_generate = True
            )

        tokens = outputs["sequences"]
        length = tok["input_ids"].shape[1]

        if args.model in ["flan-ul2"]:
            response = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        else:
            response = tokenizer.batch_decode(tokens[:, length:], skip_special_tokens=True)[0]
        del tok
        del tokens
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    return response
