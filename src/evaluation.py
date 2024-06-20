import numpy as np
import pickle
import torch
import os
import time
import openai
import tiktoken
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from autoacu import A2CU, A3CU
from summac.model_summac import SummaCConv

from utils import get_clean_model_name
from engine import prepare_texts_gptlikert


def compute_scores(texts, queries, summaries, labels, args, control=False):
    model_name = get_clean_model_name(args, control=control)
    folder = f"scores/{args.dataset_name}/{args.subset}"
    os.makedirs(folder, exist_ok=True)
    size = len(texts)
    metric = args.metric
    if args.metric.startswith("gpt-likert") and not(args.openai_model.endswith("instruct")):
        metric = args.metric + "_" + args.openai_model.split("-")[-1]
    scores_path = f"{folder}/{args.subset}_{metric}_{model_name}_{args.decoding_method}_{size}.pkl"
    print(f"Scores path: {scores_path}")

    if args.process_empty_summaries:
        n_processed = 0
        for i in range(len(summaries)):
            if len(summaries[i].split()) == 0:
                summaries[i] = "none"
                n_processed += 1
        print(f"Processed {n_processed} empty summaries")

    if args.compute_scores:
        # general stuff: length and abstractiveness
        if args.metric == "sents":
            scores = get_sents_scores(summaries)
        elif args.metric == "words":
            scores = get_words_scores(summaries)
        elif args.metric == "abstractiveness":
            scores = get_abstractiveness_scores(texts, summaries)
        # reference-based metrics
        elif args.metric == "rouge-1":
            scores = get_rouge1_scores(summaries, labels)
        elif args.metric == "rouge-2":
            scores = get_rouge2_scores(summaries, labels)
        elif args.metric == "rouge-l":
            scores = get_rougel_scores(summaries, labels)
        elif args.metric == "mean-rouge":
            scores = get_meanrouge_scores(summaries, labels)
        elif args.metric == "bertscore":
            scores = get_bertscore_scoes(summaries, labels)
        elif args.metric == "a2cu":
            scores = get_a2cu_scores(summaries, labels)
        elif args.metric == "a3cu":
            scores = get_a3cu_scores(summaries, labels)
        # reference-free metrics
        elif args.metric == "summac":
            scores = get_summac_scores(texts, summaries)
        elif args.metric.startswith("gpt-likert"):
            scores = get_gptlikert_scores(texts, queries, summaries, args)

        print(scores.shape)
        if args.save_scores and not args.random_baseline:
            pickle.dump(scores, open(scores_path, "wb"))
            print(f"saved the scores to: {scores_path}")
    else:
        scores = pickle.load(open(scores_path, "rb"))
        print(f"loaded the scores from: {scores_path}")

    print(f"\nMean {args.metric}: {np.mean(scores):.4f}")

    return scores


def complete_evaluation(summaries, labels, args):
    n_words, n_sents = [], []
    for i in range(len(summaries)):
        n_words.append(len(word_tokenize(summaries[i])))
        n_sents.append(len(sent_tokenize(summaries[i])))
    n_words, n_sents = np.array(n_words), np.array(n_sents)
    print(f"# Words: {np.mean(n_words):.2f}, # sents: {np.mean(n_sents):.2f}")

    summaries_to_use, labels_to_use = [], []
    for i in range(len(summaries)):
        labels_i = "\n".join(sent_tokenize(labels[i]))
        labels_to_use.append(labels_i)
        summaries_i = "\n".join(sent_tokenize(summaries[i]))
        summaries_to_use.append(summaries_i)

    if "rouge-1" in args.metrics:
        r1s = get_rouge1_scores(summaries, labels)
        r2s = get_rouge2_scores(summaries, labels)
        rls = get_rougel_scores(summaries, labels)
        meanrs = get_meanrouge_scores(summaries, labels)
    if "bertscore" in args.metrics:
        bs = get_bertscore_scoes(summaries, labels)

    scores = []
    if "rouge-1" in args.metrics:
        scores.append(np.expand_dims(r1s, 1))
    if "rouge-2" in args.metrics:
        scores.append(np.expand_dims(r2s, 1))
    if "rouge-l" in args.metrics:
        scores.append(np.expand_dims(rls, 1))
    if "mean-rouge" in args.metrics:
        scores.append(np.expand_dims(meanrs, 1))
    if "bertscore" in args.metrics:
        scores.append(np.expand_dims(bs, 1))
    scores = np.concatenate(scores, 1)
    print(scores.shape)

    return scores

def get_sents_scores(summaries):
    scores = []
    for i in tqdm(range(len(summaries))):
        n_sents = len(sent_tokenize(summaries[i]))
        scores.append(n_sents)
    scores = np.array(scores)

    return scores

def get_words_scores(summaries):
    scores = []
    for i in tqdm(range(len(summaries))):
        n_sents = len(word_tokenize(summaries[i]))
        scores.append(n_sents)
    scores = np.array(scores)

    return scores

def get_abstractiveness_scores(texts, summaries):
    scores = []
    for i in tqdm(range(len(texts))):
        text = texts[i].lower()
        text_words = word_tokenize(text)
        summary = summaries[i].lower()
        summary_words = word_tokenize(summary)
        text_bigrams = {}
        for j in range(len(text_words)-1):
            bigram = text_words[j] + "___" + text_words[j+1]
            text_bigrams[bigram] = 0
        count = 0
        for j in range(len(summary_words)-1):
            bigram = summary_words[j] + "___" + summary_words[j+1]
            if bigram in text_bigrams.keys():
                count += 1
        count = (100 * count) / max(1, len(summary_words)-1)
        scores.append(count)
    scores = np.array(scores)

    return scores

def get_rouge1_scores(summaries, labels):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    for i in tqdm(range(len(summaries))):
        label = labels[i]
        summary = summaries[i]
        label = "\n".join(sent_tokenize(label))
        summary = "\n".join(sent_tokenize(summary))
        rouge_scores = scorer.score(label, summary)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        scores.append(r1)
    scores = np.array(scores)

    return scores

def get_rouge2_scores(summaries, labels):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    for i in tqdm(range(len(summaries))):
        label = labels[i]
        summary = summaries[i]
        label = "\n".join(sent_tokenize(label))
        summary = "\n".join(sent_tokenize(summary))
        rouge_scores = scorer.score(label, summary)
        r1 = 100 * rouge_scores["rouge2"].fmeasure
        scores.append(r1)
    scores = np.array(scores)

    return scores

def get_rougel_scores(summaries, labels):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    for i in tqdm(range(len(summaries))):
        label = labels[i]
        summary = summaries[i]
        label = "\n".join(sent_tokenize(label))
        summary = "\n".join(sent_tokenize(summary))
        rouge_scores = scorer.score(label, summary)
        r1 = 100 * rouge_scores["rougeLsum"].fmeasure
        scores.append(r1)
    scores = np.array(scores)

    return scores

def get_meanrouge_scores(summaries, labels):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    for i in tqdm(range(len(summaries))):
        label = labels[i]
        summary = summaries[i]
        label = "\n".join(sent_tokenize(label))
        summary = "\n".join(sent_tokenize(summary))
        rouge_scores = scorer.score(label, summary)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        meanr = (r1 + r2 + rl) / 3
        scores.append(meanr)
    scores = np.array(scores)

    return scores

def get_bertscore_scoes(summaries, labels):
    p, r, f1 = bertscore(summaries, labels, lang='en', verbose=True)
    scores = 100 * f1.detach().cpu().numpy()

    return scores

def get_a2cu_scores(summaries, labels):
    a2cu = A2CU(device=0)  # the GPU device to use
    all_recall_scores, all_prec_scores, all_f1_scores = [], [], []
    for i in tqdm(range(len(labels))):
        recall_scores, prec_scores, f1_scores = a2cu.score(
            references=[labels[i]],
            candidates=[summaries[i]],
            generation_batch_size=2,  # the batch size for ACU generation
            matching_batch_size=16,  # the batch size for ACU matching
            output_path = None,  # the path to save the evaluation results,
            recall_only = False,  # whether to only compute the recall score
            acu_path = None  # the path to save the generated ACUs
        )
        all_recall_scores += recall_scores
        all_prec_scores += prec_scores
        all_f1_scores += f1_scores
    recall_scores = 100 * np.array(all_recall_scores)
    prec_scores = 100 * np.array(all_prec_scores)
    f1_scores = 100 * np.array(all_f1_scores)
    scores = f1_scores

    return scores

def get_a3cu_scores(summaries, labels):
    a3cu = A3CU(device=0)
    all_recall_scores, all_prec_scores, all_f1_scores = [], [], []
    for i in tqdm(range(len(labels))):
        recall_scores, prec_scores, f1_scores = a3cu.score(
            references=[labels[i]],
            candidates=[summaries[i]],
            batch_size=16,  # the batch size for ACU generation
            output_path = None  # the path to save the evaluation results
        )
        all_recall_scores += recall_scores
        all_prec_scores += prec_scores
        all_f1_scores += f1_scores
    recall_scores = 100 * np.array(all_recall_scores)
    prec_scores = 100 * np.array(all_prec_scores)
    f1_scores = 100 * np.array(all_f1_scores)
    scores = f1_scores

    return scores

def get_summac_scores(texts, summaries):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda",
                            start_file="default", agg="mean")
    summac_scores = []
    for i in tqdm(range(len(summaries))):
        text = texts[i]
        summary = summaries[i]
        score_conv1 = model_conv.score([text], [summary])
        score = 100 * score_conv1["scores"][0]
        summac_scores.append(score)
    scores = np.array(summac_scores)

    return scores

def get_gptlikert_scores(texts, queries, summaries, args):
    openai.api_key = args.openai_key
    gpt_model = args.openai_model
    encoding = tiktoken.encoding_for_model(gpt_model)

    if gpt_model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]:
        args.gpt_model_max_length = 16384 #16384
    elif gpt_model in ["gpt-3.5-turbo-instruct"]:
        args.gpt_model_max_length = 4096
    elif "gpt-4" in gpt_model:
        args.gpt_model_max_length = 128000
    print(f"GPT evaluator max length: {args.gpt_model_max_length}")

    general_instruction = "Score the following summary generated by another system given the source on a scale from 1 to 5 \
    with regards to overall general summary quality. 1-point indicates a low-quality summary, and 5 points a very high quality summary. \
    A high-quality summary is grammatical, fluent, informative, relevant, coherent and factually consistent with the source."
    informativeness_instruction = "Score the following summary generated by another system given the source on a scale from 1 to 5 \
    with regards to how informative the summary is. 1-point indicates a not informative summary, and 5 points a very informative summary. \
    An informative summary captures the important information in the article and presents it accurately and concisely."
    quality_instruction = "Score the following summary generated by another system given the source on a scale from 1 to 5 \
    with regards to its quality. 1-point indicates a low-quality summary, and 5 points a very high-quality summary. \
    A high quality summary is comprehensible and understandable."
    coherence_instruction = "Score the following summary generated by another system given the source on a scale from 1 to 5 \
    with regards to its coherence. 1-point indicates an incoherent summary, and 5 points a very coherent summary. \
    A coherent summary is well-structured and well-organized."
    attributable_instruction = "Score the following summary generated by another system given the source on a scale from 1 to 5 \
    with regards to how attributable it is. 1-point indicates a not very attributable summary, with many hallucinations, \
    and 5 points a summary very attributable to the source, consistent with the source. In a very attributable summary, all the information is fully attributable to the source."

    instruction = general_instruction
    if "informativeness" in args.metric:
        instruction = informativeness_instruction
    elif "quality" in args.metric:
        instruction = quality_instruction
    elif "coherence" in args.metric:
        instruction = coherence_instruction
    elif "attributable" in args.metric:
        instruction = attributable_instruction

    n_shrink_summary = 0
    gptlikerts = []
    trunc_texts = prepare_texts_gptlikert(encoding, texts, queries, args)
    for i in tqdm(range(len(texts))):
        source = trunc_texts[i]
        query = queries[i]
        summary = summaries[i]
        if (len(encoding.encode(instruction)) + len(encoding.encode(source)) + len(encoding.encode(summary, disallowed_special=(encoding.special_tokens_set - {'<|endoftext|>'})))) > (args.gpt_model_max_length - 96):
            ids = encoding.encode(summary)
            max_summary_length = args.gpt_model_max_length - 96 - len(encoding.encode(instruction)) - len(encoding.encode(source))
            ids = ids[:max_summary_length]
            print("new length", max_summary_length)
            summary = encoding.decode(ids)
            n_shrink_summary += 1
        prompt = f"{instruction} Let's think step-by-step and just output the score."
        prompt += f"\nSource:\n{source}"
        prompt += f"\nInstruction:\nSummarize the above text in {args.n} sentences."
        prompt += f"\nSummary:\n{summary}"
        prompt += f"\nYour score:"
        if "quality" in args.metric or "coherence" in args.metric:
            prompt = f"{instruction} Let's think step-by-step and just output the score."
            prompt += f"\nSummary:\n{summary}"
            prompt += f"\nYour score:"
        #print("*"*50)
        #print(prompt)
        received = False
        response = ""
        while not received:
            try:
                if gpt_model in ["gpt-3.5-turbo-instruct"]:
                    response = openai.Completion.create(
                        model=f"{gpt_model}",
                        prompt=prompt,
                        max_tokens=8,
                        temperature=0.0,
                        top_p=1.0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=f"{gpt_model}",
                        messages=[
                           {"role": "user", "content": prompt},
                        ],
                        max_tokens=8,
                        temperature=0.0,
                        top_p=1.0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                    )
                received = True
            except Exception as exc:
                print(exc)
                if str(exc).startswith("Sorry! We've encountered"):
                    print("A")
                    received = True
                else:    
                    time.sleep(1.0)
                    continue
        time.sleep(1.0)
        if response == "":
            likert = -1
        else:
            if gpt_model in ["gpt-3.5-turbo-instruct"]:
                parsed = response["choices"][0]["text"]
            else:
                parsed = response["choices"][0]["message"]["content"]
            try:
                likert = float(parsed.split()[0])
            except Exception:
                likert = -1

        gptlikerts.append(likert)
    scores = np.array(gptlikerts)
    unique_scores = np.unique(scores)
    unique_scores = np.sort(unique_scores)
    print(f"Reduced the summary size in {n_shrink_summary} cases")
    for s in unique_scores:
        print(s, np.sum(scores == s), 100 * np.sum(scores == s) / len(scores))

    return scores
