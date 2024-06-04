# Script to run stats on datasets (e.g., length, abstractiveness)

import pickle
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer

from keys import hf_token


def main():
    root = "/data/mathieu"
    subset = "test"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = "llama-2-7b"

    dataset_names = ["CNNDM", "XSum", "Reddit", "SAMSum", "Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews", "MultiXScience", "MiddleSum"]
    val_sizes = [13368, 11332, 4214, 818, 6436, 6633, 973, 338, 5622, 5066, -1]
    test_sizes = [11490, 11334, 4222, 819, 6440, 6658, 973, 337, 5622, 5093, 225]

    for i in range(len(dataset_names)):
        dataset = dataset_names[i]
        print("\n" + "*"*50 + f" {dataset}")
        if subset == "val":
            size = val_sizes[i]
        else:
            size = test_sizes[i]

        texts_path = f"summaries/{dataset}/{subset}/{subset}_texts_{size}.pkl"
        texts = pickle.load(open(texts_path, "rb"))
        print(f"loaded texts from: {texts_path}")
        labels_path = f"summaries/{dataset}/{subset}/{subset}_labels_{size}.pkl"
        labels = pickle.load(open(labels_path, "rb"))
        print(len(texts), len(labels))
    
        # # docs
        n_docs = np.mean(np.array([len(x.split("|||||")) for x in texts]))
        print(f"# docs: {n_docs:.2f}")

        # Length
        print(len(texts[0]))
        # sentences
        n_sents_texts = np.array([len(sent_tokenize(x)) for x in tqdm(texts)])
        n_sents_labels = np.array([len(sent_tokenize(x)) for x in tqdm(labels)])
        print(f"Source # sents: {np.mean(n_sents_texts):.2f}, labels # sents: {np.mean(n_sents_labels):.2f}")
        # words
        n_words_texts = np.array([len(word_tokenize(x)) for x in tqdm(texts)])
        n_words_labels = np.array([len(word_tokenize(x)) for x in tqdm(labels)])
        print(f"Source # words: {np.mean(n_words_texts):.2f}, labels # words: {np.mean(n_words_labels):.2f}")
        # tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token, cache_dir = f"{root}/hf_models/{model}", trust_remote_code = True)
        n_tokens_texts = np.array([len(tokenizer(x)["input_ids"]) for x in tqdm(texts)])
        n_tokens_labels = np.array([len(tokenizer(x)["input_ids"]) for x in tqdm(labels)])
        print(f"Source # tokens: {np.mean(n_tokens_texts):.2f}, labels # tokens: {np.mean(n_tokens_labels):.2f}")

        if dataset == "MiddleSum":
            queries_path = f"summaries/{dataset}/{subset}/{subset}_queries_{size}.pkl"
            queries = pickle.load(open(queries_path, "rb"))
            queries = np.array(queries)
            subdatasets = np.unique(queries)
            print(subdatasets)
            for subdataset in subdatasets:
                idx = np.arange(len(queries))[queries == subdataset]
                print(f"\nSubset: {subdataset}, size: {len(idx)}")
                texts_subset = [texts[x] for x in idx]
                labels_subset = [labels[x] for x in idx]

                # # docs
                n_docs = np.mean(np.array([len(x.split("|||||")) for x in texts_subset]))
                print(f"# docs: {n_docs:.2f}")

                # sentences
                n_sents_texts = np.array([len(sent_tokenize(x)) for x in tqdm(texts_subset)])
                n_sents_labels = np.array([len(sent_tokenize(x)) for x in tqdm(labels_subset)])
                print(f"Source # sents: {np.mean(n_sents_texts):.2f}, labels # sents: {np.mean(n_sents_labels):.2f}")
                # words
                n_words_texts = np.array([len(word_tokenize(x)) for x in tqdm(texts_subset)])
                n_words_labels = np.array([len(word_tokenize(x)) for x in tqdm(labels_subset)])
                print(f"Source # words: {np.mean(n_words_texts):.2f}, labels # words: {np.mean(n_words_labels):.2f}")
                # tokens
                n_tokens_texts = np.array([len(tokenizer(x)["input_ids"]) for x in tqdm(texts_subset)])
                n_tokens_labels = np.array([len(tokenizer(x)["input_ids"]) for x in tqdm(labels_subset)])
                print(f"Source # tokens: {np.mean(n_tokens_texts):.2f}, labels # tokens: {np.mean(n_tokens_labels):.2f}")

        # Abstractiveness
        all_new_unigrams, all_new_bigrams, all_new_trigrams = [], [], []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            label = labels[i]
            text_words = word_tokenize(text)
            label_words = word_tokenize(label)
            text_unigrams = text_words
            text_bigrams = [[text_words[j], text_words[j+1]] for j in range(len(text_words)-1)]
            text_trigrams = [[text_words[j], text_words[j+1], text_words[j+2]] for j in range(len(text_words)-2)]
            label_unigrams = label_words
            label_bigrams = [[label_words[j], label_words[j+1]] for j in range(len(label_words)-1)]
            label_trigrams = [[label_words[j], label_words[j+1], label_words[j+2]] for j in range(len(label_words)-2)]
            new_unigrams = 100 * len([x for x in label_unigrams if not(x in text_unigrams)]) / max(1, len(label_unigrams))
            new_bigrams = 100 * len([x for x in label_bigrams if not(x in text_bigrams)]) / max(1, len(label_bigrams))
            new_trigrams = 100 * len([x for x in label_trigrams if not(x in text_trigrams)]) / max(1, len(label_trigrams))
            all_new_unigrams.append(new_unigrams)
            all_new_bigrams.append(new_bigrams)
            all_new_trigrams.append(new_trigrams)
        new_unigrams = np.mean(np.array(all_new_unigrams))
        new_bigrams = np.mean(np.array(all_new_bigrams))
        new_trigrams = np.mean(np.array(all_new_trigrams))
        print(f"New unigrams: {new_unigrams:.2f}%, new bigrams: {new_bigrams:.2f}%, new trigrams: {new_trigrams:.2f}%")


if __name__ == '__main__':
    main()
