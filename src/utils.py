import random
import os
import numpy as np
import torch
import pickle

from parameters import *


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def seed_everything(args):
    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def settle_args(args):
    # Dataset
    index = all_datasets.index(args.dataset)

    args.dataset_name = dataset_names[index]
    args.dataset_proper_name = dataset_proper_names[index]
    args.dataset_version = dataset_versions[index]
    args.summarization_input = summarization_inputs[index]
    args.text_key = text_keys[index]
    args.summary_key = summary_keys[index]
    args.max_summary_length = max_summary_lengths[index]
    if args.subset == "val":
        args.subset_size = val_sizes[index]
        args.subset_key = validation_keys[index]
    if args.subset == "test":
        args.subset_size = test_sizes[index]
        args.subset_key = test_keys[index]
    args.max_size = min(args.max_size, args.subset_size)
    if args.clean_model_name.startswith("gpt"):
        args.max_size = min(args.max_size, 300)
    args.n = ns[index]

    # LLM
    index = clean_model_names.index(args.clean_model_name)

    args.model = models[index]
    args.model_name = model_names[index]
    args.context_length = context_lengths[index] - args.max_summary_length - 64 # -64 to account for the instruction

def get_clean_control_name(args):
    control = f"{args.control}"
    if args.control_label == "query":
        control += "_w_query"
    if args.control == "shuffle":
        control += f"_{args.n_shuffle_perm}"
    if args.control == "position":
        if args.swap_docs == True:
            control += f"_{args.control_doc_pos}"
        else:
            control += f"_just_relevant"
    if args.control == "filling":
        if not args.swap_docs:
            control = "just_first_and_last"

    return control

def get_clean_model_name(args, control=False):
    model_name = f"{args.clean_model_name}_{args.instruction_position}"
    if args.oracle_n_sents:
        model_name = f"{args.clean_model_name}_oracle_n_sents"
    if args.oracle_n_words:
        model_name = f"{args.clean_model_name}_oracle_n_words"
    if args.enforced_max_length > 0:
        model_name = f"{args.clean_model_name}_length_{args.enforced_max_length}"

    if args.focus_prompt:
        model_name += "_focus_prompt"
    if args.inference_method == "pyramidal":
        model_name += "_pyramidal"
    if args.inference_method == "incremental":
        model_name += "_incremental"
    if control:
        control = get_clean_control_name(args)
        model_name += f"_{control}"

    return model_name

def load_data(args):
    texts_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_texts_{args.max_size}.pkl"
    if args.use_control:
        control = get_clean_control_name(args)
        texts_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_texts_{control}_{args.max_size}.pkl"
    print(f"\nLoading texts from {texts_path}")
    texts = pickle.load(open(texts_path, "rb"))
    
    if args.dataset == "middlesum":
        queries_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_queries_{args.max_size}.pkl"
        if os.path.isfile(queries_path):
            queries = pickle.load(open(queries_path, "rb"))
        else:
            queries = [""] * len(texts)
        texts = (texts, queries)
    
    labels_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_labels_{args.max_size}.pkl"
    if args.use_control:
        control = get_clean_control_name(args)
        labels_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_labels_{control}_{args.max_size}.pkl"
    print(f"\nLoading labels from {labels_path}")
    labels = pickle.load(open(labels_path, "rb"))

    return texts, labels

def load_pred(args, control=False):
    model_name = get_clean_model_name(args, control=control)
    summaries_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_summaries_{model_name}_{args.decoding_method}_{args.max_size}.pkl"
    print(f"\nLoading summaries from {summaries_path}")
    summaries = pickle.load(open(summaries_path, "rb"))

    return summaries

def save_pred(summaries, args, control=False):
    size = len(summaries)
    model_name = get_clean_model_name(args, control=control)

    summaries_path = f"summaries/{args.dataset_name}/{args.subset}/{args.subset}_summaries_{model_name}_{args.decoding_method}_{size}.pkl"
    os.makedirs(f"summaries/{args.dataset_name}/{args.subset}", exist_ok=True)
    pickle.dump(summaries, open(summaries_path, "wb"))
    print(f"\nSaved summaries to: {summaries_path}")
