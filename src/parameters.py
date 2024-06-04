### Datasets

all_datasets = ["cnndm", "xsum", "reddit", "samsum", "arxiv", "pubmed", "govreport", "summscreen", "multinews", "multixscience", "middlesum"]

dataset_names = ["CNNDM", "XSum", "Reddit", "SAMSum", "Arxiv", "PubMed", "GovReport", "SummScreen", "MultiNews", "MultiXScience", "MiddleSum"]
dataset_proper_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "samsum", "ccdv/arxiv-summarization", "ccdv/pubmed-summarization", "ccdv/govreport-summarization", "_", "multi_news", "_", "_"]
dataset_versions = ["3.0.0", "default", "long", "samsum", "document", "document", "", "_", "", "_", "_"]
summarization_types = ["generic", "generic", "generic", "generic", "generic", "generic", "generic", "generic", "generic", "generic", "query"]
summarization_inputs = ["single", "single", "single", "single", "single", "single", "single", "single", "multi", "multi", "single"]
text_keys = ["article", "document", "documents", "dialogue", "article", "article", "report", "_", "document", "_", "_"]
summary_keys = ["highlights", "summary", "tldr", "summary", "abstract", "abstract", "summary", "_", "summary", "_", "_"]
validation_keys = ["validation", "validation", "", "validation", "validation", "validation", "validation", "_", "validation", "_", "_"]
test_keys = ["test", "test", "", "test", "test", "test", "test", "_", "test", "_", "_"]
max_summary_lengths = [192, 64, 128, 128, 512, 512, 768, 512, 512, 384, -1]
val_sizes = [13368, 11332, 4214, 818, 6436, 6633, 973, 338, 5622, 5066, -1]
test_sizes = [11490, 11334, 4222, 819, 6440, 6658, 973, 337, 5622, 5093, 225]
ns = [3, 1, 2, 2, 6, 7, 22, 5, 10, 5, -1]

### LLMs

clean_model_names = [
    "llama_2_7b_base", "llama_2_13b_base",
    "flan_ul2", "llama_2_7b", "llama_2_13b", "xgen_7b", "mistral_7b",
    "vicuna_7b_16k", "llama_2_7b_32k",
    "gpt-3.5-turbo-0125"
]

models = [
    "llama-2-7b", "llama-2-13b",
    "flan-ul2", "llama-2-7b-chat", "llama-2-13b-chat", "xgen-7b", "mistral-7b",
    "vicuna-1.5-7b-16k", "llama-2-7b-32k",
    "gpt-3.5-turbo-0125"
]
model_names = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
    "google/flan-ul2", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "Salesforce/xgen-7b-8k-inst", "mistralai/Mistral-7B-Instruct-v0.1",
    "lmsys/vicuna-7b-v1.5-16k", "togethercomputer/Llama-2-7B-32K-Instruct",
    "gpt-3.5-turbo-0125"]
context_lengths = [
    4096, 4096,
    2048, 4096, 4096, 8192, 8192,
    16384, 32768,
    16384
]
