## Getting Started

Once you clone the repo, create a dedicated conda environment with Python 3.8: 
```bash
cd MiddleSum/
conda create --name middlesum python=3.8.17
```

Next activate the environment:
```bash
conda activate middlesum
```

Then install all the dependencies:
```bash
pip install -r requirements.txt
```

**Do not forget to change the values in the *src/keys.py* file**. 
You need to enter the path to your home **working directory**, your **HuggingFace token** and your **OpenAI key** (if you want to use GPT-3.5).  

Next, we need to do some small extra data preparation for 2 datasets: Multi-XScience and SummScreen.  

For **Multi-XScience**, run: 
```bash
python src/prepare_data/prepare_multixscience.py
```

For **SummScreen**, first download the dataset here: https://github.com/mingdachen/SummScreen.  
Then place it in src/prepare_data/summscreen/.  
Next, run:

```bash
python src/prepare_data/prepare_summscreen.py
```

## Experiments

### Inference

First, you need to generate the summaries. 2 consumer grade (24-48GB) GPUs will be enough:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> 
```
This will save summaries under *summaries/dataset/subset/*.

Then, you need to score the generated summaries:
```bash
python src/main.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> --metric <metric_name>
```
This will save scores under *scores/dataset/subset/*.

Alternatively, you can download the summaries with [this link](https://drive.google.com/file/d/1jfzcMg1EJBNZ3VlTBbxM-TPc40OS6N4j/view?usp=sharing). 

### Research Questions

To reproduce the analysis in RQ1, about mapping bigrams in generated summaries to the source: 
```bash
python src/rq1_alignment_bigrams.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ2, about mapping sentences in generated summaries to the visible source: 
```bash
python src/rq2_alignment_sentences.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ3, about checking the correlation between the mean position of salient info and the source: 
```bash
python src/rq3_mean_salient_position.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> --metric <metric_name>
```

### Analysis

To run the control experiment of Figure 3, for instance on Multi-XScience placing the relevant document at position 0: 
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_control_inference.py --dataset multixscience --subset test --control_n_docs True --n_docs 7 --control position --control_doc_pos 0 --swap_docs True --clean_model_name <llm_name>
```

To run the control experiment of Table 3 with only the first and last documents, for instance on Multi-News: 
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_control_inference.py --dataset multinews --subset test --control_n_docs True --n_docs 5 --control filling --swap_docs False --clean_model_name <llm_name>
```
For the same setup but including 3 random documents between the first and last ones: 
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_control_inference.py --dataset multinews --subset test --control_n_docs True --n_docs 5 --control filling --swap_docs True --clean_model_name <llm_name>
```

To run inference on MiddleSum with the focus prompt:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference.py --dataset middlesum --subset test --clean_model_name <llm_name> --focus_prompt True
```
with hierarchical inference:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference.py --dataset middlesum --subset test --clean_model_name <llm_name> --inference_method pyramidal
```
with incremental inference:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference.py --dataset middlesum --subset test --clean_model_name <llm_name> --inference_method incremental
```


## Citation

If you find any of this useful, please kindly consider citing our paper in your publication.

```
@article{ravaut2023context,
  title={On Context Utilization in Summarization with Large Language Models},
  author={Ravaut, Mathieu and Joty, Shafiq and Sun, Aixin and Chen, Nancy F},
  journal={arXiv e-prints},
  pages={arXiv--2310},
  year={2023}
}
```
