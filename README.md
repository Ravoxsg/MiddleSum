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

## Experiments

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

Alternatively, you can download the summaries with [this link](https://drive.google.com/file/d/1-k_EFVVg1ynMiNTS0tebsW0ACbXELUNh/view?usp=sharing). 

To reproduce the analysis in RQ1 (about mapping bigrams in generated summaries to the source): 
```bash
python src/rq1_alignment_bigrams.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ2 (about mapping sentences in generated summaries to the visible source): 
```bash
python src/rq2_alignment_sentences.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ3 (about checking the correlation between the mean position of salient info and the source): 
```bash
python src/rq3_mean_salient_position.py --dataset <dataset_name> --subset <subset_name> --clean_model_name <llm_name> --metric <metric_name>
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
