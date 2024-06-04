## Getting Started

Once you clone the repo, create a dedicated conda environment with Python 3.8: 
```bash
cd MiddleSum/
conda create -name middlesum python=3.8.17
```

Next activate the environment:
```bash
conda activate middlesum
```

Then install all the dependencies:
```bash
pip install -r requirements.txt
```

Do not forget to change the values in the src/keys.py file. 
You need to enter the path to your home working directory, your HuggingFace token and your OpenAI key (if you want to use GPT-3.5).

## Experiments

First, you need to generate the summaries. 2 consumer grade (24-48GB) GPUs will be enough:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/llm_inference.py --dataset <dataset_name> --clean_model_name <llm_name> 
```

Then, you need to score the generated summaries:
```bash
python src/main.py --dataset <dataset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ1 (about mapping bigrams in generated summaries to the source): 
```bash
python src/rq1_alignment_bigrams.py --dataset <dataset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ2 (about mapping sentences in generated summaries to the visible source): 
```bash
python src/rq2_alignment_sentences.py --dataset <dataset_name> --clean_model_name <llm_name> 
```

To reproduce the analysis in RQ3 (about checking the correlation between the mean position of salient info and the source): 
```bash
python src/rq3_alignment_sentences.py --dataset <dataset_name> --clean_model_name <llm_name> 
```

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@misc{ravaut2024context,
  title={On Context Utilization in Summarization with Large Language Models}, 
  author={Mathieu Ravaut and Aixin Sun and Nancy F. Chen and Shafiq Joty},
  year={2024},
  eprint={2310.10570},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
