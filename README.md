# LLM Lab

This repository documents experiments in training Large Language Models (LLMs) using various model architectures. MLflow is used for experiment tracking and metrics logging.  

## Goals

- Experiment with different LLM architectures and training procedures
- Track experiments, metrics, and artifacts using MLflow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jeremyng-dev/llm-lab
   cd llm-lab
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Usage

- Start the MLflow server
```bash
mlflow server --port 5000
```
- Run experiments in `experiments.ipynb`
- View metrics and parameters in the MLflow WebUI

## 1. Dataset
For the pretraining phase, the raw variant of the [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) dataset is used. The smaller version is used to enable faster iteration when experimenting with different training procedures.

### Dataset Exploration
After loading the train dataset, we first convert the [Huggingface Dataset](https://github.com/huggingface/datasets) into a Pandas DataFrame, allowing us to have a quick glance at the data.  
The training split contains 36,718 entries, of which 21,714 are unique. The most frequent entry is an empty string.  

![Raw dataset statistics](/assets/images/Unfiltered%20Dataset.png)  

As empty strings do not provide meaningful information during pretraining, we remove them from the dataset.  
After filtering, we are left with 23,767 total entries.  

![Filtered dataset statistics](/assets/images/Filtered%20Dataset.png)  

Filtering duplicated entries reveals that most repetitions correspond to article headers.
As these provide important structure and context for the text that follows, these duplicates are intentionally retained.  

![Duplicated Rows](/assets/images/Duplicated%20Rows.png)  

### Dataset creation
Since the dataset is relatively small, all rows are concatenated using line breaks to form a single string.
We then tokenize this string and create batches of training data.  

**Example**
```text
Input text:   = Valkyria Chronicles III = 

 Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァ 
```
```text
Target text:   Valkyria Chronicles III = 

 Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァル
```

The target sequence is simply the input sequence shifted right by one token, which corresponds to a next-token prediction objective.

```python
Input Tokens:  tensor([  796,   569, 18354,  7496, 17740,  6711,   796,   220,   628,  2311,   73, 13090,   645,   569, 18354,  7496]) 

Label Tokens:  tensor([  569, 18354,  7496, 17740,  6711,   796,   220,   628,  2311,    73,   13090,   645,   569, 18354,  7496,   513])
```

With the dataset prepared, it can now be used for pretraining.
```text
Dataset statistics:
Train tokens: 2391884, Validation tokens: 247289, Test tokens: 283287
```