# Ghana-QA: 10 Million Ghanaian Question-Answer Pairs

## Project Overview

**Ghana-QA** is a project aimed at generating a massive dataset of high-quality, groundable Question-Answer (QA) pairs from a corpus of Ghanaian news and research articles.

This dataset is created to train and improve Question Answering models specific to the Ghanaian context and will be made publicly available on the Ghana NLP community page on Hugging Face.

The script uses the **Nvidia Build API** (specifically the `mistral-medium-instruct` model) to analyze raw article content and generate 10 distinct, questions and short responses to them.

We aim to create the largest, highest-quality Ghanaian QA dataset that can be translated to several Ghanaian languages. Our target is 10,000,000.

### Volunteers Needed

To reach this ambitious target, we need volunteers to help process the raw articles. We are seeking at least **50 dedicated volunteers** willing to run the code for **20+ hours** each.

## Project Progress
We have now reached 400,000 QA pairs thanks to the following awesome contributors:

| Name | QA pairs Contributed |
|------|------------|
| [Mich-Seth Owusu](https://www.linkedin.com/in/mich-seth-owusu/) | 989,200 |
| [Priscilla (Naadu) Lartey](https://www.linkedin.com/in/larteypriscilla/) | 115,560 |
| [Gerhardt Datsomor](https://www.linkedin.com/in/gerhardt-datsomor/) | 100,000 |


## How to contribute

### 1. Requirements

- **Python 3.8+**
- **Nvidia Build API Key:** Required for generating content. Get one from [NVIDIA NIM](https://build.nvidia.com/).
- **Input Dataset:** The input dataset will be provided by Ghana NLP.

### 2. Request Data

Please email michsethowusu@gmail.com and state how many hours you can dedicate to running the code. We will provide the corresponding input dataset.

## Usage Guide

Run these commands in your terminal:

1. **Clone the Repository:**

   ```
   git clone https://github.com/GhanaNLP/Ghana-QA.git
   cd Ghana-QA
   ```

2. **Install Dependencies:**

   ```
   pip install pandas python-dotenv requests
   ```

3. **Run the Script:**

   ```
   python3 generateQA.py
   ```

The script will prompt you to enter your Nvidia API key and also to select the input dataset.

## Why Contribute?

Your work directly contributes to making information more accessible for people in Ghana. Additionally, contributors are added to our special group, giving you:

- First access to community opportunities.
- Access to our curated local language datasets on our [Hugging Face repo](https://huggingface.co/ghananlpcommunity/datasets/).
