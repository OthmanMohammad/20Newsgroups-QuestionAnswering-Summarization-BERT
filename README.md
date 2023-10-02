# 20 Newsgroups NLP Analysis with BERT, BART, and RoBERTa

This repository offers a Flask web application designed for in-depth analysis of the 20 Newsgroups dataset. Utilizing state-of-the-art models like BERT, BART, and RoBERTa, users can classify articles, obtain summaries, and get answers to their questions.

## Features:

- Article Classification with BERT: Efficiently categorize articles into predefined topics.
- Document Summarization with BART: Generate concise and coherent summaries of lengthy articles.
- Question Answering with RoBERTa: Extract specific information from articles by posing questions. Two RoBERTa variants are available for this task:
  - [consciousAI/question-answering-roberta-base-s-v2](https://huggingface.co/consciousAI/question-answering-roberta-base-s-v2)
  - [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)

## Technical Details:

- Flask Framework: The application is built on Flask, offering a lightweight web server to interact with the models.
- BERT for Classification: BERT's bidirectional context capturing capabilities are used for classifying articles.
- BART for Summarization: BART, a sequence-to-sequence model, is employed to condense articles into shorter summaries.

## Getting Started:

### Dataset:

You can download the 20 Newsgroups dataset from [this link](https://huggingface.co/datasets/MohammadOthman/20-News-Groups).

### Prerequisites:

- Python 3.x
- Flask
- PyTorch
- Transformers library

### Installation:

Clone the repository:

```bash
git clone https://github.com/OthmanMohammad/20Newsgroups-QuestionAnswering-Summarization-BERT.git
```

Navigate to the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Application:

Start the Flask server:

```bash
python app.py
```
