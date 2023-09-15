from flask import render_template
from flask_cors import CORS
import json
from flask import Flask, request, jsonify
import torch
from transformers import (BertTokenizer, BertForSequenceClassification, BartForConditionalGeneration, 
                          BertForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer, BartTokenizer)


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

MODEL_DIR = "models/model"
TOKENIZER_DIR = "models/tokenizer"

# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# For summarization
summarization_model_name = "facebook/bart-large-cnn"
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

# using BERT for QA
qa_model_bert_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer_bert = BertTokenizer.from_pretrained(qa_model_bert_name)
qa_model_bert = BertForQuestionAnswering.from_pretrained(qa_model_bert_name)

# additional QA model (roberta)
qa_model_roberta_name = "consciousAI/question-answering-roberta-base-s-v2"
qa_model_roberta = AutoModelForQuestionAnswering.from_pretrained(qa_model_roberta_name)
qa_tokenizer_roberta = AutoTokenizer.from_pretrained(qa_model_roberta_name)


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_target_names = config["train_target_names"]

# ====== Classification

def classify_text(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move tensors to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model's prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted class
    predicted_class_idx = torch.argmax(outputs.logits, 1).item()

    # Convert the predicted class index into its corresponding label
    predicted_class_label = train_target_names[predicted_class_idx]

    return predicted_class_label

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    # Get the text from the request
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Use the function to classify text
    predicted_label = classify_text(text, model, tokenizer, device)

    return jsonify({"predicted_label": predicted_label})


# ====== Summarization

def summarize_text(text, model, tokenizer, device):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarize_text(text, summarization_model, summarization_tokenizer, device)
    return jsonify({"summary": summary})

# ====== Question Answering

def answer_question(question, context, model, tokenizer, device):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer

@app.route("/answer", methods=["POST"])
def answer_endpoint():
    data = request.get_json(force=True)
    question = data.get("question", "")
    context = data.get("context", "")
    model_choice = data.get("model_choice", "").lower()

    if not question or not context:
        return jsonify({"error": "Question or context not provided"}), 400

    if model_choice == "roberta":
        answer = answer_question(question, context, qa_model_roberta, qa_tokenizer_roberta, device)
    elif model_choice == "bert":
        answer = answer_question(question, context, qa_model_bert, qa_tokenizer_bert, device)
    else:
        return jsonify({"error": "Invalid model choice. Please choose either 'roberta' or 'bert'."}), 400

    return jsonify({"answer": answer})




if __name__ == "__main__":
    app.run(debug=True)
