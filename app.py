import json
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Paths based on your project structure
MODEL_DIR = "models/model"
TOKENIZER_DIR = "models/tokenizer"

# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_target_names = config["train_target_names"]

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
    predicted_class_label = train_target_names[predicted_class_idx]  # Make sure to define train_target_names

    return predicted_class_label

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    # Get the text from the request
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Use the function to classify text
    predicted_label = classify_text(text, model, tokenizer, device)

    return jsonify({"predicted_label": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
