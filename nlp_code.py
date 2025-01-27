import os
import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# Define project structure and directories
PROJECT_DIR = "nlp_pipeline_project"  # Main project folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")  # Folder for storing datasets
MODELS_DIR = os.path.join(PROJECT_DIR, "models")  # Folder for saving trained models

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Step 1: Generate synthetic dataset
def create_synthetic_dataset(file_path):
    data = {
        "text": [
            "The customer service was excellent, and the agent was very helpful.",
            "I need assistance with resetting my account password.",
            "The product quality was disappointing and below expectations.",
            "I am extremely satisfied with the delivery speed.",
            "The billing issue has not been resolved yet, causing frustration."
        ],
        "labels": [
            "positive,customer_support",
            "neutral,technical_support",
            "negative,product_quality",
            "positive,delivery",
            "negative,billing"
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# Create synthetic dataset
create_synthetic_dataset(os.path.join(DATA_DIR, "calls_dataset.csv"))

# Step 2: Preprocess dataset
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.lower()

dataset_path = os.path.join(DATA_DIR, "calls_dataset.csv")
data = pd.read_csv(dataset_path)
data["text"] = data["text"].apply(preprocess_text)

# Step 3: Multi-Label Classification setup
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"].tolist(), data["labels"].str.split(",").tolist(), test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
num_labels = len(set([label for sublist in data["labels"].str.split(",") for label in sublist]))
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Tokenize and encode text
def tokenize_texts(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_texts(train_texts)
test_encodings = tokenize_texts(test_texts)

# Map labels to integers
label_map = {label: idx for idx, label in enumerate(set([l for sublist in data["labels"].str.split(",") for l in sublist]))}
train_label_tensors = torch.zeros(len(train_labels), len(label_map))
test_label_tensors = torch.zeros(len(test_labels), len(label_map))

for i, labels in enumerate(train_labels):
    for label in labels:
        train_label_tensors[i][label_map[label]] = 1

for i, labels in enumerate(test_labels):
    for label in labels:
        test_label_tensors[i][label_map[label]] = 1

# Save label map
with open(os.path.join(MODELS_DIR, "label_map.json"), "w") as f:
    f.write(str(label_map))

# Step 4: Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(len(train_encodings["input_ids"])):
        inputs = {key: val[i].unsqueeze(0) for key, val in train_encodings.items()}
        labels = train_label_tensors[i].unsqueeze(0)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

# Save model
model.save_pretrained(MODELS_DIR)
tokenizer.save_pretrained(MODELS_DIR)

# Step 5: REST API with FastAPI
app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).detach().numpy()
        predictions = {label: float(prob) for label, prob in zip(label_map.keys(), probabilities[0])}
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
