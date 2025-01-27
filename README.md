# Multi-Label Text Classification with FastAPI and BERT

## Overview
This project demonstrates the development of a multi-label text classification system using BERT. The model is trained on a synthetic dataset and deployed as a REST API using FastAPI, allowing real-time predictions of multiple categories for input text.

## Features
- **Synthetic Dataset Creation**: Generates a labeled dataset for customer feedback analysis.
- **Data Preprocessing**: Cleans and standardizes text while encoding labels for multi-label classification.
- **Model Training**: Fine-tunes a BERT model for multi-label classification tasks.
- **REST API Deployment**: Provides a `/predict` endpoint for real-time predictions using FastAPI.

## Project Structure
```
multi-label-text-classification/
│
├── data/            # Contains the synthetic dataset
├── models/          # Stores the trained BERT model and label map
├── app.py           # FastAPI application for serving the model
├── requirements.txt # Dependencies for the project
└── README.md        # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- GPU (optional but recommended for training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-label-text-classification.git
   cd multi-label-text-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `bert-base-uncased` from [Hugging Face](https://huggingface.co/bert-base-uncased).

### Running the Project

1. **Synthetic Dataset Creation**:
   The dataset is automatically created in the `data` directory when the script runs.

2. **Model Training**:
   Run the training script to fine-tune BERT:
   ```bash
   python nlp_code.py
   ```

3. **API Deployment**:
   Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   Access the API at `http://127.0.0.1:8000`.

### Example Usage
Send a POST request to the `/predict/` endpoint with text input:
```json
{
  "text": "The customer service was excellent and very helpful."
}
```
Response:
```json
{
  "predictions": {
    "positive": 0.87,
    "customer_support": 0.92,
    "negative": 0.12,
    "product_quality": 0.15
  }
}
```

## Future Enhancements
- Use real-world datasets to improve robustness.
- Incorporate advanced techniques like transfer learning and data augmentation.
- Add Docker support for containerized deployment.

## License
This project is licensed under the MIT License.
