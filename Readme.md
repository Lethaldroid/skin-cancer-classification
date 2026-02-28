# 🏥 Melanoma Cancer Classification

A deep learning application for classifying skin lesions as benign or malignant melanoma using AlexNet. The project features a Streamlit frontend UI and a FastAPI backend service for real-time image inference.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)

## ✨ Features
- **Deep Learning Classification**: AlexNet-based neural network for accurate melanoma detection
- **Web Interface**: User-friendly Streamlit app for uploading and analyzing skin images
- **REST API**: FastAPI backend service for model inference
- **GPU Support**: Automatic CUDA detection for faster inference
- **Confidence Scores**: Returns prediction confidence along with classification
- **Pre-trained Models**: Ready-to-use model weights included

## 📁 Project Structure
```
├── frontend/              # Streamlit web application
│   └── app.py            # Frontend interface for image upload and analysis
├── model-service/        # FastAPI backend service
│   ├── app.py           # API endpoints for inference
│   ├── model_utils.py   # Model utilities and preprocessing
│   └── training/
│       └── train.ipynb  # Model training notebook
├── model/               # Pre-trained models
│   ├── melanoma_CNN.pt
│   └── model_weights.pth
├── data/                # Dataset directory
│   └── melanoma_cancer_dataset/
│       ├── train/       # Training images (benign/malignant)
│       └── test/        # Test images (benign/malignant)
├── requirements.txt     # Python dependencies
└── Readme.md           # This file
```

## 🛠️ Prerequisites
- Python 3.7+
- pip or conda
- GPU (optional, for faster inference)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "AI Project"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   Ensure the model weights are available in the `model/` directory:
   - `model_weights.pth` (required)
   - `melanoma_CNN.pt` (optional)
   The model weights can be downloaded from [here](https://drive.google.com/drive/folders/1nSNnwzWRPFUiRWJo_QdX83uE7ORC9RDw?usp=sharing)
## 🚀 Usage

### Running the Application

1. **Start the FastAPI Model Service** (from `model-service` directory)
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the Streamlit Frontend** (from `frontend` directory, in another terminal)
   ```bash
   streamlit run app.py
   ```
   The web interface will open at `http://localhost:8501`

3. **Using the Application**
   - Upload a skin image (JPG/PNG format)
   - Click "🚀 Analyze" to get predictions
   - View the classification result (benign or malignant) with confidence score

### API Endpoints

**POST `/predict`** - Predict classification for uploaded image
- **Request**: Multipart form with image file
- **Response**: JSON with prediction, confidence, and available classes
  ```json
  {
    "prediction": "benign",
    "confidence": 0.95,
    "classes": ["benign", "malignant"]
  }
  ```

**GET `/`** - Health check
- **Response**: API status

## 🧠 Model Details
- **Architecture**: AlexNet (Convolutional Neural Network)
- **Input**: 224×224 RGB images
- **Output**: Binary classification (Benign / Malignant)
- **Framework**: PyTorch
- **Training**: See `model-service/training/train.ipynb` for training details

## 📊 Dataset
The melanoma cancer dataset includes:
- **Training Set**: Labeled images split into benign and malignant categories
- **Test Set**: Validation images with same structure
- **Location**: `data/melanoma_cancer_dataset/`

To use custom data:
1. Organize images into `train/benign/`, `train/malignant/`, `test/benign/`, `test/malignant/`
2. Retrain the model using the notebook or adapt the training script

## 📝 Requirements
- fastapi
- uvicorn
- torch
- torchvision
- pillow
- python-multipart
- streamlit
- requests 

