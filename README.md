[README.md](https://github.com/user-attachments/files/29250798/README.md)
# ECG-Fatigue Detection System

> **AI-powered fatigue detection from ECG signals using deep learning and classical ML models, served via a FastAPI backend and React frontend.**

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Models](#models)
- [API Reference](#api-reference)
- [Notebooks](#notebooks)
- [Contributing](#contributing)

---

## Overview

ECG-Fatigue is a full-stack application that detects human fatigue states by analyzing electrocardiogram (ECG) signals. It combines multiple trained machine learning models — including a CNN, Random Forest, and XGBoost — to classify fatigue from raw ECG input. Results are exposed through a REST API and visualized in a React-based web interface.

---

## Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Backend   | Python, FastAPI, PyTorch          |
| ML Models | CNN (PyTorch), Random Forest, XGBoost |
| Frontend  | React (JSX/TSX), Vite, CSS        |
| Notebooks | Jupyter (model training & EDA)    |

---

## Project Structure

```
ECG_Fatigue/
├── Backend/
│   ├── api/
│   │   └── main.py              # FastAPI application entry point
│   ├── Data/                    # Raw and processed ECG datasets
│   ├── notebooks/               # Jupyter notebooks for EDA & model training
│   ├── results/                 # Model evaluation outputs
│   └── src/
│       ├── models/
│       │   ├── alarm_cnn.pt         # CNN model checkpoint
│       │   ├── alarm_cnn_final.pt   # Final trained CNN
│       │   ├── pretrained_cnn.pt    # Pretrained CNN weights
│       │   ├── rf_model.pkl         # Random Forest model
│       │   ├── xgboost_model.pkl    # XGBoost model
│       │   └── model_metadata.pkl   # Metadata for model loading
│       └── pipelines/               # Preprocessing & inference pipelines
├── Frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── App.css              # Application styles
│   │   └── main.tsx             # Vite entry point
│   ├── index.html
│   ├── package.json
│   └── eslint.config.js
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip
- npm or yarn

---

### Backend Setup

```bash
# 1. Navigate to the backend directory
cd Backend

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

### Frontend Setup

```bash
# 1. Navigate to the frontend directory
cd Frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

The React app will be available at `http://localhost:5173` (default Vite port).

---

## Models

Three models are available for fatigue classification:

| Model | File | Framework | Notes |
|-------|------|-----------|-------|
| CNN (Alarm) | `alarm_cnn_final.pt` | PyTorch | Primary deep learning model |
| Random Forest | `rf_model.pkl` | scikit-learn | Classical ML baseline |
| XGBoost | `xgboost_model.pkl` | XGBoost | Gradient boosting model |

Model metadata and preprocessing parameters are stored in `model_metadata.pkl` and loaded automatically at server startup.

---

## API Reference

Once the backend is running, full API documentation is available via:

- **Swagger UI** → `http://localhost:8000/docs`
- **ReDoc** → `http://localhost:8000/redoc`

Key endpoints (verify against `api/main.py` for exact routes):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Health check |
| `POST` | `/predict` | Submit ECG signal and receive fatigue prediction |

---

## Notebooks

Jupyter notebooks for data exploration and model training are located in `Backend/notebooks/`. To run them:

```bash
cd Backend
jupyter notebook
```

Results and evaluation metrics are saved to `Backend/results/`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is currently unlicensed. Please contact the repository owner before using it in other projects.

---

*Built by [adityasitapara](https://github.com/adityasitapara)*
