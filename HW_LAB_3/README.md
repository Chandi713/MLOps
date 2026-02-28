# HW_LAB_3: Polynomial Regression with MLOps Pipeline

A complete MLOps project demonstrating polynomial regression with FastAPI, automated testing, CI/CD pipelines, and Google Cloud Storage integration.

---

## 📁 Project Structure

```
HW_LAB_3/
├── .github/workflows/          # CI/CD Pipelines
│   ├── test.yml               # Run tests on push
│   ├── train_on_push.yml      # Train model on code changes
│   ├── scheduled_retrain.yml  # Retrain after train_on_push succeeds
│   └── ci_cd.yml              # Full pipeline (Test → Train → Docker → GCP)
│
├── src/                        # Source Code
│   ├── data.py                # Data generation (sine wave + noise)
│   ├── model.py               # Polynomial regression model class
│   ├── train.py               # Training logic
│   ├── predict.py             # Model evaluation metrics
│   ├── main.py                # FastAPI web application
│   ├── train_cli.py           # CLI for training with GCS support
│   └── gcs_utils.py           # Google Cloud Storage utilities
│
├── test/                       # Test Suites
│   ├── test_pytest.py         # Pytest tests (12 tests)
│   ├── test_unittest.py       # Unittest tests (8 tests)
│   └── test_mocks.py          # Mock tests for GCS functions
│
├── models/                     # Saved models (.pkl files)
├── metrics/                    # Evaluation metrics (JSON) - gitignored
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **FastAPI Web App** | Interactive polynomial regression predictions |
| **Automated Testing** | pytest and unittest test suites |
| **CI/CD Pipelines** | GitHub Actions for testing, training, deployment |
| **Model Versioning** | Timestamped model storage |
| **GCS Integration** | Upload models to Google Cloud Storage |
| **Docker Support** | Containerized deployment |
| **CLI Training** | Command-line interface for model training |

---

## 🚀 Quick Start

### Step 1: Initial Setup (One-time)

```bash
# Navigate to project
cd /Users/smitchandi/Documents/Coding/MLOps/MLOps/HW_LAB_3

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment (One-time)

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your values:
# - GCS_BUCKET_NAME=your-bucket-name
# - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# - GCP_PROJECT_ID=your-project-id
```

---

## 📋 Command Flow & Sequence

### Daily Development Workflow

```bash
# 1. Navigate & activate environment
cd /Users/smitchandi/Documents/Coding/MLOps/MLOps/HW_LAB_3
source venv/bin/activate

# 2. Run tests
pytest test/test_pytest.py -v

# 3. Train model (local only)
python src/train_cli.py --points 100 --degree 5 --evaluate --verbose

# 4. Train model + Upload to GCS
python src/train_cli.py --points 100 --degree 5 --evaluate --save-to-gcs --verbose

# 5. Run FastAPI server
uvicorn src.main:app --reload --port 8000
# Open: http://localhost:8000
```

### Docker Workflow

```bash
# Build image
docker build -t polynomial-regression .

# Run container (background)
docker run -d -p 8080:8080 --name poly-app polynomial-regression
# Open: http://localhost:8080

# Stop container
docker stop poly-app

# Remove container
docker rm poly-app
```

### Push to GitHub (Triggers CI/CD)

```bash
# Add changes
git add .

# Commit
git commit -m "Your commit message"

# Push to HW-LAB-3 branch
git push origin HW-LAB-3
```

---

## 🔧 CLI Training Options

```bash
python src/train_cli.py [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--points` | `-p` | 80 | Number of training data points |
| `--degree` | `-d` | 5 | Polynomial degree |
| `--evaluate` | `-e` | False | Run evaluation after training |
| `--save-to-gcs` | - | False | Upload model to GCS bucket |
| `--verbose` | `-v` | False | Show detailed output |
| `--timestamp` | `-t` | Auto | Custom timestamp for versioning |

### Examples

```bash
# Basic training
python src/train_cli.py -p 100 -d 5

# With evaluation
python src/train_cli.py -p 100 -d 5 --evaluate

# Full: evaluate + GCS + verbose
python src/train_cli.py -p 100 -d 5 -e --save-to-gcs -v
```

---

## 🔄 CI/CD Workflow Triggers

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PUSH TO HW-LAB-3 BRANCH                      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
   │  test.yml   │    │train_on_push │    │  ci_cd.yml  │
   │ (Run Tests) │    │  (if src/    │    │(Full Pipeline)
   │             │    │   changed)   │    │             │
   └─────────────┘    └──────┬───────┘    └─────────────┘
                             │
                             ▼ (if successful)
                    ┌────────────────────┐
                    │ scheduled_retrain  │
                    │ (degrees 2, 5, 9)  │
                    └────────────────────┘
```

### Trigger Summary

| Workflow | Trigger | Condition |
|----------|---------|-----------|
| `test.yml` | Push to `HW-LAB-3` | Always |
| `train_on_push.yml` | Push to `HW-LAB-3` | Only if `src/` or `requirements.txt` changed |
| `scheduled_retrain.yml` | After `train_on_push.yml` | Only if train succeeds |
| `ci_cd.yml` | Push to `HW-LAB-3` | Always (Test → Train → GCS → Docker → GCP) |

---

## ☁️ Google Cloud Storage Setup

### Prerequisites

1. GCP Project created
2. Cloud Storage bucket created
3. Service account with **Storage Admin** role
4. Service account JSON key downloaded

### Local Setup

```bash
# 1. Place your service account JSON in project folder
# 2. Update .env file:
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=your-project-id
```

### GitHub Secrets (for CI/CD)

Add these secrets in GitHub → Settings → Secrets → Actions:

| Secret Name | Value |
|-------------|-------|
| `GCP_SA_KEY` | Contents of service account JSON file |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCS_BUCKET_NAME` | Your GCS bucket name |

---

## 🧪 Testing

### Run All Tests

```bash
# Pytest (12 tests)
pytest test/test_pytest.py -v

# Unittest (8 tests)
python -m unittest test.test_unittest -v

# All tests
pytest test/ -v
```

### Test Coverage

| Test File | Tests | Modules Covered |
|-----------|-------|-----------------|
| `test_pytest.py` | 12 | data, model, train, predict, FastAPI |
| `test_unittest.py` | 8 | data, model, train, predict |
| `test_mocks.py` | 6 | GCS utility functions (mocked) |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with input form |
| `/predict` | POST | Train model and return predictions |

### Using the Web Interface

1. Start server: `uvicorn src.main:app --reload --port 8000`
2. Open: http://localhost:8000
3. Enter number of points and polynomial degree
4. Click "Predict" to see results

---

## 📊 Output Files

### Models (Local)
```
models/
├── polynomial_regression_degree_2.pkl
├── polynomial_regression_degree_5.pkl
└── polynomial_regression_degree_9.pkl
```

### Models (GCS)
```
gs://your-bucket/
├── trained_models/
│   ├── polynomial_v1_degree5_20240228120000.pkl
│   └── polynomial_v2_degree5_20240228130000.pkl
└── model_version.txt
```

### Metrics
```
metrics/
└── 20240228120000_degree5_metrics.json
```

Example metrics JSON:
```json
{
    "timestamp": "20240228120000",
    "model_config": {
        "degree": 5,
        "training_points": 100
    },
    "training": {
        "mse": 0.0234,
        "r2": 0.9812
    },
    "testing": {
        "mse": 0.0312,
        "r2": 0.9756
    }
}
```

---

## 🐳 Docker

### Build & Run

```bash
# Build
docker build -t polynomial-regression .

# Run (foreground)
docker run -p 8080:8080 polynomial-regression

# Run (background)
docker run -d -p 8080:8080 --name poly-app polynomial-regression

# View logs
docker logs poly-app

# Stop & remove
docker stop poly-app && docker rm poly-app
```

---

## 📝 Quick Reference

### All Commands at a Glance

```bash
# Setup
cd /Users/smitchandi/Documents/Coding/MLOps/MLOps/HW_LAB_3
source venv/bin/activate
pip install -r requirements.txt

# Test
pytest test/test_pytest.py -v

# Train (local)
python src/train_cli.py -p 100 -d 5 -e -v

# Train (with GCS)
python src/train_cli.py -p 100 -d 5 -e --save-to-gcs -v

# FastAPI server
uvicorn src.main:app --reload --port 8000

# Docker
docker build -t polynomial-regression .
docker run -d -p 8080:8080 --name poly-app polynomial-regression

# Git push (triggers CI/CD)
git add . && git commit -m "message" && git push origin HW-LAB-3
```

---

## Acknowledgments

- Northeastern University MLOps Course (IE-7374)
- GitHub Actions for CI/CD
- Google Cloud Platform for cloud storage

---
