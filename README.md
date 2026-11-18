# PathSnap
### Intelligent Map Matching and Route Reconstruction using Machine Learning

PathSnap is a machine-learning-based map-matching system that reconstructs vehicle routes by matching GPS observations to road-network segments.

The system combines machine learning, candidate generation, and sequential route decoding to produce accurate route predictions.

## 1. System Architecture

The project consists of the following major components:

- **Data Processing:** GPS trajectory processing and candidate road-segment generation.
- **Feature Engineering:** Extraction of spatial, directional and temporal features.
- **Machine Learning:** Random Forest classification for candidate road-segment prediction.
- **Route Reconstruction:** HMM and Viterbi decoding for sequential route reconstruction.
- **Backend:** FastAPI-based prediction and evaluation API.
- **Frontend:** React-based dashboard for trip selection, prediction and visualization.

## 2. Technologies Used

| Component | Technology |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Backend | FastAPI |
| Frontend | React, Vite |
| Route Decoder | HMM, Viterbi |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib, Git LFS |
| Visualization | Matplotlib and project map visualization |

## 3. Prerequisites

Install the following before setting up the project:

- Python 3.10 or compatible project-supported version
- Git
- Git LFS
- Node.js and npm
- Windows PowerShell

Check installations:

```powershell
python --version
git --version
git lfs version
node --version
npm --version
```

## 4. Clone the Repository

Clone the repository:

```powershell
git clone https://github.com/crosswinrobert/PathSnap.git
```

Navigate into the project:

```powershell
cd PathSnap
```

Initialize Git LFS and download model files:

```powershell
git lfs install
git lfs pull
```

Verify that the Random Forest model exists:

```powershell
Get-Item artifacts\models\member2_random_forest.joblib
```

The model should be downloaded as an actual binary file, not a small Git LFS pointer.

## 5. Backend Installation

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

Install the Python dependencies listed in the repository's backend requirements file.

For a root-level requirements file:

```powershell
pip install -r requirements.txt
```

If dependencies are maintained in a different file, use that file instead.

### Step 4: Start the FastAPI server

From the project root:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend will be available at:

- API: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## 6. Frontend Installation

Open another PowerShell terminal.

Navigate to the frontend directory:

```powershell
cd PathSnap\frontend
```

If already inside the project root:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm ci
```

Start the development server:

```powershell
npm run dev
```

The frontend will normally be available at:

http://localhost:5173

## 7. Dataset Partition Configuration

The backend supports configurable dataset partitions using the environment variable:

```powershell
$env:PATHSNAP_PARTITION = "student_test"
```

The default partition is `student_test`.

To switch partitions, use:

```powershell
$env:PATHSNAP_PARTITION = "evaluator_test"
```

Restart the backend after changing this variable.

**Note:** The selected partition must exist in the project's dataset. The availability of `evaluator_test` must be verified before using it.

## 8. Machine Learning and Route Reconstruction

The integrated prediction pipeline follows these stages:

1. Load GPS trajectory observations.
2. Generate candidate road segments.
3. Extract the 25 input features.
4. Load the trained Random Forest classifier.
5. Generate candidate probabilities using `predict_proba()`.
6. Pass probabilities into the HMM/Viterbi route reconstruction pipeline.
7. Generate the predicted route.
8. Return prediction results and evaluation metrics through FastAPI.

### Model Information

| Property | Value |
|---|---|
| Model | Random Forest Classifier |
| Input features | 25 |
| Classes | 0 and 1 |
| Number of estimators | 100 |
| Model file | `artifacts/models/member2_random_forest.joblib` |

## 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root |
| GET | `/health` | Backend health check |
| GET | `/trips/` | Retrieve available trips |
| POST | `/predict/{trip_id}` | Predict and reconstruct a route |

### Example prediction request

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/00000000"
```

Interactive API documentation:

http://127.0.0.1:8000/docs

## 10. Two-Computer Deployment

For a demonstration using two computers on the same local network:

**Computer 1 — Backend**
- Runs FastAPI.
- Hosts the machine learning model.
- Processes predictions and route reconstruction.
- Serves generated route maps.

**Computer 2 — Frontend**
- Runs the React application.
- Sends requests to the backend computer.
- Displays predictions, metrics and route visualizations.

Both computers must be connected to the same network.

Configure the frontend API URL to point to the backend computer's local IPv4 address and allow the frontend origin in FastAPI CORS settings.

Backend startup:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 11. Troubleshooting

**Model file not found**

```powershell
git lfs pull
```

**Port 8000 already in use**

Check the process occupying the port:

```powershell
netstat -ano | findstr :8000
```

**Frontend dependency errors**

```powershell
npm ci
```

**Backend import errors**

Ensure the virtual environment is activated and dependencies are installed.

**Frontend cannot connect to backend**

Check:
- Both computers are on the same network.
- The backend is running.
- The frontend API URL uses the backend computer's IPv4 address.
- FastAPI CORS permits the frontend origin.

## 12. Repository

GitHub: https://github.com/crosswinrobert/PathSnap

---

**PathSnap — Intelligent GPS Map Matching and Route Reconstruction**
