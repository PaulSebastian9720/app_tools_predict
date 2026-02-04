# Tools Detection — Full-Stack ML Pipeline

End-to-end machine learning system for **detecting hand tools** in images using
YOLOv8 object detection. Includes a training pipeline in Jupyter notebooks, a
FastAPI inference backend, and a Next.js interactive frontend.

**Detected classes**: Hammer, Pliers, Screwdriver, Wrench

![Results](attachments/val_batch0_labels.jpg)
---

## System Architecture

```mermaid
flowchart TB
    subgraph USER["User"]
        BROWSER["Browser"]
    end

    subgraph FRONTEND["Frontend — Next.js 16"]
        DASH["Dashboard"]
        PRED["Predict Page"]
        TRAIN["Training Page"]
    end

    subgraph BACKEND["Backend — FastAPI"]
        API["REST API\n/api/v1"]
        STORE["ModelStore\nInference Engine"]
        WORKER["Training Worker\n(subprocess)"]
    end

    subgraph DATA["Data Layer"]
        DB[("SQLite\ndb.sqlite3")]
        MLFLOW["MLflow\nTracking Server"]
        FS["File Storage\nmodels / datasets"]
    end

    subgraph NOTEBOOKS["Notebooks — Jupyter"]
        NB["Training &\nEvaluation\nPipeline"]
    end

    BROWSER --> FRONTEND
    FRONTEND -->|HTTP REST| API
    API --> STORE
    API --> DB
    API --> WORKER
    WORKER --> MLFLOW
    WORKER --> FS
    STORE --> FS
    NB --> MLFLOW
    MLFLOW --> FS

    style FRONTEND fill:#3b82f6,color:#fff
    style BACKEND fill:#10b981,color:#fff
    style DATA fill:#f59e0b,color:#000
    style NOTEBOOKS fill:#8b5cf6,color:#fff
```

---

## Notebook Pipeline

The training pipeline is organized as five sequential notebooks:

```mermaid
flowchart LR
    N1["<b>1. EDA</b>\nExploratory\nData Analysis"]
    N2["<b>2. Transforms</b>\nData Splitting\n& Preprocessing"]
    N3["<b>3. Transfer Learning</b>\nBootstrap Training\nYOLOv8-Nano"]
    N4["<b>4. Evaluation</b>\nModel Testing\n& Prediction"]
    N5["<b>5. Fine-Tuning</b>\nIncremental\nRetraining"]

    N1 --> N2 --> N3 --> N4
    N3 --> N5 --> N4

    style N1 fill:#3b82f6,color:#fff
    style N2 fill:#8b5cf6,color:#fff
    style N3 fill:#f59e0b,color:#000
    style N4 fill:#10b981,color:#fff
    style N5 fill:#ef4444,color:#fff
```

| # | Notebook | What It Does |
|---|----------|--------------|
| 1 | **EDA** | Analyze class distribution, image dimensions, annotation statistics |
| 2 | **Transforms** | Split raw Roboflow data into bootstrap / dataset_1 / dataset_2 / test subsets |
| 3 | **Transfer Learning** | Train YOLOv8-Nano on the small bootstrap dataset (199 images). Register model in MLflow |
| 4 | **Evaluation** | Load any model from MLflow Model Registry. Evaluate on the fixed 893-image test set. Single and batch prediction |
| 5 | **Fine-Tuning** | Load a base model from the registry, retrain on new data, compare before vs. after metrics |

### Model Registry Flow

```mermaid
flowchart TD
    TRAIN3["Notebook 3\nBootstrap Training"] -->|register v1| REG["MLflow\nModel Registry\ntools_detection_yolo"]
    REG -->|load latest| EVAL4["Notebook 4\nEvaluation"]
    REG -->|load base| FT5["Notebook 5\nFine-Tuning"]
    FT5 -->|register v2, v3...| REG
    REG -->|shared store| API["API Backend"]
```

---

## Application Architecture

```mermaid
flowchart LR
    subgraph FE["Frontend :3000"]
        direction TB
        FE1["Dashboard"]
        FE2["Predict\nSingle / Batch"]
        FE3["Training\nDatasets / Jobs / Models"]
    end

    subgraph BE["Backend :8000"]
        direction TB
        BE1["/health"]
        BE2["/predict/upload\n/predict/batch"]
        BE3["/datasets\n/train\n/jobs\n/models"]
    end

    subgraph DB["Database"]
        direction TB
        DB1[("SQLite\ndatasets\ntraining_jobs\nmodels")]
    end

    subgraph ML["ML Layer"]
        direction TB
        ML1["YOLO\nInference"]
        ML2["Training\nWorker"]
        ML3["MLflow\nTracking"]
    end

    FE1 --> BE1
    FE2 --> BE2
    FE3 --> BE3
    BE2 --> ML1
    BE3 --> DB1
    BE3 --> ML2
    ML2 --> ML3
    ML1 --> DB1

    style FE fill:#3b82f6,color:#fff
    style BE fill:#10b981,color:#fff
    style DB fill:#f59e0b,color:#000
    style ML fill:#8b5cf6,color:#fff
```

---

## Technology Stack

### Notebooks / ML Pipeline

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white) | Programming language |
| ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white) | Object detection model |
| ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-EE4C2C?logo=pytorch&logoColor=white) | Deep learning framework |
| ![MLflow](https://img.shields.io/badge/MLflow-3.9-0194E2?logo=mlflow&logoColor=white) | Experiment tracking & model registry |
| ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) | Interactive development |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv&logoColor=white) | Image processing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data analysis |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white) | Visualization |

### Backend API

| Technology | Purpose |
|------------|---------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-1.0-009688?logo=fastapi&logoColor=white) | REST API framework |
| ![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white) | Lightweight database |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848?logo=gunicorn&logoColor=white) | ASGI server |

### Frontend

| Technology | Purpose |
|------------|---------|
| ![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white) | React framework |
| ![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black) | UI library |
| ![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white) | Type safety |
| ![Tailwind](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white) | Styling |

---

## Datasets

Data sourced from [Roboflow](https://roboflow.com/) — licensed under
**CC BY 4.0**.

| Dataset | Format | Images | Classes | Link |
|---------|--------|--------|---------|------|
| **Tools Detection** | YOLOv8 bbox | 5,204 | 11 | [paul-space-qcfcl/tools-bynck-wpynu](https://universe.roboflow.com/paul-space-qcfcl/tools-bynck-wpynu) |
| **Tools Segmentation** | YOLOv8 segmentation | 3,097 | 4 | [paul-space-qcfcl/tools-segmentation-f2nhg-tf2cd](https://universe.roboflow.com/paul-space-qcfcl/tools-segmentation-f2nhg-tf2cd) |

### Preprocessed Splits

After running Notebook 2, the detection dataset is split into:

| Split | Images | Purpose |
|-------|--------|---------|
| `bootstrap` | 199 | Initial transfer learning (Notebook 3) |
| `dataset_1` | 800 | First fine-tuning round (Notebook 5) |
| `dataset_2` | 1,666 | Second fine-tuning round (Notebook 5) |
| `test` | 893 | Fixed evaluation set (never used for training) |

---

## Project Structure

```
tools-detection/
├── README.md                   # This file
├── launch_mlflow_ui.sh         # Start MLflow UI/tracking server
│
├── notebooks/                  # ML training pipeline
│   ├── 1_EDA_tools.ipynb
│   ├── 2_Transforms.ipynb
│   ├── 3_Model_trasfer_learning.ipynb
│   ├── 4_Evaluation_model.ipynb
│   ├── 5_Incremental_fine_tuning.ipynb
│   ├── utils/                  # Shared Python modules
│   ├── data/                   # Roboflow datasets
│   ├── mlruns/                 # MLflow tracking store
│   ├── runs/                   # YOLO training outputs
│   └── requirements.txt
│
├── api_tols/                   # FastAPI backend
│   ├── app/
│   │   ├── main.py             # Entry point
│   │   ├── routers/            # API endpoints
│   │   ├── models_store.py     # YOLO inference engine
│   │   ├── database.py         # SQLite ORM
│   │   └── training_worker.py  # Background training
│   ├── storage/                # Runtime data (DB, models, datasets)
│   └── requirements.txt
│
└── fronted_tols/               # Next.js frontend
    ├── app/                    # Pages (Dashboard, Predict, Training)
    ├── components/             # Reusable UI components
    ├── lib/                    # API client & types
    └── package.json
```

---

## Quick Start

### 1. MLflow Tracking Server

```bash
./launch_mlflow_ui.sh
# Opens at http://127.0.0.1:5000
```

![Register Epoch-maP50 Mlflow UI](attachments/mAP50.png)



### 2. Notebooks (Training)

```bash
cd notebooks
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run notebooks 1 through 5 in order. Models are automatically registered in
MLflow after training.

### 3. API Backend

```bash
cd api_tols
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### 4. Frontend

```bash
cd fronted_tols
npm install
npm run dev
```

Opens at `http://localhost:3000`.

![Fronted UI](attachments/fronted_UI.png)
![Fronted UI](attachments/fronted_UI2.png)



---

## MLflow

All components share a single MLflow tracking store at `notebooks/mlruns/`.
The `launch_mlflow_ui.sh` script starts the MLflow server that both notebooks
and the API connect to.

```mermaid
flowchart LR
    NB["Notebooks"] -->|train & register| MLF["MLflow Server\n:5000"]
    API["API Backend"] -->|read models| MLF
    MLF -->|file store| FS["notebooks/mlruns/"]

    style MLF fill:#8b5cf6,color:#fff
```

Models are registered under `tools_detection_yolo` in the Model Registry.
Each training or fine-tuning run creates a new version that can be loaded by
version number or as "latest".

---

## Training Results

| Model | Dataset | mAP50 | mAP50-95 | Precision | Recall | F1 |
|-------|---------|-------|----------|-----------|--------|-----|
| Bootstrap (v1) | bootstrap (199 imgs) | 0.485 | 0.319 | 0.532 | 0.457 | 0.492 |
| Fine-tune (v2) | dataset_1 (800 imgs) | 0.845 | 0.654 | 0.836 | 0.768 | 0.801 |
| Fine-tune (v3) | dataset_2 (1,666 imgs) | 0.848 | 0.644 | 0.855 | 0.751 | 0.799 |

All models evaluated on the same fixed test set (893 images).

