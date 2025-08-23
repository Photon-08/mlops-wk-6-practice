# Heart Disease Prediction MLOps Pipeline

A complete MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. This project demonstrates end-to-end machine learning operations including model training, API serving, containerization, Kubernetes deployment, and continuous integration/deployment with monitoring and explainability.

## 🎯 Project Overview

This project implements a heart disease prediction system using machine learning with a focus on MLOps best practices. The system predicts whether a patient has heart disease based on 13 clinical features, providing both binary predictions and probability scores.

### Key Features

- ⚕️ **Heart Disease Prediction**: Binary classification using logistic regression
- 🚀 **FastAPI REST API**: Production-ready API with automatic documentation
- 🐳 **Containerized Deployment**: Docker container for consistent environments
- ☸️ **Kubernetes Ready**: K8s manifests for scalable deployment
- 🔄 **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions
- 📊 **Model Monitoring**: Data drift detection with Evidently
- 🔍 **Model Explainability**: SHAP analysis for interpretable predictions
- 📈 **Cloud Deployment**: Google Cloud Platform (GKE) integration

## 📊 Dataset Information

**Source**: UCI Machine Learning Repository - Heart Disease Dataset (ID: 45)
- **Instances**: 303 patients
- **Features**: 13 clinical attributes
- **Target**: Binary classification (heart disease presence/absence)

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Gender (1 = male, 0 = female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina (1 = yes) | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment (0-2) | Categorical |
| ca | Number of major vessels colored by fluoroscopy (0-3) | Numeric |
| thal | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Model Training │───▶│   Model Serving │
│  (UCI ML Repo)  │    │    (train.py)   │    │   (serve.py)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│  Containerization│◀───│   Kubernetes   │◀──────────┘
│   (Dockerfile)  │    │  (k8s/*.yaml)   │
└─────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌─────────────────┐
│   CI/CD Pipeline│───▶│  Cloud Deploy   │
│ (GitHub Actions)│    │      (GKE)      │
└─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │  Explainability │
│  (Evidently)    │    │     (SHAP)      │
└─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerization)
- kubectl (optional, for Kubernetes deployment)
- Google Cloud SDK (optional, for GCP deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Photon-08/mlops-wk-6-practice.git
   cd mlops-wk-6-practice
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train.py
   ```
   This will:
   - Download the UCI Heart Disease dataset (requires internet connection)
   - Preprocess the data (handle missing values, binarize target)
   - Train a logistic regression model
   - Save the model as `model.joblib`
   
   **Note:** If you encounter network issues, the repository already includes a pre-trained `model.joblib` file.

4. **Start the API server**
   ```bash
   uvicorn serve:app --host 0.0.0.0 --port 8000
   ```

5. **Test the API**
   - Open your browser to `http://localhost:8000` for the welcome message
   - Visit `http://localhost:8000/docs` for interactive API documentation
   - Make predictions using the `/predict` endpoint

## 🔌 API Usage

### Endpoints

#### `GET /`
Returns a welcome message.

**Response:**
```json
{
  "message": "Heart Disease Prediction API"
}
```

#### `POST /predict`
Predicts heart disease based on input features.

**Request Body:**
```json
{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
```

**Response:**
```json
{
  "prediction": "heart disease",
  "probability": [0.0028355672621042105, 0.9971644327378958]
}
```

**Note:** The probability array shows [probability_no_disease, probability_disease].

### Example Usage

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}'
```

**Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

## 🐳 Docker Usage

### Build the Docker image
```bash
docker build -t heart-disease-api .
```

### Run the container
```bash
docker run -p 8000:8000 heart-disease-api
```

The API will be available at `http://localhost:8000`.

## ☸️ Kubernetes Deployment

### Local Kubernetes (minikube/kind)

1. **Apply the manifests**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

2. **Get the service URL**
   ```bash
   kubectl get services
   minikube service uci-heart-health-practice --url  # for minikube
   ```

### Google Kubernetes Engine (GKE)

The repository includes a complete CI/CD pipeline for GKE deployment using GitHub Actions.

**Required Secrets:**
- `GCP_SA_KEY`: Service account JSON key for Google Cloud
- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GKE_CLUSTER_NAME`: Name of your GKE cluster
- `GCP_REGION`: Google Cloud region
- `CML_PAT`: GitHub Personal Access Token for CML reports

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cml-k8s-deploy.yml`) automatically:

1. **Builds and pushes** Docker image to Google Artifact Registry
2. **Deploys** to Google Kubernetes Engine
3. **Tests** the deployed API endpoint
4. **Creates** CML report with deployment status and sample predictions

**Trigger:** Push to `main` branch or manual workflow dispatch

## 📊 Model Monitoring & Explainability

### Data Drift Monitoring
- **File**: `data_drift_report.html`
- **Tool**: Evidently
- **Purpose**: Detect changes in data distribution over time

### Model Explainability
- **File**: `force_plot.html` 
- **Tool**: SHAP (SHapley Additive exPlanations)
- **Purpose**: Understand feature importance and model predictions
- **Notebook**: `exploration_explainable_ai.ipynb` contains detailed analysis

### Key Insights
- Age and chest pain type are significant predictors
- Model provides interpretable explanations for each prediction
- Regular monitoring helps ensure model performance over time

## 📁 Project Structure

```
mlops-wk-6-practice/
├── .github/
│   └── workflows/
│       └── cml-k8s-deploy.yml    # CI/CD pipeline
├── k8s/
│   ├── deployment.yaml           # Kubernetes deployment
│   └── service.yaml             # Kubernetes service
├── train.py                     # Model training script
├── serve.py                     # FastAPI application
├── model.joblib                 # Trained model
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── exploration_explainable_ai.ipynb  # Analysis notebook
├── force_plot.html             # SHAP visualization
├── data_drift_report.html      # Data drift report
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🛠️ Model Details

### Algorithm
- **Model**: Logistic Regression
- **Preprocessing**: Mean imputation for missing values
- **Target**: Binary classification (0: no heart disease, 1: heart disease)
- **Evaluation**: Train/test split (80/20) with random state 42

### Performance Considerations
- The model uses `max_iter=10000` to ensure convergence
- Missing values are handled using mean imputation
- Target variable is binarized for binary classification
- Model provides both predictions and probability scores

## 🔧 Development

### Adding New Features
1. Update the `PredictionInput` model in `serve.py`
2. Modify the feature validation logic
3. Retrain the model with `train.py`
4. Update tests and documentation

### Testing
```bash
# Install test dependencies (if any)
pip install pytest

# Run tests (add test files as needed)
pytest
```

### Code Quality
The project follows Python best practices:
- Type hints with Pydantic models
- Input validation
- Error handling
- Structured logging (can be added)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) for the dataset
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [Evidently](https://evidentlyai.com/) for model monitoring

## 📞 Support

For questions and support, please open an issue in the GitHub repository.

## 🔧 Troubleshooting

### Common Issues

**1. Network Error During Training**
```
ConnectionError: Error connecting to server
```
**Solution:** The repository includes a pre-trained model (`model.joblib`). You can skip training and directly run the API server.

**2. Module Not Found Errors**
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution:** Install dependencies with `pip install -r requirements.txt`

**3. Port Already in Use**
```
[Errno 98] Address already in use
```
**Solution:** Use a different port: `uvicorn serve:app --host 0.0.0.0 --port 8001`

**4. Docker Build Issues**
- Ensure Docker is running
- Check if you have sufficient disk space
- Try building with `--no-cache` flag

**5. Kubernetes Deployment Issues**
- Verify cluster connectivity: `kubectl cluster-info`
- Check pod status: `kubectl get pods`
- View logs: `kubectl logs <pod-name>`

---

**Built with ❤️ for the MLOps community**