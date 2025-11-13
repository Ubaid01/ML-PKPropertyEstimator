# 🏠 Pakistan House Price Predictor

A ML based web application for predicting property prices across major Pakistani cities built with Python, scikit-learn, and Streamlit.

## Overview

This project addresses the lack of transparent property valuation in Pakistan's real estate market by providing data-driven price predictions using machine learning. The system analyzes over **1.7 lakh** property listings from zameen.com across five major cities.

### Key Features

- Purpose-specific models for "For Sale" and "For Rent" properties.
- Achieved R² scores of 0.84 (sale) and 0.85 (rent).
- Priority-based similar property recommendations.
- User-friendly Streamlit web interface with instant price estimates alongside insights.

## Installation

- Need Python 3.8 or higher

1. **Clone the repository**
```bash
git clone https://github.com/Ubaid01/ML-PKPropertyEstimator.git
```

2. **Create virtual environment (recommended)**
```bash
python -m venv cust_env
source cust_env/Scripts/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required files**

Place these files in the project directory:
- `model_sale.pkl` - Trained model for sale properties
- `model_rent.pkl` - Trained model for rental properties
- `train_stats_sale.pkl` - Training statistics for sale model
- `train_stats_rent.pkl` - Training statistics for rent model
- `cleaned_data.csv` - Preprocessed dataset for similar property matching

5. **Run the app**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Training Models from Scratch

1. **Preprocess data**
```bash
python 01_data_preprocessing.py
```

2. **Train models**
```bash
python 02_machine_learning.py
```

## 🧪 Model Performance

### For Sale Properties (77,961 training samples)

| Model | R² Score | RMSE (PKR) | MAPE |
|-------|----------|------------|------|
| Linear Regression | 0.767 | 15.9M | 43.0% |
| Random Forest | 0.839 | 12.1M | 34.0% |
| XGBoost | 0.839 | 12.1M | 33.6% |
| **LightGBM** | **0.844** | **12.0M** | **33.5%** |

### For Rent Properties (25,294 training samples)

| Model | R² Score | RMSE (PKR) | MAPE |
|-------|----------|------------|------|
| Linear Regression | 0.785 | 96,081 | 30.4% |
| Random Forest | 0.843 | 88,168 | 24.5% |
| LightGBM | 0.846 | 89,546 | 24.8% |
| **XGBoost** | **0.849** | **88,000** | **24.4%** |

## 🔍 Methodology

### Data Preprocessing Pipeline
- Removed invalid entries and duplicates
- Standardized area measurements (Marla, Kanal → Square Feet)
- Applied log transformation to handle price skewness
- City-wise outlier removal (top/bottom 0.5%)
- `room_density`: Rooms per square foot
- `bed_bath_ratio`: Bedroom to bathroom ratio
- `area_category`: Property size classification
- `bedroom_category`: Bedroom count bins

### Why Separate Models?

Combining sale and rental properties in one model yielded misleading R² = 0.97 because the model simply learned to classify property purpose rather than predict actual prices. Separate models provide realistic performance (R² ≈ 0.84).

### Algorithm Selection
- **LightGBM for Sale**: Better for larger datasets with complex patterns
- **XGBoost for Rent**: Superior regularization prevents overfitting on smaller dataset

## 👥 Contributors

<p>
  <a href="https://github.com/Ubaid01/ML-PKPropertyEstimator/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=Ubaid01/ML-PKPropertyEstimator" height="50px" />
  </a>
</p>


## ⚠️ Disclaimer

This tool provides price estimates based on historical data and should be used as a reference only. Actual current property prices will surely vary based on market conditions, negotiations, and factors not captured in the dataset.

---

**⭐ Star this repo if you find it helpful!**