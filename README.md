📋 Project Overview

This project aims to predict electric vehicle (EV) charging energy consumption (in kWh) using neural networks. The model combines EV charging session data with local traffic distribution data to understand how traffic patterns might influence charging behavior.

###📊 Datasets Used

Dataset 1: EV Charging Reports

Rows: 6,878 entries
Features: 15 columns including:

Session metadata (session_ID, Garage_ID, User_ID, User_type)
Timing information (Start_plugin, End_plugout, Duration_hours)
Energy consumption (El_kWh)
Temporal categories (month_plugin, weekdays_plugin, Plugin_category, Duration_category)
Dataset 6: Local Traffic Distribution

Rows: 10,248 entries
Features: 7 columns including:

Time intervals (Date_from, Date_to)
Traffic counts at 5 locations (KROPPAN BRU, MOHOLTLIA, SELSBAKK, MOHOLT RAMPE 2, Jonsvannsveien vest for Steinanvegen)
🛠️ Data Processing Pipeline

1. Data Merging

Used pd.merge_asof() to merge charging sessions with nearest hourly traffic data
Applied 1-hour tolerance window for matching
Required datetime conversion and sorting of both datasets

2. Data Cleaning

Columns removed: 12 columns not needed for analysis (session_ID, Garage_ID, etc.)

Data type conversion:

Converted comma-separated numeric strings to float (e.g., "0,3" → 0.3)
Applied pd.to_numeric() with errors='coerce' to handle invalid values
Rows dropped: 45 rows with missing values (final dataset: 6,833 rows)

3. Feature Engineering

Categorical encoding: Applied LabelEncoder to:

User_type (Private/Other)
month_plugin
weekdays_plugin
Scaling: Standardized all features and target using StandardScaler
Train/Test split: 80/20 split with random seed 42

###🤖 Neural Network Architecture

Input Layer: 9 features
Hidden Layer 1: 56 neurons (ReLU activation)
Hidden Layer 2: 28 neurons (ReLU activation)
Output Layer: 1 neuron (energy consumption in kWh)

Training Configuration:

Loss Function: Mean Squared Error (MSE)
Optimizer: Adam (learning rate: 0.007)
Epochs: 3,000
Train/Test Split: 80/20

Features:
The model uses the following 9 input features:

User_type - Type of user (encoded)
Duration_hours - Charging session duration
month_plugin - Month when charging started (encoded)
weekdays_plugin - Day of week (encoded)
KROPPAN BRU - Traffic volume at location 1
MOHOLTLIA - Traffic volume at location 2
SELSBAKK - Traffic volume at location 3
MOHOLT RAMPE 2 - Traffic volume at location 4
Jonsvannsveien vest for Steinanvegen - Traffic volume at location 5

###📈 Results

Model Performance (Test Set)

MSE (scaled): 0.8194
MSE (kWh): 115.90
MAE (kWh): 7.20
R² Score: 0.1092
Baseline Comparison (Predicting mean of training data)

MSE (kWh): 130.11
MAE (kWh): 8.01
R² Score: ≈0.00
Interpretation: The neural network shows modest improvement over the simple baseline predictor, with 11% variance explained (R² = 0.109) and about 10% reduction in MSE compared to predicting the mean.

###💾 Model Persistence

The trained model weights are saved as model_state_dict.pth for portability and future inference.

🔧 Technical Details

###Dependencies

Python 3.12.12
Core libraries:

torch (PyTorch for neural networks)
pandas (data manipulation)
numpy (numerical operations)
scikit-learn (preprocessing and metrics)
random (reproducibility)
Reproducibility

Seed set to 48 for all random operations (Python, NumPy, PyTorch)
Ensures consistent results across runs

###🚀 Usage Example

1. Prepare your data:

Place Dataset 1_EV charging reports.csv and Dataset 6_Local traffic distribution.csv in the project directory


2. Run the notebook:

jupyter notebook code.ipynb

3. Load the trained model:

   import torch
   import torch.nn as nn
   
4. Define model architecture
   model = nn.Sequential(
       nn.Linear(9, 56),
       nn.ReLU(), 
       nn.Linear(56, 28),
       nn.ReLU(),
       nn.Linear(28, 1)
   )
   
5.  Load trained weights
   model.load_state_dict(torch.load('model_state_dict.pth'))
   model.eval()

6. Data Preprocessing
The pipeline includes:

Temporal alignment of charging and traffic data
Conversion of numeric columns (handling comma decimal separators)
Label encoding for categorical features
Standard scaling of all features and target variable
Removal of rows with missing or invalid values (45 rows dropped)


###📁 File Structure

├── code.ipynb                              # Main notebook
├── Dataset 1_EV charging reports.csv       # Charging session data
├── Dataset 6_Local traffic distribution.csv # Traffic data
├── model_state_dict.pth                    # Trained model weights
└── README.md                               # This file

###Future Improvements:

- Experiment with deeper architectures or different activation functions
- Add regularization techniques (dropout, L2 regularization)
- Incorporate additional features (weather data, holidays, etc.)
- Try ensemble methods or gradient boosting models
- Perform hyperparameter tuning
- Analyze feature importance and model interpretability
