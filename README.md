📋 Project Overview

This project aims to predict electric vehicle (EV) charging energy consumption (in kWh) using neural networks. The model combines EV charging session data with local traffic distribution data to understand how traffic patterns might influence charging behavior.

📊 Datasets Used

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
🤖 Neural Network Architecture

Model Structure (PyTorch)

text
Sequential(
  (0): Linear(in_features=9, out_features=56)
  (1): ReLU()
  (2): Linear(in_features=56, out_features=28)
  (3): ReLU()
  (4): Linear(in_features=28, out_features=1)
)
Training Configuration

Loss function: Mean Squared Error (MSE)
Optimizer: Adam with learning rate = 0.007
Epochs: 3,000
Batch size: Full batch (no mini-batching)
Input features: 9 features after preprocessing
📈 Results

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

💾 Model Persistence

The trained model weights are saved as model_state_dict.pth for portability and future inference.

🔧 Technical Details

Dependencies

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
🚀 Usage Example

python
# Load and preprocess data
df1 = pd.read_csv('Dataset 1_EV charging reports.csv', sep=';')
df2 = pd.read_csv('Dataset 6_Local traffic distribution.csv', sep=';')

# Apply same preprocessing pipeline
# ... (see notebook for details)

# Load trained model
model = nn.Sequential(...)  # Same architecture
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_new_tensor)
📝 Notes

The project demonstrates a basic neural network regression approach
Traffic data merging assumes charging sessions are influenced by nearby traffic patterns within a 1-hour window
Performance suggests room for improvement - potential next steps include:

Feature engineering (interaction terms, lag features)
More complex model architectures
Hyperparameter tuning
Additional external data sources
📁 File Structure

text
project/
├── code.ipynb               # Main Jupyter notebook
├── model_state_dict.pth     # Trained model weights
├── Dataset 1_EV charging reports.csv
├── Dataset 6_Local traffic distribution.csv
└── README.md               # This file
