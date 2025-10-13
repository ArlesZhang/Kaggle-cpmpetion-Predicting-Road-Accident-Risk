# Kaggle Competition Setup & Execution - Day 1

## 🛠️ Core Linux Shell Commands Used

### 1. Environment Setup
```bash
# Create competition workspace
mkdir -p ~/kaggle_competitions/road_accident
cd ~/kaggle_competitions/road_accident

# Create and activate virtual environment
python3 -m venv kaggle_env
source kaggle_env/bin/activate

# Install competition packages
pip install pandas numpy scikit-learn lightgbm xgboost catboost kaggle
```

### 2. Kaggle API Configuration
```bash
# Setup Kaggle credentials (after downloading kaggle.json)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
kaggle competitions download -c playground-series-s5e10
unzip playground-series-s5e10.zip
```

### 3. TO START WORKING
```bash
cd ~/ml_competitions
source kaggle_env/bin/activate
cd ~/kaggle_competitions/road_accident
```

### 4. Core Model Execution Scripts

**First Model (Random Forest):**
```bash
cat > first_model.py << 'EOF'
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Basic preprocessing
X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']
X_test = test.drop('id', axis=1)

# Convert categorical to numeric
for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Predict and save
predictions = model.predict(X_test)
submission = pd.read_csv('sample_submission.csv')
submission['accident_risk'] = predictions
submission.to_csv('first_model_submission.csv', index=False)
print("First submission created!")
EOF

python first_model.py
```

**LightGBM Model:**
```bash
cat > lightgbm_model.py << 'EOF'
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']
X_test = test.drop('id', axis=1)

# Convert categorical
for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

# LightGBM with validation
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=1000, random_state=42)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

predictions = model.predict(X_test)
submission = pd.read_csv('sample_submission.csv')
submission['accident_risk'] = predictions
submission.to_csv('lightgbm_submission.csv', index=False)
print("LightGBM submission created!")
EOF

python lightgbm_model.py
```

### 5. Submission Commands
```bash
# Submit models to Kaggle
kaggle competitions submit -c playground-series-s5e10 -f first_model_submission.csv -m "First model - Random Forest"
kaggle competitions submit -c playground-series-s5e10 -f lightgbm_submission.csv -m "LightGBM baseline"

# Check submission status
kaggle competitions submissions -c playground-series-s5e10
```

### 6. Quick Analysis Script
```bash
cat > quick_analysis.py << 'EOF'
import pandas as pd

train = pd.read_csv('train.csv')
print(f"Data shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"Target stats - Min: {train['accident_risk'].min():.3f}, Max: {train['accident_risk'].max():.3f}, Mean: {train['accident_risk'].mean():.3f}")
EOF

python quick_analysis.py
```

## 📊 Results Summary
- **Best Score**: 0.05564 RMSE (Random Forest)
- **Models Built**: 5 (RF, LightGBM, CatBoost, Enhanced RF, Ensemble)
- **Submissions Used**: 5/5 daily limit
- **Gap to 1st**: 0.00027 RMSE

## Tomorrow's Priority Tasks

1. Hyperparameter tuning for that final 0.00027 improvement
2. Cross-validation implementation for more reliable scores
3. Ensemble optimization of top 2 models
4. Feature selection based on importance analysis
5. Submission limit management (5 per day)

## 🎯 Key Shell Workflow
1. **Setup**: `mkdir → venv → pip install`
2. **Data**: `kaggle download → unzip`
3. **Model**: `cat > script.py → python script.py`
4. **Submit**: `kaggle competitions submit`
5. **Check**: `kaggle competitions submissions`

## 💡 Three Technical Discussion Points

1. **Virtual Environment Necessity**: The `python3 -m venv` + `pip install` workflow proved essential for managing dependencies. In production, would you use Docker instead, or are lightweight venvs sufficient for ML competitions?

2. **The Kaggle CLI Power**: Using `kaggle competitions download/submit` transformed a manual process into a 3-command workflow. What other data science tasks could benefit from similar CLI automation?

3. **The 0.00027 Gap**: When models are this close (0.05564 vs 0.05537), is the difference meaningful model improvement or just statistical noise? At what point should we declare "good enough" and focus effort elsewhere?

---

*Welcome to follow My GitHub.* 🚀
