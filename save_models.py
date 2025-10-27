# save_models.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Step 1: Load your dataset
# ------------------------------
df = pd.read_csv('final_clean_house_data.csv')  

# ------------------------------
# Step 2: Define features and target
# ------------------------------
# Replace with your actual feature columns
feature_cols = ['FLOOR','BEDROOM','BATHROOM','Land_in_aana','road_access_in_feet','AGE','car_parking','bhaktapur','chitwan','kaski','kathmandu','lalitpur']
target_col = 'price_in_crore'

X = df[feature_cols]
y = df[target_col]

# Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Step 3: Split into train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 4: Train models
# ------------------------------

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# ------------------------------
# Step 5: Save models as pickle files
# ------------------------------
with open('linear_model.pickle', 'wb') as f:
    pickle.dump(linear_model, f)

with open('RandomForest_model.pickle', 'wb') as f:
    pickle.dump(rf_model, f)

with open('GradientBoosting_model.pickle', 'wb') as f:
    pickle.dump(gb_model, f)

# Optional: save scaler if needed in app.py
with open('scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

print("All models and scaler saved successfully!")
