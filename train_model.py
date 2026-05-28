import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("solar_data1.csv")

# Inputs
X = df[['Temperature',
        'Pressure',
        'Humidity',
        'WindDirection(Degrees)',
        'Speed',
        'hour',
        'month']]

# Target
y = df['Radiation']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor()

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "solar_model.pkl")

print("Model trained successfully")