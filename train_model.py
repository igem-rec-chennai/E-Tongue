import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt

paths = [
    r"C:\Users\mervi\Downloads\ig\NaCl_only_Dataset_with_300_Concentrations.csv",
    r"C:\Users\mervi\Downloads\ig\NaCl_Pep1_Dataset_with_300_Concentrations.csv",
    r"C:\Users\mervi\Downloads\ig\NaCl_Pep2_Dataset_with_300_Concentrations.csv",
    r"C:\Users\mervi\Downloads\ig\NaCl_Pep3_Dataset_with_300_Concentrations.csv"
]

dfs = [pd.read_csv(p) for p in paths]
data = pd.concat(dfs, ignore_index=True)
print("✅ Combined dataset shape:", data.shape)

X = data.drop(columns=['Saltiness_Score', 'Interpretation', 'Sample_ID', 'Peptide'])
y = data['Saltiness_Score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv1D(64, 1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 1, activation='relu'),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_train_pred, y_train, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title("Train Set: Predicted vs True Saltiness")
plt.xlabel("Predicted Saltiness")
plt.ylabel("True Saltiness")

plt.subplot(1,2,2)
plt.scatter(y_test_pred, y_test, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Test Set: Predicted vs True Saltiness")
plt.xlabel("Predicted Saltiness")
plt.ylabel("True Saltiness")

plt.tight_layout()
plt.show()

loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Model evaluation — Loss: {loss:.4f}, MAE: {mae:.4f}")

model.save("saltiness_cnn_lstm_model.h5")
joblib.dump(scaler, "saltiness_scaler.pkl")

print("\n📁 Model saved as 'saltiness_cnn_lstm_model.h5'")
print("📁 Scaler saved as 'saltiness_scaler.pkl'")
