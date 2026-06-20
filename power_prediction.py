import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

print(">>> RUNNING CLEAN VERSION")


data = []
with open('power_consumption.txt', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        data.append(row)


df = pd.DataFrame(data)
df.columns = df.iloc[0]
df = df[1:]

numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(">>> Data shape BEFORE dropna:", df.shape)
print(">>> NaNs per numeric column:")
print(df[numeric_cols].isna().sum())

df = df.dropna()

print(">>> Data shape AFTER dropna:", df.shape)
print(">>> First 5 rows:")
print(df.head())

if df.empty:
    print(" No data left after cleaning. Please check your dataset formatting.")
    exit()


X = df.drop(['Voltage', 'Date', 'Time'], axis=1)
y = df['Voltage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


def print_full_metrics(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r = np.corrcoef(y_true.astype(float), y_pred.astype(float))[0, 1]

    mean_actual = np.mean(y_true.astype(float))
    rae_denom = np.sum(np.abs(mean_actual - y_true))
    rrse_denom = np.sqrt(np.mean((y_true - mean_actual) ** 2))

    
    try:
        rae = np.sum(np.abs(y_pred - y_true)) / rae_denom * 100 if rae_denom != 0 else float('nan')
    except Exception as e:
        print(f"RAE Error: {e}")
        rae = float('nan')

    try:
        rrse = (rmse / rrse_denom) * 100 if rrse_denom != 0 else float('nan')
    except Exception as e:
        print(f"RRSE Error: {e}")
        rrse = float('nan')

    print("MAE:     ", round(mae, 4))
    print("MSE:     ", round(mse, 4))
    print("RMSE:    ", round(rmse, 4))
    print("R:       ", round(r, 4))
    print("RAE (%): ", round(rae, 2) if not np.isnan(rae) else "NaN")
    print("RRSE (%):", round(rrse, 2) if not np.isnan(rrse) else "NaN")


print_full_metrics("Linear Regression", y_test, y_pred_lr)
print_full_metrics("Random Forest", y_test, y_pred_rf)


plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue')
plt.plot(y_test, y_test, color='black', linestyle='--')
plt.xlabel('Actual Voltage')
plt.ylabel('Predicted Voltage')
plt.title('Linear Regression: Actual vs Predicted')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='orange')
plt.plot(y_test, y_test, color='black', linestyle='--')
plt.xlabel('Actual Voltage')
plt.ylabel('Predicted Voltage')
plt.title('Random Forest: Actual vs Predicted')
plt.grid(True)

plt.tight_layout()
plt.savefig("model_comparison_voltage.png", dpi=300)
plt.show()
