import pandas as pd
import xgboost as xgb
import numpy as np

# Load test data
df_raw = pd.read_csv("x_test.csv", sep=';')
ids = df_raw['id'].values
print(f"Loaded {len(ids)} IDs")

df_proc = pd.read_csv("x_test_processed.csv", sep=';')
X_test = df_proc.iloc[:, 1:].values.astype(np.float32)  # (189, 40)
print(f" Features shape: (189, 40)")

feature_names = [
    'day', 'hr_maxHeartRate', 'hr_minHeartRate', 'hr_restingHeartRate',
    'hr_lastSevenDaysAvgRestingHeartRate', 'resp_lowestRespirationValue',
    'resp_highestRespirationValue', 'resp_avgWakingRespirationValue',
    'resp_avgSleepRespirationValue', 'resp_avgTomorrowSleepRespirationValue',
    'str_maxStressLevel', 'str_avgStressLevel', 'sleep_sleepTimeSeconds',
    'sleep_napTimeSeconds', 'sleep_unmeasurableSleepSeconds',
    'sleep_deepSleepSeconds', 'sleep_lightSleepSeconds', 'sleep_remSleepSeconds',
    'sleep_awakeSleepSeconds', 'act_totalCalories', 'act_activeKilocalories',
    'act_distance', 'act_activeTime', 'sleep_averageRespirationValue',
    'sleep_lowestRespirationValue', 'sleep_highestRespirationValue',
    'sleep_awakeCount', 'sleep_avgSleepStress', 'sleep_avgHeartRate',
    'hr_time_series_mean', 'hr_time_series_std', 'hr_time_series_min',
    'hr_time_series_max', 'resp_time_series_mean', 'resp_time_series_std',
    'resp_time_series_min', 'resp_time_series_max', 'stress_time_series_mean',
    'stress_time_series_std', 'stress_time_series_min', 'stress_time_series_max'
]

day_column = np.zeros((X_test.shape[0], 1), dtype=np.float32)
X_test_full = np.hstack([day_column, X_test])  # (189, 41)
print(f" Full features with 'day': (189, 41)")

dtest = xgb.DMatrix(X_test_full, feature_names=feature_names)
print(" DMatrix with EXACT feature names created!")

# Load model
model = xgb.Booster(model_file="final_model.json")
print("Model LOADED from JSON!")

# Predict
preds = model.predict(dtest)
print(f"Predictions: {preds.mean():.1f}±{preds.std():.1f} [{preds.min():.1f}-{preds.max():.1f}]")

# Scale to 0-100 if needed
if preds.max() <= 1.0:
    preds *= 100

# Save submission
submission = pd.DataFrame({'id': ids, 'label': preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Final stats: mean={preds.mean():.2f}, std={preds.std():.2f}")
