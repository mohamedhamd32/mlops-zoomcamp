import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# === Load and preprocess data ===
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# === File paths ===
input_file = '/home/cdsw/data/yellow_tripdata_2023-05.parquet'   ### i have changes the csv according to the questions request
model_output_path = '/home/cdsw/models/model.bin'

# === Prepare training data ===
df = read_data(input_file)
categorical = ['PULocationID', 'DOLocationID']
target = 'duration'

# Convert to dictionary format for DictVectorizer
train_dicts = df[categorical].to_dict(orient='records')

# Vectorize
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df[target].values

# === Train the model ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === Evaluate the model ===
y_pred = lr.predict(X_train)
# rmse = mean_squared_error(y_train, y_pred, squared=False)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)

print(f"RMSE on training set: {rmse:.2f} minutes")

# === Save the model and vectorizer ===
with open(model_output_path, 'wb') as f_out:
    pickle.dump((dv, lr), f_out)

print(f"Model saved to {model_output_path}")
