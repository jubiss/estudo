import mlflow
from mlflow.models import infer_signature
import pandas as pd
import src.features.build_features as bf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv(rf'data\external\simulated_listings1.csv')

train.loc[train['rooms'] >= 4, 'rooms'] = 4
train.loc[train['garages'] >= 4, 'garages'] = 4

X = train[['rooms', 'garages', 'useful_area', 'interior_quality']]
y = train['value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LinearRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
#
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("check-localhost-connection")
# 
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("Mean Squared Error", mse)
    
    mlflow.set_tag("Training info", "Testing Simple linear regression")