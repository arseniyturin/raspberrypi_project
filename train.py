import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SEED = 11
TEST_SIZE = 0.2

df = pd.read_csv("data/boston.csv")

################################### PIPELINE ##################################

X = df.drop("TARGET", axis=1)
y = df["TARGET"]

X_train, X_test,\
y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("random_forest", RandomForestRegressor()),
    ]
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model score: {score:.3f}")
print(f"MSE: {mse:.3f}")


with open("model/boston.pickle", "wb") as f:
    pickle.dump(pipe, f)