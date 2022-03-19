import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SEED = 11
TEST_SIZE = 0.2

################################### PIPELINE ##################################

X_train, X_test,\
y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)


pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)