#%% Import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import seaborn as sns

#%% Import Data
dataset1 = pd.read_csv('student-mat.csv', delimiter = ";")
dataset2 = pd.read_csv('student-mat.csv', delimiter = ";")

#%% Data Preprocessing
dataset1.drop(["G1", "G2"], axis = 1)
dataset2.drop(["G1", "G2"], axis = 1)

#Label Encoder
le = LabelEncoder()
columns1 = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
           "nursery", "higher", "internet", "romantic"]
for col in columns1:
    dataset1[col] = le.fit_transform(dataset1[col])
    dataset2[col] = le.transform(dataset2[col])

#OneHotEncoder
columns2 = ["Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian"]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns2)], remainder='passthrough')
dataset1 = np.array(ct.fit_transform(dataset1))
dataset2 = np.array(ct.fit_transform(dataset2))

print(dataset1)

# %%
