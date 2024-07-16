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
y1 = dataset1.iloc[:, -1].values
y2 = dataset2.iloc[:, -1].values
dataset1.drop(["G1", "G2", "G3"], axis = 1)
dataset2.drop(["G1", "G2", "G3"], axis = 1)

#%% Data Preprocessing
#Label Encoder
le = LabelEncoder()
columns1 = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
           "nursery", "higher", "internet", "romantic"]
for col in columns1:
    dataset1[col] = le.fit_transform(dataset1[col])
    dataset2[col] = le.transform(dataset2[col])

#OneHotEncoder
columns2 = ["Mjob", "Fjob", "reason", "guardian"]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns2)], remainder='passthrough')
X1 = np.array(ct.fit_transform(dataset1))
X2 = np.array(ct.fit_transform(dataset2))

#%% Split data 
x1_train , x1_test , y1_train , y1_test = train_test_split(X1,y1 , test_size= 0.2 , random_state= 42,shuffle=True)
x2_train , x2_test , y2_train , y2_test = train_test_split(X2,y2 , test_size= 0.2 , random_state= 42, shuffle=True)

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,max_depth=10)
RFR.fit(x1_train,y1_train)
r2_score(y1_test , RFR.predict(x1_test))

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(criterion="squared_error", random_state=69)
DTR.fit(x1_train,y1_train)
r2_score(y1_test, DTR.predict(x1_test))

#%% Linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x1_train , y1_train)
r2_score(y1_test , LR.predict(x1_test))
#%% Polynomial Features Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
PolyReg = PolynomialFeatures(degree=3)
X_Poly = PolyReg.fit_transform(x1_train)
LR = LinearRegression()
LR.fit(X_Poly , y1_train)
r2_score(y1_test , LR.predict(PolyReg.transform(x1_test)))

#%% Data Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(x1_train.reshape(-1, 1))
y_train = sc_y.fit_transform(y1_train.reshape(-1, 1))
#%% SVM
from sklearn.svm import SVR
SupportVR = SVR(kernel="rbf")
SupportVR.fit(X_train , y_train)
r2_score(y1_test,SupportVR.predict(sc_X.transform(x1_test)))
# %% test
r2_score(y2_test,LR.predict(x2_test))
