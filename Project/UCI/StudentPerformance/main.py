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
#%% Split data 
X = dataset1.drop("G3")
y = dataset1["G3"]
x_train , y_train , x_test , y_test = train_test_split(X,y , test_size= 0.2 , random_state= 42, shuffle=True)

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,random_state=69,criterion="squared_error",max_depth=10)
RFR.fit(x_train,y_train)
r2_score(y_test , RFR.predict(x_test))

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(criterion="squared_error", random_state=69)
DTR.fit(x_train,y_train)
r2_score(y_test, DTR.predict(x_test))

#%% Linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train , y_train)
r2_score(y_test , LR.predict(x_test))
#%% Polynomial Features Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
PolyReg = PolynomialFeatures(degree=3)
X_Poly = PolyReg.fit_transform(x_train)
LR = LinearRegression()
LinearRegression.fit(X_Poly , y_train)
r2_score(y_test , LR.predict(PolyReg.transform(x_test)))

#%% Data Scaling 
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)
#%% SVM
from sklearn.svm import SVR
SupportVR = SVR(kernel="rbf")
SupportVR.fit(x_train , y_train)
r2_score(y_test,SupportVR.predict(x_test))
