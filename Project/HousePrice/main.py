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
from sklearn.metrics import r2_score
# %% Read input and split Inde and De
data = pd.read_csv("train.csv")
x = data.drop("SalePrice" , axis= "columns")
y = data["SalePrice"]
y.info()
# %% Check For Abnormal data


#im testing how to merge github


# %% Clean Data
def CleanData(data):
    data = data[["YearBuilt","OverallQual","OverallCond","1stFlrSF","2ndFlrSF","MSZoning","LotArea"
                 ,"HouseStyle","Condition1" , "Condition2","RoofMatl","Exterior1st","Exterior2nd"]]
    data["HouseStyle"].str.replace("")
    return data
X = CleanData(x)
X.head()
X.info()
#baka
#test2
#test3
#test4
#test5
#test6
#test7
#test8
#test9
#test10
#test11
#test12
#test13
#test14
#test15
#test16
#test17
#
# %% Split Training and Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# %% Training Random Forest Regressor
RFG = RandomForestRegressor(n_estimators = 10, random_state = 0) # Change n_estimators to fit the model.
RFG.fit(X_train, y_train)
y_pred = RFG.predict(X_test)
r2_score(y_test, y_pred)

# %% Training Multiple Linear Regression 
MLR = LinearRegression()
MLR.fit(X_train, y_train)
y_pred = MLR.predict(X_test)
r2_score(y_test, y_pred)

# %% Training Polynomial Linear Regression
poly_reg = PolynomialFeatures(degree = 4) #Change the degree to fit the model.
X_poly = poly_reg.fit_transform(X_train)
PLR = LinearRegression()
PLR.fit(X_poly, y_train)
y_pred = PLR.predict(poly_reg.transform(X_test))
r2_score(y_test, y_pred)

# %% Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# %% Training SVR 
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))
r2_score(y_test, y_pred)