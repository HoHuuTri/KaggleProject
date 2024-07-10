#%% Import 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# %% Clean Data
data = pd.read_csv("train.csv")
x = data.drop("SalePrice" , axis= "columns")
y = data["SalePrice"]
y.info()
def CleanData(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(df["LotFrontage"].mean())
    df.drop(["Alley","MasVnrType","MiscFeature","PoolQC","Fence","FireplaceQu","Id"] , inplace = True , axis= 1)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean())
    df["BsmtExposure"] = df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0])
    df["BsmtFinType2"] = df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0])
    df["BsmtExposure"] = df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0])
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna(df["BsmtFinType1"].mode()[0])
    df["BsmtQual"] = df["BsmtQual"].fillna(df["BsmtQual"].mode()[0])
    df["BsmtCond"] = df["BsmtCond"].fillna(df["BsmtCond"].mode()[0])
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    df["GarageFinish"] = df["GarageFinish"].fillna(df["GarageFinish"].mode()[0])
    df["GarageType"] = df["GarageType"].fillna(df["GarageType"].mode()[0])
    df["GarageQual"] = df["GarageQual"].fillna(df["GarageQual"].mode()[0])
    df["GarageCond"] = df["GarageCond"].fillna(df["GarageCond"].mode()[0])
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].mean())
    return df 
X = CleanData(x)

X.head()

sns.heatmap(X.isnull() , yticklabels=False, cbar=False)
collums = ["MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope",'Neighborhood',"Condition1"
           ,"Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","ExterQual"
           ,"ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating"
           ,"HeatingQC","CentralAir","Electrical","KitchenQual","Functional","GarageType","GarageFinish","GarageQual"
           ,"GarageCond","PavedDrive","SaleType","SaleCondition"]

def Catagory_onehot_multcols(multcolums,df):
    df_final=df 
    i=0 
    for fields in multcolums:
        df1 = pd.get_dummies(df[fields],drop_first=True)
        df.drop([fields],axis = 1 , inplace = True)
        if i ==0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final,df1] , axis= 1)
    df_final = pd.concat([df,df_final] , axis= 1) 
    return df_final
Catagory_onehot_multcols(collums , X)
X.head()

#%% Check For Data
X["GarageCond"].value_counts()
X.info()

# %% Check For Abnormal data

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