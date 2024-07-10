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
import seaborn as sns
# %% Read input and split Inde and De
data = pd.read_csv("train.csv")
X = data
y = data["SalePrice"]
y.info()
test = pd.read_csv("test.csv")
test2 = pd.read_csv("test.csv")

# %% Handling train data
X['LotFrontage']=X['LotFrontage'].fillna(X['LotFrontage'].mean())
X.drop(['Alley'],axis=1,inplace=True)
X['FireplaceQu']=X['FireplaceQu'].fillna(X['FireplaceQu'].mode()[0])
X['GarageType']=X['GarageType'].fillna(X['GarageType'].mode()[0])
X.drop(['GarageYrBlt'],axis=1,inplace=True)
X['GarageFinish']=X['GarageFinish'].fillna(X['GarageFinish'].mode()[0])
X['GarageQual']=X['GarageQual'].fillna(X['GarageQual'].mode()[0])
X['GarageCond']=X['GarageCond'].fillna(X['GarageCond'].mode()[0])
X.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
X.drop(['Id'],axis=1,inplace=True)
X['MasVnrType']=X['MasVnrType'].fillna(X['MasVnrType'].mode()[0])
X['MasVnrArea']=X['MasVnrArea'].fillna(X['MasVnrArea'].mode()[0])
X['BsmtExposure']=X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0])
X['BsmtFinType2']=X['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0])
X.dropna(inplace=True)

#%% Handing test data

test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.drop(['Alley'],axis=1,inplace=True)
test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['FireplaceQu']=test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])
test.drop(['GarageYrBlt'],axis=1,inplace=True)
test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])

test.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)
test['MasVnrType']=test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtExposure']=test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])
test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mean())
test['SaleType']=test['SaleType'].fillna(test['SaleType'].mode()[0])
test.to_csv('formulatedtest.csv',index=False)

#%% Combine

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
main_df=X.copy()
test_df=pd.read_csv('formulatedtest.csv')
final_df=pd.concat([X,test_df],axis=0)
final_df=category_onehot_multcols(columns)
final_df =final_df.loc[:,~final_df.columns.duplicated()]
X_Train = final_df.iloc[:1422,:]
X_Test = final_df.iloc[1422:,:]
X_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=X_Train.drop(['SalePrice'],axis=1)
y_train = X_Train['SalePrice']

# %% Training Random Forest Regressor and Output
RFG = RandomForestRegressor(n_estimators = 8) # Change n_estimators to fit the model.
RFG.fit(X_train, y_train)
Predictions = RFG.predict(X_Test)
arr = pd.DataFrame(test2["Id"] )
Predictions = pd.DataFrame(Predictions)
arr["SalePrice"] = Predictions
arr.to_csv("submission.csv" , index= False)
# %%
