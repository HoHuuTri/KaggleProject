#%% Import 
import pandas as pd 

# %% Read input and split Inde and De
data = pd.read_csv("train.csv")
x = data.drop("SalePrice" , axis= "columns")
y = data["SalePrice"]
y.info()
# %% Check For Abnormal data

# %% Clean Data
def CleanData(data):
    data = data[["YearBuilt","OverallQual","OverallCond","1stFlrSF","2ndFlrSF","MSZoning","LotArea"
                 ,"HouseStyle","Condition1" , "Condition2","RoofMatl","Exterior1st","Exterior2nd"]]
    return data
X = CleanData(x)
X.head()
X.info()
# %%
