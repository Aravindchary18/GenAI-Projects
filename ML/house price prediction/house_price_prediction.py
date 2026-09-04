import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import joblib as jb

df=pd.read_csv("train.csv")

X=df[["GrLivArea","BedroomAbvGr","YearBuilt","OverallQual","FullBath","GarageCars"]]
y=df["SalePrice"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=19)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)

print("RMSE:",rmse)
print("R2 score:",r2)

jb.dump(model,"house_model.pkl")
