import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib

data={"Experience":[1,2,3,4,5,6,7,8,9,10],
        "Salary":[45000,50000,60000,80000,110000,150000,200000,300000,500000,1000000]}

df=pd.DataFrame(data)

x=df[["Experience"]]
y=df["Salary"]

model=LinearRegression()

model.fit(x,y)

joblib.dump(model,"model.pkl")
print("Model trained and saved as model.pkl")