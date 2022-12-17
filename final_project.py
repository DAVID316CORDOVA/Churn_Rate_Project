import pandas as pd
import numpy as np
import joblib

pd.options.display.float_format="{:.2f}".format

#Loading the new dataset created in the JupyerNotebook using Imblearn library

df=pd.read_csv("new_data_churn_lab_platzi.csv")

#Defining X and y

X=df.drop(columns=["Exited"])
y=df[["Exited"]]


#Standardization the data

from sklearn.preprocessing import StandardScaler

scaler_x=StandardScaler()

scaler_x.fit(X)

X=pd.DataFrame(scaler_x.transform(X),columns=X.columns)


#Importing the model chosen in the JupyterNotebook with the best max_depth found in the fine tuning
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(max_depth=19)

model.fit(X,y)

#Saving the model and the scaler

joblib.dump(model, open("model_lab_platzi.pkl", "wb"))

#Saving the scaler

joblib.dump(scaler_x,"scaler_x.pkl")

#Saving a dictionary with all the columns

diccionario=dict(zip(X.columns,range(X.shape[1])))

joblib.dump(diccionario,open("indice_diccionario.pkl","wb"))

print("Diccionario de columnas:")

print(diccionario)