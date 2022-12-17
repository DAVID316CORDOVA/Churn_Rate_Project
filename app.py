import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import joblib
diccionario=joblib.load(open("indice_diccionario.pkl","rb"))
app = Flask(__name__)
model = joblib.load("model_lab_platzi.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    result=request.form
    CreditScore=result["CreditScore"]
    Age=result["Age"]
    Tenure=result["Tenure"]
    Balance=result["Balance"]
    NumOfProducts=result["NumOfProducts"]
    HasCrCard=result["HasCrCard"]
    IsActiveMember=result["IsActiveMember"]
    EstimatedSalary=result["EstimatedSalary"]
    Origin=result["Origin"]
    Sex=result["Sexo"]
    
    vector_zeros=np.zeros(len(diccionario))

    vector_zeros[0]=CreditScore
    
    vector_zeros[1]=Age
    
    vector_zeros[2]=Tenure
    
    vector_zeros[3]=Balance
    
    vector_zeros[4]=NumOfProducts
    
    vector_zeros[diccionario["HasCrCard"]]=HasCrCard
    
    vector_zeros[diccionario["IsActiveMember"]]=IsActiveMember
    
    vector_zeros[7]=EstimatedSalary
    
    vector_zeros[diccionario[str(Origin)]]=1

    vector_zeros[diccionario[str(Sex)]]=1

    escalador_x=joblib.load(open("scaler_x.pkl","rb"))

    data=pd.DataFrame(vector_zeros).T
    data.columns=[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'France', 'Germany', 'Spain',
       'Female', 'Male']]

    data_transformada=escalador_x.transform(data)
    prediction = model.predict(data_transformada).round()[[0]]
    
    def pred(x):
        if x==1:
            return "El cliente cancelará su suscripción"
        elif x==0:
            return "El cliente continuará con su suscripción"
    

    prediccion_real=pred(prediction)

    return render_template('home.html', prediction_text=f"{(prediccion_real)}")

if __name__ == '__main__':
    app.run(debug=True)
