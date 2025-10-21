import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
import streamlit as st

model=tf.keras.models.load_model('model.h5')

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open ('encoder_geo.pkl','rb') as file:
    encoder_geo=pickle.load(file)    

with open('scaler.plk','rb') as file:
  scaler=pickle.load(file)    


st.title("Customer Churn Prediction ")

CreditScore=st.number_input('Credit Score')
Gender=st.selectbox('Gender',label_encoder_gender.classes_)
Age=st.slider('Age',18,90)
Tenure=st.slider('Tenure',0,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider("NumOfProducts",1,4)
HasCrCard=st.selectbox("HasCrCard",[0,1])
IsActiveMember=st.selectbox("IsActiveMember",[0,1])
EstimatedSalary=st.number_input("EstimatedSalary")
Geography=st.selectbox("Geography",encoder_geo.categories_[0])


#input data

input_data=pd.DataFrame({
   'CreditScore':[CreditScore],
   'Gender':[label_encoder_gender.transform([Gender])[0],
   'Age':[Age],
   'Tenure':[Tenure],
   'Balance':[Balance],
   'NumOfProduts':[NumOfProducts],
   'HasCrCard':[HasCrCard],
   'IsActiveMember':[IsActiveMember],
   'EstimatedSalary':[EstimatedSalary]
   
})

geo_encoded=encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=encoder_geo.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaler=scaler.transform(input_data)


predtion=model.predict(input_data_scaler)
predtion_proba=predtion[0][0]

st.write(f'churn probabillity:{predtion_proba:.2f}')

if predtion_proba >0.5:
   st.write("the customer is likely to churn")
else:
   st.write("the customer is not likely to churn")   


