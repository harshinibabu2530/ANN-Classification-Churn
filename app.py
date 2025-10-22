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

#user input
creditScore=st.number_input('Credit Score')
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,90)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
numOfProducts=st.slider("NumOfProducts",1,4)
hasCrCard=st.selectbox("HasCrCard",[0,1])
isActiveMember=st.selectbox("IsActiveMember",[0,1])
estimatedSalary=st.number_input("EstimatedSalary")
geography=st.selectbox("Geography",encoder_geo.categories_[0])


#input data

input_data=pd.DataFrame ({
   'CreditScore':[creditScore],
   'Gender':[label_encoder_gender.transform([gender])[0]],
   'Age':[age],
   'Tenure':[tenure],
   'Balance':[balance],
   'NumOfProducts':[numOfProducts],
   'HasCrCard':[hasCrCard],
   'IsActiveMember':[isActiveMember],
   'EstimatedSalary':[estimatedSalary]
   })

geo_encoded=encoder_geo.transform([[geography]]).toarray()
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



