# importing dataset and packages
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# loading the dataset
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    # adding a new column species to dataframe
    df['species']=iris.target
    return df,iris.target_names

df,target_names=load_data()


# defining the model
model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

# gathering input from user using streamlit
sepal_len=st.sidebar.slider("Sepal Length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_wid=st.sidebar.slider("Sepal Width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_len=st.sidebar.slider("Petal Length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_wid=st.sidebar.slider("Petal Width",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

# storing user input as list
features=[[sepal_len,sepal_wid,petal_len,petal_wid]]

# using model to generate a prediction
prediction=model.predict(features)
predicted_species=target_names[prediction[0]]

# output of the predictin
st.write("Prediction")
st.write(f"Predicted Species = {predicted_species}")