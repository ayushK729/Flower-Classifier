# importing dataset and packages
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Flower Classifier",
    page_icon="res/ico.png"
    )

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


# adding conditional flower information
if predicted_species=="setosa":
    st.write(f"Setosa is one of the three species of the Iris flower and is the smallest among them. It is characterized by its relatively small sepal length, sepal width, petal length, and petal width. The flowers of Setosa are typically found to have shorter petals and broader sepals compared to the other species. Setosa is often the easiest to distinguish due to its distinct and compact appearance. The flowers typically have a striking blue or purple color and a pleasant scent. It is known for its resilience in varying growing conditions and has a high adaptability rate.")
elif predicted_species=="virginica":
    st.write(f"Virginica is the largest of the three Iris species and is known for its long and narrow petals. The flower has a prominent and vibrant color, typically a deep violet or blue, and it has a distinct structure with a pronounced difference between the length of its petals and sepals. Virginica flowers are commonly found in wetlands, particularly in areas along the eastern United States. These flowers tend to have a more elongated and slender shape compared to Setosa and Versicolor. Known for its hardiness, Virginica is often seen in natural landscapes and is valued for its aesthetic appeal in gardens. It requires more care and attention when cultivated in non-native habitats.")
elif predicted_species=="versicolor":
    st.write(f"Versicolor is a hybrid species that exhibits characteristics between Setosa and Virginica. Its flowers typically have a medium sepal length and width, and its petals are larger than Setosa's but smaller than Virginica's. Versicolor’s petals are a mix of blue and purple hues, giving it a unique appearance. It tends to be less compact than Setosa, and its blooms are a bit more elongated. The plant thrives in various soil types and is widely cultivated for ornamental purposes. Versicolor is often described as a “middle ground” species due to its intermediate size and characteristics.")