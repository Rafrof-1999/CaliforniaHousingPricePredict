import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# California Houses Prediction 
""")
st.write('---')

california = datasets.fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = pd.DataFrame(california.target, columns=["MedHouseVal"])

print(california)
st.sidebar.header('Manipulate Input Parameters')

def user_input_features():
    Longitude = st.sidebar.slider('Longitude', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))
    Latitude = st.sidebar.slider('Latitude', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
    HouseAge = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
    AveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
    AveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
    Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
    AveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()),float( X.AveOccup.max()), float(X.AveOccup.mean()))
    MedInc = st.sidebar.slider('MedInc', float(X.MedInc.min()),float( X.MedInc.max()), float(X.MedInc.mean()))
    data = {'Longitude': Longitude,
            'Latitude': Latitude,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'MedInc': MedInc,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.header('Specified Input parameters')
st.write(df)
st.write('---')

model = RandomForestRegressor()
model.fit(X, Y)
prediction = model.predict(df)

st.header('Prediction of Median House Value')
st.write(prediction *100000)
st.write('---')

