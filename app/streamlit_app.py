'''import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import time
import sys
import os
sys.path.append(os.getcwd())

st.set_page_config(page_title="Bhutan Suicide Rate Predictor", layout="wide")
st.title("Bhutan Suicide Rates: Full Analysis & Prediction")
st.markdown("**Male • Female • Both Sexes | 2000–2021 | Forecast to 2030**")

page = st.sidebar.radio("Go to", [
    "0. View Raw Data",
    "1. View Cleaned Data",
    "2. Suicide Rates by Sex (Table View)",
    "3. All Visualizations",
    "4. Preprocessing & Features",
    "5. Train ML Model (See Accuracy Here!)",
    "6. Forecast & Predict Any Year"
])

@st.cache_data
def load_raw():
    return pd.read_csv("data/raw/mental_health_indicators_btn.csv", skiprows=1)

@st.cache_data
def load_clean():
    clean = pd.read_csv("data/processed/bhutan_mental_health_clean.csv")
    prep = pd.read_csv("data/processed/bhutan_suicide_preprocessed.csv")
    return clean, prep

raw_df = load_raw()
clean_df, prep_df = load_clean()
suicide = clean_df[clean_df["GHO_CODE"] == "MH_12"][["Year", "DIM_NAME", "Numeric"]].dropna()

if page == "5. Train ML Model (See Accuracy Here!)":
    st.header("Step 5: Train Machine Learning Models")
    
    if st.button("TRAIN MODELS NOW", type="primary", use_container_width=True):
        with st.spinner("Training Male & Female models..."):
            start_time = time.time()
            
            from src.train_model import train_models
            result = train_models()  # Returns dictionary with metrics
            
            training_time = time.time() - start_time
        
        
        # BEAUTIFUL ACCURACY DISPLAY
        st.subheader("Model Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Used", "Random Forest Regressor")
        col3.metric("Cost Function", "MSE")
        col4.metric("Training Time", f"{training_time:.2f} sec")
        
        st.markdown("### Accuracy Scores")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Model R² Score", f"{result['r2_male']:.4f}" )
            st.metric("Male Model MAE", f"{result['mae_male']:.3f} per 100k")
        with col2:
            st.metric("Female Model R² Score", f"{result['r2_female']:.4f}")
            st.metric("Female Model MAE", f"{result['mae_female']:.3f} per 100k")
    

elif page == "0. View Raw Data":
    st.header("Step 0: Original Raw Dataset")
    st.dataframe(raw_df, use_container_width=True)

elif page == "1. View Cleaned Data":
    st.header("Step 1: Cleaned & Validated Dataset")
    full_clean = clean_df[clean_df["GHO_CODE"] == "MH_12"].sort_values("Year")
    st.subheader("Full Cleaned Table")
    st.dataframe(full_clean, use_container_width=True)
    st.markdown("---")
    st.subheader("Separated by Gender")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Female Only**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Female"][["Year", "Numeric"]])
    with col2:
        st.markdown("**Male Only**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Male"][["Year", "Numeric"]])
    with col3:
        st.markdown("**Both Sexes**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Both sexes"][["Year", "Numeric"]])

elif page == "2. Suicide Rates by Sex (Table View)":
    st.header("Suicide Rates by Gender (2000–2021) — Table Format")
    table = suicide.pivot(index="Year", columns="DIM_NAME", values="Numeric").round(2)
    table.columns.name = None
    table = table.reset_index()[["Year", "Male", "Female", "Both sexes"]].sort_values("Year")
    st.dataframe(table, use_container_width=True, height=600)

elif page == "3. All Visualizations":
    st.header("All Charts")
    tab1, tab2 = st.tabs(["Trend & Bar", "Scatter & Gender Gap"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.line(suicide, x="Year", y="Numeric", color="DIM_NAME", markers=True))
        with col2: st.plotly_chart(px.bar(suicide, x="Year", y="Numeric", color="DIM_NAME", barmode="group"))
    with tab2:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.scatter(suicide, x="Year", y="Numeric", color="DIM_NAME", size="Numeric"))
        with col2:
            prep_df["Gap"] = prep_df["Male"] - prep_df["Female"]
            st.plotly_chart(px.area(prep_df, x="Year", y="Gap"))

elif page == "4. Preprocessing & Features":
    st.header("Feature Engineering")
    st.dataframe(prep_df)

elif page == "6. Forecast & Predict Any Year":
    st.header("Future Forecast & Custom Prediction")
    try:
        model_male = joblib.load("models/male_suicide_model.pkl")
        model_female = joblib.load("models/female_suicide_model.pkl")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Forecast 2024–2030")
            years = list(range(2024, 2031))
            future = pd.DataFrame({"Year": years, "Year_since_2000": [y-2000 for y in years], "Year_squared": [y**2 for y in years],
                                   "Male_Lag1": [prep_df["Male"].iloc[-1]]*7, "Female_Lag1": [prep_df["Female"].iloc[-1]]*7,
                                   "Gender_Gap": [prep_df["Gender_Gap"].iloc[-1]]*7})
            pred_m = model_male.predict(future[["Year","Year_since_2000","Year_squared","Male_Lag1","Female_Lag1","Gender_Gap"]])
            pred_f = model_female.predict(future[["Year","Year_since_2000","Year_squared","Male_Lag1","Female_Lag1","Gender_Gap"]])
            forecast = pd.DataFrame({"Year": years, "Male": np.round(pred_m,2), "Female": np.round(pred_f,2)})
            st.dataframe(forecast)
            st.plotly_chart(px.line(forecast.melt(id_vars="Year"), x="Year", y="value", color="variable"))
        with col2:
            st.subheader("Predict Any Year")
            year = st.number_input("Year", 2024, 2100, 2030)
            gender = st.selectbox("Gender", ["Male", "Female"])
            if st.button("PREDICT"):
                model = model_male if gender == "Male" else model_female
                X = pd.DataFrame([{"Year": year, "Year_since_2000": year-2000, "Year_squared": year**2,
                                   "Male_Lag1": prep_df["Male"].iloc[-1], "Female_Lag1": prep_df["Female"].iloc[-1],
                                   "Gender_Gap": prep_df["Gender_Gap"].iloc[-1]}])
                rate = model.predict(X)[0]
                st.metric(f"{gender} Rate in {year}", f"{rate:.2f} per 100k")
    except:
        st.error("Train models first!")

st.sidebar.success("FULL ACCURACY NOW SHOWS ON SCREEN!")
'''
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import time
import sys
import os
sys.path.append(os.getcwd())

st.set_page_config(page_title="Bhutan Suicide Rate Predictor", layout="wide")
st.title("Bhutan Suicide Rates: Full Analysis & Prediction")
st.markdown("**Male • Female • Both Sexes | 2000–2021 | Forecast to 2030**")

page = st.sidebar.radio("Go to", [
    "0. View Raw Data",
    "1. View Cleaned Data",
    "2. Suicide Rates by Sex (Table View)",
    "3. All Visualizations",
    "4. Preprocessing & Features",
    "5. Train ML Model (See Accuracy Here!)",
    "6. Forecast & Predict Any Year"
])

@st.cache_data
def load_raw():
    return pd.read_csv("data/raw/mental_health_indicators_btn.csv", skiprows=1)

@st.cache_data
def load_clean():
    clean = pd.read_csv("data/processed/bhutan_mental_health_clean.csv")
    prep = pd.read_csv("data/processed/bhutan_suicide_preprocessed.csv")
    return clean, prep

raw_df = load_raw()
clean_df, prep_df = load_clean()
suicide = clean_df[clean_df["GHO_CODE"] == "MH_12"][["Year", "DIM_NAME", "Numeric"]].dropna()

# TRAIN MODEL PAGE — NOW FULLY WORKING WITH ACCURACY!
if page == "5. Train ML Model (See Accuracy Here!)":
    st.header("Step 5: Train Machine Learning Models")
    
    if st.button("TRAIN MODELS NOW", type="primary", use_container_width=True):
        with st.spinner("Training Male & Female models..."):
            start_time = time.time()
            from src.train_model import train_models
            result = train_models()
            training_time = time.time() - start_time
        
       
        # Beautiful accuracy display
        st.subheader("Model Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Used", "Random Forest Regressor")
        col2.metric("Number of Trees", "500")
        col3.metric("Cost Function", "Mean Squared Error (MSE)")
        col4.metric("Training Time", f"{training_time:.2f} seconds")
        
        st.markdown("### Accuracy Scores")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Model R² Score", f"{result['r2_male']:.4f}")
            st.metric("Male Model MAE", f"{result['mae_male']:.3f} per 100k")
        with col2:
            st.metric("Female Model R² Score", f"{result['r2_female']:.4f}")
            st.metric("Female Model MAE", f"{result['mae_female']:.3f} per 100k")
        
       
# Other pages (perfect as you wrote them)
elif page == "0. View Raw Data":
    st.header("Step 0: Original Raw Dataset")
    st.dataframe(raw_df, use_container_width=True)

elif page == "1. View Cleaned Data":
    st.header("Step 1: Cleaned & Validated Dataset")
    full_clean = clean_df[clean_df["GHO_CODE"] == "MH_12"].sort_values("Year")
    st.subheader("Full Cleaned Table")
    st.dataframe(full_clean, use_container_width=True)
    st.markdown("---")
    st.subheader("Separated by Gender")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Female Only**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Female"][["Year", "Numeric"]])
    with col2:
        st.markdown("**Male Only**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Male"][["Year", "Numeric"]])
    with col3:
        st.markdown("**Both Sexes**")
        st.dataframe(full_clean[full_clean["DIM_NAME"] == "Both sexes"][["Year", "Numeric"]])

elif page == "2. Suicide Rates by Sex (Table View)":
    st.header("Suicide Rates by Gender (2000–2021) — Table Format")
    table = suicide.pivot(index="Year", columns="DIM_NAME", values="Numeric").round(2)
    table.columns.name = None
    table = table.reset_index()[["Year", "Male", "Female", "Both sexes"]].sort_values("Year")
    st.dataframe(table, use_container_width=True, height=600)

elif page == "3. All Visualizations":
    st.header("All Charts")
    tab1, tab2 = st.tabs(["Trend & Bar", "Scatter & Gender Gap"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.line(suicide, x="Year", y="Numeric", color="DIM_NAME", markers=True))
        with col2: st.plotly_chart(px.bar(suicide, x="Year", y="Numeric", color="DIM_NAME", barmode="group"))
    with tab2:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.scatter(suicide, x="Year", y="Numeric", color="DIM_NAME", size="Numeric"))
        with col2:
            prep_df["Gap"] = prep_df["Male"] - prep_df["Female"]
            st.plotly_chart(px.area(prep_df, x="Year", y="Gap"))

elif page == "4. Preprocessing & Features":
    st.header("Feature Engineering")
    st.dataframe(prep_df)

elif page == "6. Forecast & Predict Any Year":
    st.header("Future Forecast & Custom Prediction")
    try:
        model_male = joblib.load("models/male_suicide_model.pkl")
        model_female = joblib.load("models/female_suicide_model.pkl")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Forecast 2024–2030")
            years = list(range(2024, 2031))
            future = pd.DataFrame({"Year": years, "Year_since_2000": [y-2000 for y in years], "Year_squared": [y**2 for y in years],
                                   "Male_Lag1": [prep_df["Male"].iloc[-1]]*7, "Female_Lag1": [prep_df["Female"].iloc[-1]]*7,
                                   "Gender_Gap": [prep_df["Gender_Gap"].iloc[-1]]*7})
            pred_m = model_male.predict(future[["Year","Year_since_2000","Year_squared","Male_Lag1","Female_Lag1","Gender_Gap"]])
            pred_f = model_female.predict(future[["Year","Year_since_2000","Year_squared","Male_Lag1","Female_Lag1","Gender_Gap"]])
            forecast = pd.DataFrame({"Year": years, "Male": np.round(pred_m,2), "Female": np.round(pred_f,2)})
            st.dataframe(forecast)
            st.plotly_chart(px.line(forecast.melt(id_vars="Year"), x="Year", y="value", color="variable"))
        with col2:
            st.subheader("Predict Any Year")
            year = st.number_input("Year", 2024, 2100, 2030)
            gender = st.selectbox("Gender", ["Male", "Female"])
            if st.button("PREDICT"):
                model = model_male if gender == "Male" else model_female
                X = pd.DataFrame([{"Year": year, "Year_since_2000": year-2000, "Year_squared": year**2,
                                   "Male_Lag1": prep_df["Male"].iloc[-1], "Female_Lag1": prep_df["Female"].iloc[-1],
                                   "Gender_Gap": prep_df["Gender_Gap"].iloc[-1]}])
                rate = model.predict(X)[0]
                st.metric(f"{gender} Rate in {year}", f"{rate:.2f} per 100k")
    except:
        st.error("Train models first in Step 5!")

