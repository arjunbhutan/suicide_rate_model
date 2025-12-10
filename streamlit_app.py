import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Bhutan Clean Cooking Forecast", layout="wide")
st.sidebar.title("Data Science Workflow")
page = st.sidebar.radio("Go to", [
    "1. View Dataset",
    "2. Explore & Visualize", 
    "3. Clean & Preprocess",
    "4. Train Model",
    "5. Forecast 2024–2035"
])

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/bhutan_clean.csv")

df = load_data()

if page == "1. View Dataset":
    st.header("1. Raw WHO Dataset – Bhutan")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(200))
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "bhutan_who_data.csv")

if page == "2. Explore & Visualize":
    st.header("2. Exploratory Data Analysis")
    ind = st.selectbox("Select Indicator", sorted(df["Indicator_Name"].unique()))
    dim = st.multiselect("Dimension", df["Dimension_Name"].unique(), default=["Total"])
    filtered = df[(df["Indicator_Name"] == ind) & (df["Dimension_Name"].isin(dim))]
    fig = px.line(filtered, x="Year", y="Numeric", color="Dimension_Name", title=ind, markers=True)
    st.plotly_chart(fig, use_container_width=True)

if page == "3. Clean & Preprocess":
    st.header("3. Clean & Preprocess")
    st.success("Already cleaned: removed header row, fixed names, filtered Bhutan")
    target = df[(df["Indicator_Code"] == "PHE_HHAIR_PROP_POP_POLLUTING_FUELS") & (df["Dimension_Name"] == "Total")].copy()
    target = target[["Year","Numeric"]].sort_values("Year").dropna()
    target["Year_since_1990"] = target["Year - 1990
    target["Year_squared"] = target["Year"] ** 2
    st.dataframe(target)

if page == "4. Train Model":
    st.header("4. Train Forecasting Model")
    data = df[(df["Indicator_Code"] == "PHE_HHAIR_PROP_POP_POLLUTING_FUELS") & (df["Dimension_Name"] == "Total")].copy()
    data = data[["Year","Numeric"]].sort_values("Year").dropna()
    data["Year_since_1990"] = data["Year"] - 1990
    data["Year_squared"] = data["Year"] ** 2
    X = data[["Year","Year_since_1990","Year_squared"]]
    y = data["Numeric"]
    train = data[data["Year"] < 2020]
    test  = data[data["Year"] >= 2020]
    if st.button("Train Random Forest Model", type="primary"):
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(train[["Year","Year_since_1990","Year_squared"]], train["Numeric"])
        pred = model.predict(test[["Year","Year_since_1990","Year_squared"]])
        joblib.dump(model, "models/trained_model.pkl")
        st.success("Model trained & saved!")
        c1, c2 = st.columns(2)
        c1.metric("Test MAE", f"{mean_absolute_error(test['Numeric'], pred):.2f}%")
        c2.metric("Test R²", f"{r2_score(test['Numeric'], pred):.3f}")

if page == "5. Forecast 2024–2035":
    st.header("5. Future Forecast")
    st.balloons()
    try:
        model = joblib.load("models/trained_model.pkl")
        years = list(range(2024, 2036))
        future = pd.DataFrame({"Year": years, "Year_since_1990": [y-1990 for y in years], "Year_squared": [y**2 for y in years]})
        pred = model.predict(future)
        forecast = pd.DataFrame({"Year": years, "Predicted % Polluting Fuels": np.round(pred,2)})
        st.dataframe(forecast)
        fig = px.line(forecast, x="Year", y="Predicted % Polluting Fuels", markers=True, title="Bhutan Clean Cooking Forecast")
        fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="SDG Target")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Bhutan will reach the SDG clean cooking target before 2035!")
    except:
        st.error("Train the model first in step 4!")