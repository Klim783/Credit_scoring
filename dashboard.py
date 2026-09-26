import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Credit Scoring Dashboard", page_icon="💳", layout="wide")

st.title("💳 Home Credit Risk Assessment Dashboard")
st.write("Interactively evaluate client risk profiles using the trained LightGBM pipeline.")

# Sidebar Controls
st.sidebar.header("Client Financial Profile")

income = st.sidebar.number_input("Total Income ($)", value=150000.0, step=5000.0)
credit = st.sidebar.number_input("Requested Credit ($)", value=500000.0, step=10000.0)
annuity = st.sidebar.number_input("Annuity ($)", value=25000.0, step=1000.0)
goods_price = st.sidebar.number_input("Goods Price ($)", value=450000.0, step=10000.0)

gender = st.sidebar.selectbox("Gender", ["M", "F"])
education = st.sidebar.selectbox("Education",
								 ["Secondary / secondary special", "Higher education", "Incomplete higher"])
housing = st.sidebar.selectbox("Housing Type", ["House / apartment", "With parents", "Rented apartment"])

threshold = st.sidebar.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

# Build Payload
payload = {
	"NAME_CONTRACT_TYPE": "Cash loans",
	"CODE_GENDER": gender,
	"FLAG_OWN_CAR": "Y",
	"FLAG_OWN_REALTY": "Y",
	"CNT_CHILDREN": 0,
	"AMT_INCOME_TOTAL": income,
	"AMT_CREDIT": credit,
	"AMT_ANNUITY": annuity,
	"AMT_GOODS_PRICE": goods_price,
	"NAME_TYPE_SUITE": "Unaccompanied",
	"NAME_INCOME_TYPE": "Working",
	"NAME_EDUCATION_TYPE": education,
	"NAME_FAMILY_STATUS": "Married",
	"NAME_HOUSING_TYPE": housing,
	"DAYS_BIRTH": -12000,
	"DAYS_EMPLOYED": -2000,
	"DAYS_REGISTRATION": -4000,
	"DAYS_ID_PUBLISH": -1500,
	"EXT_SOURCE_1": 0.5,
	"EXT_SOURCE_2": 0.6,
	"EXT_SOURCE_3": 0.7
}

if st.button("Evaluate Credit Application", type="primary"):
	try:
		response = requests.post(f"http://127.0.0.1:8000/predict?threshold={threshold}", json=payload)

		if response.status_code == 200:
			result = response.json()
			prob = result["default_probability"]
			decision = result["decision"]

			col1, col2, col3 = st.columns(3)

			with col1:
				st.metric(label="Default Probability", value=f"{prob * 100:.1f}%")
			with col2:
				st.metric(label="Decision Threshold", value=f"{threshold * 100:.1f}%")
			with col3:
				if decision == "rejected":
					st.error("🚫 DECISION: REJECTED")
				else:
					st.success("✅ DECISION: ACCEPTED")

			# Progress Bar for Risk Level
			st.write("### Risk Score Progress")
			st.progress(prob)

		else:
			st.error(f"API Error: {response.status_code}")
	except Exception as e:
		st.error(f"Failed to connect to FastAPI service: {e}")