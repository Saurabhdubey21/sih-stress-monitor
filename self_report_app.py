
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Personnel Self-Check", page_icon="shield", layout="centered")

@st.cache_resource
def load_model():
    with open("stress_model.pkl", "rb") as f:
        return pickle.load(f)

saved = load_model()
MODEL = saved["model"]
FEATS = saved["features"]

st.title("Personnel Wellness Self-Check")
st.caption("Confidential - Anonymous - Takes 2 minutes")
st.markdown("---")

with st.form("self_report"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Your age", 18, 60, 30)
        rank = st.selectbox("Rank", ["Constable","Head Constable","ASI","SI","Inspector","DSP","SP"])
        yrs = st.number_input("Years of service", 0, 40, 5)
    with col2:
        dep = st.number_input("Current deployment (months)", 0, 48, 6)
        fam_sep = st.radio("Separated from family?", ["No","Yes"])
        transfers = st.number_input("Transfers in last 2 years", 0, 10, 1)

    col3, col4 = st.columns(2)
    with col3:
        duty_hours = st.slider("Avg duty hours/day", 6.0, 18.0, 10.0, 0.5)
        nights = st.slider("Night shifts/month", 0, 25, 5)
        incidents = st.slider("Stressful incidents (6 months)", 0, 10, 1)
    with col4:
        leaves_availed = st.number_input("Leaves availed this year", 0, 40, 10)
        leaves_cancelled = st.number_input("Leaves cancelled this year", 0, 40, 2)
        training_days = st.number_input("Training days this year", 0, 60, 15)

    col5, col6 = st.columns(2)
    with col5:
        sleep_hours = st.slider("Avg sleep/night (hrs)", 3.0, 10.0, 6.5, 0.5)
        exercise = st.slider("Exercise days/week", 0, 7, 3)
    with col6:
        wellness = st.slider("Overall wellness (1-10)", 1.0, 10.0, 6.0, 0.5)
        social = st.slider("Social support (1-10)", 1.0, 10.0, 5.0, 0.5)

    consent = st.checkbox("I consent to an anonymized risk flag being shared if needed.")
    submitted = st.form_submit_button("Check My Wellness Score", type="primary")

if submitted:
    if not consent:
        st.warning("Please confirm consent to continue.")
    else:
        rank_map = {"Constable":0,"Head Constable":1,"ASI":2,"SI":3,"Inspector":4,"DSP":5,"SP":6}
        rank_encoded = rank_map[rank]
        fam = 1 if fam_sep=="Yes" else 0
        overwork = max(0, duty_hours-8)
        lv_ratio = leaves_cancelled/max(leaves_availed,1)
        burnout = (dep*overwork)/10
        recovery = (sleep_hours*exercise)/7
        isolation = fam*(1-social/10)
        wk_stress = nights*overwork
        resilience = wellness+social+exercise

        inp = {"age":age,"years_of_service":yrs,"deployment_months":dep,
            "family_separation":fam,"leave_cancel_ratio":lv_ratio,"overwork_score":overwork,
            "transfers_last_2yr":transfers,"night_shifts_per_month":nights,"incidents_exposed":incidents,
            "training_days_yr":training_days,"wellness_score":wellness,"sleep_hours":sleep_hours,
            "exercise_freq_per_wk":exercise,"social_support_score":social,"rank_encoded":rank_encoded,
            "burnout_index":burnout,"recovery_index":recovery,"isolation_score":isolation,
            "workload_stress":wk_stress,"resilience_score":resilience}

        row = pd.DataFrame([inp])[FEATS]
        pred = MODEL.predict(row)[0]
        proba = MODEL.predict_proba(row)[0]
        conf = dict(zip(MODEL.classes_, [round(p*100,1) for p in proba]))

        st.markdown("---")
        color = {"High":"red","Medium":"orange","Low":"green"}
        st.markdown(f"## Result: :{color[pred]}[{pred} Stress Risk]")
        messages = {
            "Low":"You are doing well. Keep maintaining your routine.",
            "Medium":"You may be under moderate strain. Consider speaking to a peer or counselor.",
            "High":"Significant strain detected. Please speak with a welfare/medical officer soon."
        }
        st.info(messages[pred])
        with st.expander("Confidence breakdown"):
            st.write(conf)

st.markdown("---")
st.caption("SIH 2025 | Confidential Personnel Wellness Self-Check")
