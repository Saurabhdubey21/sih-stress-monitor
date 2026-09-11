
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Personnel Self-Check", page_icon="shield", layout="centered")

FEATURES = ["age","years_of_service","deployment_months","family_separation",
    "leave_cancel_ratio","overwork_score","transfers_last_2yr","night_shifts_per_month",
    "incidents_exposed","training_days_yr","wellness_score","sleep_hours",
    "exercise_freq_per_wk","social_support_score","rank_encoded","burnout_index",
    "recovery_index","isolation_score","workload_stress","resilience_score"]

@st.cache_data
def load_data():
    df = pd.read_csv("personnel_stress_data.csv")
    df["leave_cancel_ratio"] = df["leaves_cancelled"]/(df["leaves_availed"]+1)
    df["overwork_score"]     = (df["duty_hours_per_day"]-8).clip(lower=0)
    df["rank_encoded"]       = LabelEncoder().fit_transform(df["rank"])
    df["burnout_index"]      = (df["deployment_months"]*df["overwork_score"])/10
    df["recovery_index"]     = (df["sleep_hours"]*df["exercise_freq_per_wk"])/7
    df["isolation_score"]    = df["family_separation"]*(1-df["social_support_score"]/10)
    df["workload_stress"]    = df["night_shifts_per_month"]*df["overwork_score"]
    df["resilience_score"]   = df["wellness_score"]+df["social_support_score"]+df["exercise_freq_per_wk"]
    return df

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["risk_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

df = load_data()
with st.spinner("Loading AI model..."):
    MODEL = train_model(df)
FEATS = FEATURES

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
