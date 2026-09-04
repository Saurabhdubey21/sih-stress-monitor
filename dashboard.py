
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Personnel Welfare Dashboard", page_icon="shield", layout="wide")

@st.cache_resource
def load_model():
    with open("stress_model.pkl","rb") as f:
        return pickle.load(f)

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

saved = load_model()
MODEL = saved["model"]; FEATS = saved["features"]; df = load_data()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=70)
st.sidebar.title("Welfare Monitor")
st.sidebar.selectbox("Login as",["Welfare Officer","Commander","Medical Officer"])
st.sidebar.markdown("---")
st.sidebar.info("All data is anonymized. Individual identities are protected.")

st.title("Personnel Stress & Welfare Monitoring")
st.caption("AI-powered early warning system for CAPF personnel welfare")

rc = df["risk_level"].value_counts()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Personnel", len(df))
c2.metric("High Risk", rc.get("High",0), str(round(rc.get("High",0)/len(df)*100,1))+"%", delta_color="inverse")
c3.metric("Medium Risk", rc.get("Medium",0), str(round(rc.get("Medium",0)/len(df)*100,1))+"%", delta_color="inverse")
c4.metric("Low Risk", rc.get("Low",0), str(round(rc.get("Low",0)/len(df)*100,1))+"%")

st.markdown("---")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Risk Distribution")
    fig1 = px.pie(values=rc.values, names=rc.index, color=rc.index,
                  color_discrete_map={"High":"#e74c3c","Medium":"#f59e0b","Low":"#27ae60"}, hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("Deployment Duration by Rank")
    fig2 = px.box(df, x="rank", y="deployment_months", color="risk_level",
                  color_discrete_map={"High":"#e74c3c","Medium":"#f59e0b","Low":"#27ae60"})
    fig2.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("High-Risk Personnel — Immediate Attention Required")
high_df = df[df["risk_level"]=="High"].sort_values("deployment_months",ascending=False)
cols = ["personnel_id","rank","age","deployment_months","duty_hours_per_day","night_shifts_per_month","incidents_exposed"]
st.dataframe(high_df[cols].head(20).style.background_gradient(subset=["deployment_months"],cmap="Reds"), use_container_width=True)

st.markdown("---")
st.subheader("Individual Risk Assessment Tool")
with st.form("f"):
    a1,a2,a3 = st.columns(3)
    with a1:
        age = st.slider("Age",20,58,30)
        yrs = st.slider("Years of Service",1,30,8)
        dep = st.slider("Deployment Months",1,36,18)
        fam = st.radio("Family Separated?",[0,1],format_func=lambda x:"Yes" if x else "No")
    with a2:
        duty = st.slider("Duty Hours/Day",8.0,16.0,10.0,0.5)
        lva = st.slider("Leaves Availed",0,30,10)
        lvc = st.slider("Leaves Cancelled",0,30,2)
        trans = st.slider("Transfers 2yr",0,5,1)
    with a3:
        nights = st.slider("Night Shifts/Month",0,20,5)
        inc = st.slider("Incidents Exposed",0,10,2)
        sleep = st.slider("Sleep Hours",4.0,9.0,6.5,0.5)
        ex = st.slider("Exercise Days/Week",0,7,3)
        well = st.slider("Wellness Score",1.0,10.0,6.0,0.5)
        soc = st.slider("Social Support",1.0,10.0,5.0,0.5)
        rank_e = st.selectbox("Rank",[0,1,2,3,4,5,6],
                    format_func=lambda x:["Constable","Head Constable","ASI","SI","Inspector","DSP","SP"][x])
    go = st.form_submit_button("Predict Risk", type="primary")

if go:
    overwork = max(0,duty-8)
    inp = {
        "age":age,"years_of_service":yrs,"deployment_months":dep,
        "family_separation":fam,"leave_cancel_ratio":lvc/max(lva,1),
        "overwork_score":overwork,"transfers_last_2yr":trans,
        "night_shifts_per_month":nights,"incidents_exposed":inc,
        "training_days_yr":15,"wellness_score":well,"sleep_hours":sleep,
        "exercise_freq_per_wk":ex,"social_support_score":soc,"rank_encoded":rank_e,
        "burnout_index":(dep*overwork)/10,"recovery_index":(sleep*ex)/7,
        "isolation_score":fam*(1-soc/10),"workload_stress":nights*overwork,
        "resilience_score":well+soc+ex
    }
    row = pd.DataFrame([inp])[FEATS]
    pred = MODEL.predict(row)[0]
    proba = MODEL.predict_proba(row)[0]
    pd_ = dict(zip(MODEL.classes_,[round(p*100,1) for p in proba]))
    score = pd_.get("High",0)+0.5*pd_.get("Medium",0)
    color = {"High":"red","Medium":"orange","Low":"green"}
    st.markdown(f"### Risk: :{color[pred]}[{pred}] | Score: {score:.1f}/100")
    r1,r2 = st.columns(2)
    with r1:
        gauge = go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Stress Score"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"darkblue"},
                   "steps":[{"range":[0,40],"color":"#27ae60"},{"range":[40,70],"color":"#f59e0b"},{"range":[70,100],"color":"#e74c3c"}]}))
        st.plotly_chart(gauge,use_container_width=True)
    with r2:
        st.markdown("### Welfare Recommendations")
        recs = {
            "High":["Immediate counseling within 48 hours","Review deployment rotation","Mandatory 2-week rest","Family reunification support","Medical checkup required"],
            "Medium":["Wellness check-in within 2 weeks","Reduce night shift frequency","Encourage leave utilization","Peer support group referral"],
            "Low":["Continue regular monitoring","Maintain wellness routine","Next check-in in 30 days"]
        }
        icons = {"High":"\U0001F534","Medium":"\U0001F7E1","Low":"\U0001F7E2"}
        for r in recs[pred]:
            st.markdown(icons[pred]+" "+r)

st.markdown("---")
st.caption("SIH 2025 | AI Personnel Welfare Monitoring | Anonymized & Encrypted")
