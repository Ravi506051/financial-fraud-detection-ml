import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,confusion_matrix
st.title("Financial Fraud Detection Lab"); st.warning("Synthetic educational demo, not a production fraud system.")
r=np.random.default_rng(7); n=6000
X=pd.DataFrame({"amount":r.lognormal(3,1,n),"hour":r.integers(0,24,n),"distance_km":r.exponential(20,n),"velocity":r.poisson(2,n),"new_device":r.integers(0,2,n)})
p=1/(1+np.exp(-(-5+.0015*X.amount+.09*(X.hour<5)+.025*X.distance_km+.45*X.velocity+.9*X.new_device)))
y=r.binomial(1,p); a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
m=HistGradientBoostingClassifier(random_state=42).fit(a,c); score=m.predict_proba(b)[:,1]
t=st.sidebar.slider("Alert threshold",.05,.95,.45); pred=(score>=t).astype(int)
x1,x2=st.columns(2); x1.metric("Fraud rate",f"{y.mean():.2%}"); x2.metric("ROC-AUC",f"{roc_auc_score(d,score):.3f}")
cm=confusion_matrix(d,pred); st.plotly_chart(px.imshow(cm,text_auto=True,title="Confusion matrix"),use_container_width=True)
st.subheader("Score a transaction"); row=pd.DataFrame([{"amount":st.number_input("Amount",10.,50000.,250.),"hour":st.slider("Hour",0,23,22),"distance_km":st.number_input("Distance km",0.,1000.,35.),"velocity":st.slider("Transactions last hour",0,30,4),"new_device":int(st.checkbox("New device"))}])
st.metric("Estimated risk",f"{m.predict_proba(row)[0,1]:.1%}")
