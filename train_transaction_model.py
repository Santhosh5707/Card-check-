from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

ROOT=Path(__file__).parent; DATA=ROOT/'data'/'transactions.csv'; OUT=ROOT/'models'; OUT.mkdir(exist_ok=True)
if not DATA.exists(): raise SystemExit("Missing data/transactions.csv. Add an authorized labeled dataset with target column 'Class'.")
df=pd.read_csv(DATA)
if 'Class' not in df.columns: raise SystemExit("Dataset must have binary target column 'Class' (1=fraud, 0=legitimate).")
df=df.dropna(subset=['Class']).copy(); y=pd.to_numeric(df.pop('Class'),errors='coerce'); valid=y.isin([0,1]); df,y=df.loc[valid],y.loc[valid].astype(int)
if y.nunique()!=2 or y.value_counts().min()<2: raise SystemExit('Need at least two examples of each class.')
X=df.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
if X.shape[1]==0: raise SystemExit('No numeric feature columns found.')
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scaler',StandardScaler()),('classifier',LogisticRegression(max_iter=2000,class_weight='balanced'))])
model.fit(Xtr,ytr); probs=model.predict_proba(Xte)[:,1]; preds=(probs>=.5).astype(int)
metrics={'rows_total':int(len(df)),'fraud_rows':int((y==1).sum()),'legitimate_rows':int((y==0).sum()),'features':list(X.columns),'roc_auc':float(roc_auc_score(yte,probs)),'average_precision':float(average_precision_score(yte,probs)),'threshold':.5,'classification_report':classification_report(yte,preds,output_dict=True,zero_division=0),'note':'Holdout metrics only; not a guarantee of live performance.'}
joblib.dump(model,OUT/'transaction_model.joblib'); (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps({k:v for k,v in metrics.items() if k!='classification_report'},indent=2)); print('Saved models/transaction_model.joblib')
