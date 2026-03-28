"""
ml_engine.py — WesternWear Analytics
All machine-learning model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data_engine import get_model_features


@st.cache_resource(show_spinner=False)
def train_all(df: pd.DataFrame):
    """Train all classifiers, KMeans clusters, and return everything needed for dashboard."""
    features = get_model_features(df)
    # Keep only features that exist in df
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    classifiers = {
        'Decision Tree':     DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        'Random Forest':     RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        'Logistic Reg.':     LogisticRegression(max_iter=600, random_state=42, multi_class='auto'),
        'SVM':               SVC(probability=True, random_state=42, C=1.0),
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        use_scaled = name in ['Logistic Reg.', 'SVM']
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test

        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        proba = clf.predict_proba(Xte)

        # ROC-AUC (one-vs-rest for multiclass)
        try:
            auc = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')
        except Exception:
            auc = 0.0

        cv_scores = cross_val_score(clf, Xtr, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

        results[name] = dict(
            model  = clf,
            preds  = preds,
            proba  = proba,
            acc    = accuracy_score(y_test, preds),
            f1     = f1_score(y_test, preds, average='weighted'),
            auc    = auc,
            cv_mean= cv_scores.mean(),
            cv_std = cv_scores.std(),
            cm     = confusion_matrix(y_test, preds),
            report = classification_report(y_test, preds,
                                           target_names=['Not Interested','Neutral','Interested'],
                                           output_dict=True),
        )

    # Best model by AUC
    best_name = max(results, key=lambda n: results[n]['auc'])

    # Clustering (KMeans on scaled features, k=5 to match personas)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sc.transform(X))   # full dataset

    # PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(sc.transform(X))

    return dict(
        results    = results,
        best_name  = best_name,
        features   = features,
        X_test     = X_test,
        y_test     = y_test,
        scaler     = sc,
        kmeans     = kmeans,
        cluster_labels = cluster_labels,
        X_pca      = X_pca,
        X_full     = X,
        pca        = pca,
    )


def predict_new(df_new: pd.DataFrame, ml: dict) -> pd.DataFrame:
    """Run prediction on a new uploaded dataset."""
    features = [f for f in ml['features'] if f in df_new.columns]
    X = df_new[features].fillna(0)
    X_sc = ml['scaler'].transform(X)

    best = ml['results'][ml['best_name']]
    use_scaled = ml['best_name'] in ['Logistic Reg.', 'SVM']
    Xinput = X_sc if use_scaled else X

    proba = best['model'].predict_proba(Xinput)
    preds = best['model'].predict(Xinput)

    out = df_new.copy() if 'Respondent_ID' not in df_new.columns else df_new[['Respondent_ID']].copy()
    out['Predicted_Label'] = pd.Series(preds).map({0:'Not Interested',1:'Neutral',2:'Interested'}).values
    out['Prob_NotInterested'] = (proba[:, 0] * 100).round(1)
    out['Prob_Neutral']       = (proba[:, 1] * 100).round(1)
    out['Prob_Interested']    = (proba[:, 2] * 100).round(1)
    out['Confidence_%']       = (np.max(proba, axis=1) * 100).round(1)
    return out
