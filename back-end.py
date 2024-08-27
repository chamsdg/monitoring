# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:03:36 2024

@author: caidara01
"""

# import des librairies

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import prometheus_client
from prometheus_client import Gauge, Counter, start_http_server
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

app = FastAPI()

# Charger le modèle, le scaler et les colonnes attendues
model = joblib.load('churn_model_rf.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('model_features.pkl')  # Liste des noms de colonnes

# Prometheus metrics
model_accuracy = Gauge('model_accuracy', 'Accuracy of the churn model')
total_predictions = Counter('total_predictions', 'Total number of predictions made by users')
#positive_predictions = Counter('positive_predictions', 'Total number of positive predictions made by users')
#negative_predictions = Counter('negative_predictions', 'Total number of negative predictions made by users')

# Métriques de la matrice de confusion
tn_metric = Gauge('confusion_matrix_tn', 'True Negatives in the confusion matrix')
fp_metric = Gauge('confusion_matrix_fp', 'False Positives in the confusion matrix')
fn_metric = Gauge('confusion_matrix_fn', 'False Negatives in the confusion matrix')
tp_metric = Gauge('confusion_matrix_tp', 'True Positives in the confusion matrix')

@app.on_event("startup")
async def startup_event():
    # Charger les données pour calculer les métriques
    data = pd.read_csv('Churn_Modelling.csv')
    X = data.drop(['Exited', 'RowNumber', 'Surname', 'Gender', 'Geography'], axis=1)
    y = data['Exited']

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardiser les données
    X_test_scaled = scaler.transform(X_test)  # Utiliser le scaler chargé pour transformer X_test

    # Faire des prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Calculated Accuracy: {accuracy}")

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Mettre à jour les métriques Prometheus
    model_accuracy.set(accuracy)
    tn_metric.set(tn)
    fp_metric.set(fp)
    fn_metric.set(fn)
    tp_metric.set(tp)

@app.get("/")
def read_root():
    return {"Message": "API fonctionne"}

@app.get("/expected_columns/")
def get_expected_columns():
    return JSONResponse(content={"expected_columns": expected_columns.tolist()})

@app.get("/customer_ids/")
def get_customer_ids():
    data = pd.read_csv('Churn_Modelling.csv')
    customer_ids = data['CustomerId'].unique().tolist()
    return JSONResponse(content={"customer_ids": customer_ids})

@app.get("/customer_data/{customer_id}")
def get_customer_data(customer_id: int):
    # Charger les données
    data = pd.read_csv('Churn_Modelling.csv')
    customer_data = data[data['CustomerId'] == customer_id]

    if customer_data.empty:
        return JSONResponse(content={"error": "Customer ID not found"}, status_code=404)

    return JSONResponse(content=customer_data.to_dict(orient='records')[0])


@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    # Charger les données
    data = pd.read_csv('Churn_Modelling.csv')
    customer_data = data[data['CustomerId'] == customer_id]

    if customer_data.empty:
        return JSONResponse(content={"error": "Customer ID not found"}, status_code=404)

    # Préparer les données du client pour la prédiction
    X_customer = customer_data.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')

    # Assurez-vous que X_customer a les colonnes attendues par le modèle
    for col in expected_columns:
        if col not in X_customer.columns:
            X_customer[col] = 0

    # Réindexer X_customer pour correspondre aux colonnes du modèle
    X_customer = X_customer[expected_columns]

    # Standardiser les données du client pour la prédiction
    X_customer_scaled = scaler.transform(X_customer)

    # Prédire la probabilité de churn
    probability_of_churn = model.predict_proba(X_customer_scaled)[:, 1][0] * 100
    prediction = model.predict(X_customer_scaled)[0]

    # Incrémenter les compteurs
    total_predictions.inc()
    #if customer_data['Exited'].values[0] == 1:
        #positive_predictions.inc()
    #else:
        #negative_predictions.inc()

    return {"Customer ID": customer_id, "Churn Probability": probability_of_churn}

if __name__ == "__main__":
    # Démarrer le serveur Prometheus sur le port 8005
    start_http_server(8005)
    import uvicorn
    uvicorn.run(app, host="192.168.179.200", port=8004)  # FastAPI démarre sur le port 8004
