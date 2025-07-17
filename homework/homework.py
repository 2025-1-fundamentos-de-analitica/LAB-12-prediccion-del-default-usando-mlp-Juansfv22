# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import os
import gzip
import json
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix, make_scorer
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------
# Función de preprocesamiento de los datos
# ----------------------------------------
def limpiar_dataset(tabla):
    datos = tabla.copy()
    datos.rename(columns={'default payment next month': 'default'}, inplace=True)
    datos.drop(columns='ID', inplace=True)
    datos['EDUCATION'].replace(0, np.nan, inplace=True)
    datos['MARRIAGE'].replace(0, np.nan, inplace=True)
    datos.dropna(inplace=True)
    datos.loc[datos['EDUCATION'] > 4, 'EDUCATION'] = 4
    return datos

# ----------------------------------------
# Separar variables independientes y dependiente
# ----------------------------------------
def separar_variables(dataset, objetivo):
    X = dataset.drop(columns=objetivo)
    y = dataset[objetivo]
    return X, y

# ----------------------------------------
# Crear pipeline de procesamiento y modelo
# ----------------------------------------
def armar_pipeline(dataframe):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_cols = [c for c in dataframe.columns if c not in cat_cols]

    procesador = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), cat_cols)],
        remainder=StandardScaler()
    )

    modelo = Pipeline([
        ('procesador', procesador),
        ('k_best', SelectKBest(f_classif)),
        ('reductor', PCA()),
        ('clasificador', MLPClassifier(max_iter=15000))
    ])

    return modelo

# ----------------------------------------
# Scorer personalizado para validación cruzada
# ----------------------------------------
def mezcla_precision_balance(y_verdad, y_predicho):
    p = precision_score(y_verdad, y_predicho)
    b = balanced_accuracy_score(y_verdad, y_predicho)
    return (p + b) / 2

scorer_personalizado = make_scorer(mezcla_precision_balance, greater_is_better=True)

# ----------------------------------------
# Búsqueda de hiperparámetros
# ----------------------------------------
def buscar_hiperparametros(pipeline, X_train, y_train):
    parametros = {
        'reductor__n_components': [20],
        'k_best__k': [20],
        'clasificador__hidden_layer_sizes': [(35, 35, 30, 30, 30, 30, 30, 30)],
        'clasificador__activation': ['relu'],
        'clasificador__solver': ['adam'],
        'clasificador__alpha': [0.353],
        'clasificador__learning_rate_init': [0.0005]
    }

    optimizador = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring='balanced_accuracy',
        cv=10,
        verbose=1,
        n_jobs=-1
    )

    optimizador.fit(X_train, y_train)

    return optimizador

# ----------------------------------------
# Guardar el modelo en gzip
# ----------------------------------------
def exportar_modelo(modelo_final):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo_final, archivo)

# ----------------------------------------
# Calcular métricas principales
# ----------------------------------------
def evaluar_metricas(est, x_tr, y_tr, x_ts, y_ts):
    pred_tr = est.predict(x_tr)
    pred_ts = est.predict(x_ts)

    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(precision_score(y_tr, pred_tr)),
        'balanced_accuracy': float(balanced_accuracy_score(y_tr, pred_tr)),
        'recall': float(recall_score(y_tr, pred_tr)),
        'f1_score': float(f1_score(y_tr, pred_tr))
    }

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(precision_score(y_ts, pred_ts)),
        'balanced_accuracy': float(balanced_accuracy_score(y_ts, pred_ts)),
        'recall': float(recall_score(y_ts, pred_ts)),
        'f1_score': float(f1_score(y_ts, pred_ts))
    }

    return train_metrics, test_metrics

# ----------------------------------------
# Calcular matrices de confusión
# ----------------------------------------
def matrices_confusion(est, x_tr, y_tr, x_ts, y_ts):
    pred_tr = est.predict(x_tr)
    pred_ts = est.predict(x_ts)

    matriz_train = confusion_matrix(y_tr, pred_tr)
    matriz_test = confusion_matrix(y_ts, pred_ts)

    cm_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {"predicted_0": int(matriz_train[0, 0]), "predicted_1": int(matriz_train[0, 1])},
        'true_1': {"predicted_0": int(matriz_train[1, 0]), "predicted_1": int(matriz_train[1, 1])}
    }

    cm_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {"predicted_0": int(matriz_test[0, 0]), "predicted_1": int(matriz_test[0, 1])},
        'true_1': {"predicted_0": int(matriz_test[1, 0]), "predicted_1": int(matriz_test[1, 1])}
    }

    return cm_train, cm_test

# ----------------------------------------
# Ejecución principal
# ----------------------------------------
if __name__ == "__main__":
    archivo_entrenamiento = 'files/input/train_data.csv.zip'
    archivo_prueba = 'files/input/test_data.csv.zip'

    df_train = pd.read_csv(archivo_entrenamiento, compression='zip')
    df_test = pd.read_csv(archivo_prueba, compression='zip')

    df_train = limpiar_dataset(df_train)
    df_test = limpiar_dataset(df_test)

    X_tr, y_tr = separar_variables(df_train, 'default')
    X_ts, y_ts = separar_variables(df_test, 'default')

    modelo_pipeline = armar_pipeline(X_tr)

    modelo_ajustado = buscar_hiperparametros(modelo_pipeline, X_tr, y_tr)

    exportar_modelo(modelo_ajustado)

    met_tr, met_ts = evaluar_metricas(modelo_ajustado, X_tr, y_tr, X_ts, y_ts)
    cm_tr, cm_ts = matrices_confusion(modelo_ajustado, X_tr, y_tr, X_ts, y_ts)

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as salida:
        for fila in [met_tr, met_ts, cm_tr, cm_ts]:
            salida.write(json.dumps(fila) + "\n")
