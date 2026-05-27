"""
Tech Challenge Fase 3 - análise de atrasos de voos

Versão revisada após uma auditoria técnica do projeto.
O objetivo aqui é consolidar as etapas principais em um script executável,
mantendo coerência metodológica com o notebook.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto: str) -> None:
    print(f"\n- {texto}")


def classificar_periodo(hora: int) -> str:
    if 5 <= hora < 12:
        return "Manhã"
    if 12 <= hora < 18:
        return "Tarde"
    if 18 <= hora < 23:
        return "Noite"
    return "Madrugada"


def classificar_estacao(mes: int) -> str:
    if mes in [12, 1, 2]:
        return "Inverno"
    if mes in [3, 4, 5]:
        return "Primavera"
    if mes in [6, 7, 8]:
        return "Verão"
    return "Outono"


titulo("Tech Challenge Fase 3 - auditoria e consolidação dos resultados")

subtitulo("Carregando as bases")
flights = pd.read_csv("flights.csv", low_memory=False)
airlines = pd.read_csv("airlines.csv")
airports = pd.read_csv("airports.csv")

print(f"Flights:  {flights.shape[0]:,} linhas x {flights.shape[1]} colunas")
print(f"Airlines: {airlines.shape[0]} companhias")
print(f"Airports: {airports.shape[0]} aeroportos")


titulo("Etapa 1 - leitura exploratória e preparação")

subtitulo("Valores ausentes na base completa")
missing = flights.isnull().sum()
missing_pct = (missing / len(flights) * 100).round(2)
missing_df = (
    pd.DataFrame({"faltantes": missing, "percentual": missing_pct})
    .query("faltantes > 0")
    .sort_values("percentual", ascending=False)
)
print(f"Colunas com valores ausentes: {len(missing_df)}")
for coluna, linha in missing_df.head(5).iterrows():
    print(f"  - {coluna}: {linha['percentual']:.1f}%")

# Para medir atraso de partida, faz mais sentido trabalhar só com voos que de fato ocorreram.
flights_valid = flights[flights["CANCELLED"] == 0].copy()

flights_valid["IS_DELAYED"] = (flights_valid["DEPARTURE_DELAY"] > 15).astype(int)
flights_valid["SCHEDULED_DEPARTURE_HOUR"] = (
    flights_valid["SCHEDULED_DEPARTURE"].fillna(0).astype(int) // 100
).clip(0, 23)
flights_valid["PERIOD_OF_DAY"] = flights_valid["SCHEDULED_DEPARTURE_HOUR"].apply(classificar_periodo)
flights_valid["SEASON"] = flights_valid["MONTH"].apply(classificar_estacao)
flights_valid["IS_WEEKEND"] = flights_valid["DAY_OF_WEEK"].isin([6, 7]).astype(int)

subtitulo("Indicadores descritivos")
print(f"Voos não cancelados: {len(flights_valid):,}")
print(f"Voos cancelados: {int(flights['CANCELLED'].sum()):,} ({flights['CANCELLED'].mean()*100:.2f}%)")
print(f"Atraso médio na partida: {flights_valid['DEPARTURE_DELAY'].mean():.2f} min")
print(f"Atraso mediano na partida: {flights_valid['DEPARTURE_DELAY'].median():.2f} min")
print(f"Taxa de atraso (>15 min): {flights_valid['IS_DELAYED'].mean()*100:.2f}%")

subtitulo("Respostas descritivas às perguntas-guia")
top_delayed_airports = (
    flights_valid.groupby("ORIGIN_AIRPORT")
    .agg(total_voos=("IS_DELAYED", "count"), pct_atraso=("IS_DELAYED", "mean"))
    .reset_index()
)
top_delayed_airports = top_delayed_airports[top_delayed_airports["total_voos"] >= 1000]
top_delayed_airports = top_delayed_airports.sort_values("pct_atraso", ascending=False)
principal_airport = top_delayed_airports.iloc[0]

dow_labels = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 7: "Dom"}
dow_delay = flights_valid.groupby("DAY_OF_WEEK")["IS_DELAYED"].mean() * 100
period_delay = flights_valid.groupby("PERIOD_OF_DAY")["IS_DELAYED"].mean() * 100
season_delay = flights_valid.groupby("SEASON")["IS_DELAYED"].mean() * 100

print(
    f"Aeroporto mais crítico: {principal_airport['ORIGIN_AIRPORT']} "
    f"({principal_airport['pct_atraso']*100:.1f}% de atraso)"
)
print(f"Dia mais crítico: {dow_labels[dow_delay.idxmax()]} ({dow_delay.max():.1f}%)")
print(f"Período mais crítico: {period_delay.idxmax()} ({period_delay.max():.1f}%)")
print(f"Estação com maior taxa de atraso: {season_delay.idxmax()} ({season_delay.max():.1f}%)")


titulo("Etapa 2 - modelagem supervisionada")

subtitulo("Montando a amostra de modelagem")
sample_size = 200_000
sample_per_class = sample_size // 2

delayed_sample = flights_valid[flights_valid["IS_DELAYED"] == 1].sample(
    n=min(sample_per_class, (flights_valid["IS_DELAYED"] == 1).sum()),
    random_state=42,
)
on_time_sample = flights_valid[flights_valid["IS_DELAYED"] == 0].sample(
    n=min(sample_per_class, (flights_valid["IS_DELAYED"] == 0).sum()),
    random_state=42,
)

flights_sample = (
    pd.concat([delayed_sample, on_time_sample], axis=0)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
flights_sample.to_csv("flights_sample.csv", index=False)

print(f"Amostra usada no treino: {len(flights_sample):,} voos")
print("Observação: a amostra foi balanceada de propósito para comparar os modelos.")
print(f"  - Atrasados: {int(flights_sample['IS_DELAYED'].sum()):,}")
print(f"  - Pontuais:  {int((flights_sample['IS_DELAYED'] == 0).sum()):,}")

feature_cols = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DISTANCE",
    "SCHEDULED_DEPARTURE_HOUR",
    "IS_WEEKEND",
]

X = flights_sample[feature_cols].copy()
y_class = flights_sample["IS_DELAYED"].copy()
y_reg = flights_sample["DEPARTURE_DELAY"].copy()

for coluna in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
    encoder = LabelEncoder()
    X[coluna] = encoder.fit_transform(X[coluna].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_reg = y_reg.loc[X_train.index]
y_test_reg = y_reg.loc[X_test.index]

subtitulo("Classificação")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

metrics_class = {
    "Regressão logística": {
        "roc_auc": roc_auc_score(y_test, y_proba_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
    },
    "Random forest": {
        "roc_auc": roc_auc_score(y_test, y_proba_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
    },
}

for nome, valores in metrics_class.items():
    print(
        f"{nome:<22} | ROC-AUC: {valores['roc_auc']:.4f} | "
        f"Precisão: {valores['precision']:.4f} | "
        f"Recall: {valores['recall']:.4f} | F1: {valores['f1']:.4f}"
    )

importance_df = (
    pd.DataFrame({"atributo": feature_cols, "importancia": rf.feature_importances_})
    .sort_values("importancia", ascending=False)
    .reset_index(drop=True)
)

print("\nAtributos mais relevantes no random forest:")
for _, linha in importance_df.head(5).iterrows():
    print(f"  - {linha['atributo']}: {linha['importancia']:.4f}")

subtitulo("Regressão")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train_reg)
y_pred_ridge = ridge.predict(X_test_scaled)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    random_state=42,
    n_jobs=-1,
)
rf_reg.fit(X_train, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test)

metrics_reg = {
    "Ridge": {
        "rmse": np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge)),
        "mae": mean_absolute_error(y_test_reg, y_pred_ridge),
        "r2": r2_score(y_test_reg, y_pred_ridge),
    },
    "Random forest regressor": {
        "rmse": np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg)),
        "mae": mean_absolute_error(y_test_reg, y_pred_rf_reg),
        "r2": r2_score(y_test_reg, y_pred_rf_reg),
    },
}

for nome, valores in metrics_reg.items():
    print(
        f"{nome:<24} | RMSE: {valores['rmse']:.2f} min | "
        f"MAE: {valores['mae']:.2f} min | R²: {valores['r2']:.4f}"
    )


titulo("Etapa 3 - análise não supervisionada")

airport_profile = (
    flights_valid.groupby("ORIGIN_AIRPORT")
    .agg(
        total_voos=("DEPARTURE_DELAY", "count"),
        atraso_medio=("DEPARTURE_DELAY", "mean"),
        desvio_atraso=("DEPARTURE_DELAY", "std"),
        pct_atraso=("IS_DELAYED", "mean"),
        distancia_media=("DISTANCE", "mean"),
        tempo_programado_medio=("SCHEDULED_TIME", "mean"),
    )
    .reset_index()
    .fillna(0)
)
airport_profile = airport_profile[airport_profile["total_voos"] >= 500].copy()

cluster_features = [
    "atraso_medio",
    "pct_atraso",
    "distancia_media",
    "tempo_programado_medio",
    "total_voos",
]

X_cluster = airport_profile[cluster_features].copy()
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

silhouette_scores = {}
for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_cluster_scaled)
    silhouette_scores[k] = silhouette_score(X_cluster_scaled, labels)

best_k = max(silhouette_scores, key=silhouette_scores.get)
chosen_k = 4
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
airport_profile["cluster"] = kmeans.fit_predict(X_cluster_scaled)

pca = PCA()
pca.fit(X_cluster_scaled)
var_2pc = np.sum(pca.explained_variance_ratio_[:2]) * 100

subtitulo("Clustering de aeroportos")
print(f"Aeroportos analisados: {len(airport_profile)}")
print(
    f"Melhor silhouette entre k=2 e k=6: k={best_k} "
    f"({silhouette_scores[best_k]:.4f})"
)
print(
    f"k adotado para leitura dos perfis: {chosen_k} "
    f"({silhouette_scores[chosen_k]:.4f})"
)
for cluster_id in sorted(airport_profile["cluster"].unique()):
    qtd = int((airport_profile["cluster"] == cluster_id).sum())
    print(f"  - Cluster {cluster_id}: {qtd} aeroportos")
print(f"Variância explicada nas 2 primeiras componentes: {var_2pc:.1f}%")

subtitulo("Detecção de anomalias")
delayed_pool = flights_valid[flights_valid["DEPARTURE_DELAY"] > 0].copy()
anomaly_sample_size = min(150_000, len(delayed_pool))
delayed_flights = delayed_pool.sample(n=anomaly_sample_size, random_state=42)
anomaly_features = ["DEPARTURE_DELAY", "DISTANCE", "SCHEDULED_TIME", "AIR_TIME"]
X_anomaly = delayed_flights[anomaly_features].fillna(0)

iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
anomaly_labels = iso_forest.fit_predict(X_anomaly)
anomalies_count = int((anomaly_labels == -1).sum())

print(f"Amostra usada na análise de anomalias: {len(delayed_flights):,} voos com atraso positivo")
print(f"Anomalias identificadas: {anomalies_count:,} ({anomalies_count / len(delayed_flights) * 100:.2f}%)")


titulo("Etapa 4 - análises complementares")

subtitulo("Companhias com maior e menor taxa de atraso")
airline_delay = (
    flights_valid.groupby("AIRLINE")
    .agg(
        total_voos=("IS_DELAYED", "count"),
        atraso_medio=("DEPARTURE_DELAY", "mean"),
        pct_atraso=("IS_DELAYED", "mean"),
    )
    .reset_index()
    .merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
    .sort_values("pct_atraso", ascending=False)
)

for _, linha in airline_delay.head(5).iterrows():
    print(f"  - {linha['AIRLINE_y']}: {linha['pct_atraso']*100:.1f}%")

print("\nCompanhias com menor taxa de atraso:")
for _, linha in airline_delay.tail(5).iterrows():
    print(f"  - {linha['AIRLINE_y']}: {linha['pct_atraso']*100:.1f}%")

subtitulo("Estados com maior taxa de atraso")
state_delay = (
    flights_valid.merge(
        airports[["IATA_CODE", "STATE"]],
        left_on="ORIGIN_AIRPORT",
        right_on="IATA_CODE",
        how="left",
    )
    .groupby("STATE")
    .agg(total_voos=("IS_DELAYED", "count"), pct_atraso=("IS_DELAYED", "mean"))
    .reset_index()
)
state_delay = state_delay[state_delay["total_voos"] >= 100].sort_values("pct_atraso", ascending=False)

for _, linha in state_delay.head(5).iterrows():
    print(f"  - {linha['STATE']}: {linha['pct_atraso']*100:.1f}%")


titulo("Resumo final")

print("Modelos de classificação")
for nome, valores in metrics_class.items():
    print(f"  - {nome}: ROC-AUC = {valores['roc_auc']:.4f} | F1 = {valores['f1']:.4f}")

print("\nModelos de regressão")
for nome, valores in metrics_reg.items():
    print(
        f"  - {nome}: RMSE = {valores['rmse']:.2f} min | "
        f"MAE = {valores['mae']:.2f} min | R² = {valores['r2']:.4f}"
    )

print("\nPerguntas-guia")
print(
    f"  1. Aeroportos mais críticos: {principal_airport['ORIGIN_AIRPORT']} "
    f"({principal_airport['pct_atraso']*100:.1f}% de atraso)"
)
print(
    f"  2. Atributo mais relevante no modelo: "
    f"{importance_df.iloc[0]['atributo']} ({importance_df.iloc[0]['importancia']:.3f})"
)
print(f"  3. Horário mais crítico: {period_delay.idxmax()} ({period_delay.max():.1f}%)")
print(f"  4. Agrupamento de aeroportos: {chosen_k} clusters interpretáveis")
print(f"  5. Capacidade preditiva: ROC-AUC = {metrics_class['Random forest']['roc_auc']:.3f}")

print("\nArquivos principais")
print("  - Notebook: tech_challenge_fase3.ipynb")
print("  - Script: tech_challenge_completo.py")
print("  - Amostra balanceada de modelagem: flights_sample.csv")