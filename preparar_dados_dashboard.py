import sys

import pandas as pd


sys.stdout.reconfigure(encoding="utf-8")


def classificar_periodo(hora: int) -> str:
    if 5 <= hora < 12:
        return "Manhã"
    if 12 <= hora < 18:
        return "Tarde"
    if 18 <= hora < 23:
        return "Noite"
    return "Madrugada"


print("Carregando dados para gerar os resumos do dashboard...")

flights = pd.read_csv(
    "flights.csv",
    usecols=[
        "MONTH",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "DEPARTURE_DELAY",
        "CANCELLED",
    ],
    low_memory=False,
)

airports = pd.read_csv("airports.csv", usecols=["IATA_CODE", "STATE"])

flights = flights[flights["CANCELLED"] == 0].copy()
flights["IS_DELAYED"] = (flights["DEPARTURE_DELAY"] > 15).astype(int)
flights["SCHEDULED_DEPARTURE_HOUR"] = (
    flights["SCHEDULED_DEPARTURE"].fillna(0).astype(int) // 100
).clip(0, 23)
flights["PERIOD_OF_DAY"] = flights["SCHEDULED_DEPARTURE_HOUR"].apply(classificar_periodo)

flights_state = flights.merge(
    airports,
    left_on="ORIGIN_AIRPORT",
    right_on="IATA_CODE",
    how="left",
)

dashboard_resumo_geral = (
    flights.groupby(["MONTH", "AIRLINE"], as_index=False)
    .agg(
        total_voos=("IS_DELAYED", "count"),
        total_atrasados=("IS_DELAYED", "sum"),
        soma_delay=("DEPARTURE_DELAY", "sum"),
    )
)

dashboard_resumo_periodo = (
    flights.groupby(["MONTH", "AIRLINE", "PERIOD_OF_DAY"], as_index=False)
    .agg(
        total_voos=("IS_DELAYED", "count"),
        total_atrasados=("IS_DELAYED", "sum"),
    )
)

dashboard_resumo_aeroporto = (
    flights.groupby(["MONTH", "AIRLINE", "ORIGIN_AIRPORT"], as_index=False)
    .agg(
        total_voos=("IS_DELAYED", "count"),
        total_atrasados=("IS_DELAYED", "sum"),
        soma_delay=("DEPARTURE_DELAY", "sum"),
    )
)

dashboard_resumo_estado = (
    flights_state.groupby(["MONTH", "AIRLINE", "STATE"], as_index=False)
    .agg(
        total_voos=("IS_DELAYED", "count"),
        total_atrasados=("IS_DELAYED", "sum"),
    )
)

dashboard_resumo_geral.to_csv("dashboard_resumo_geral.csv", index=False)
dashboard_resumo_periodo.to_csv("dashboard_resumo_periodo.csv", index=False)
dashboard_resumo_aeroporto.to_csv("dashboard_resumo_aeroporto.csv", index=False)
dashboard_resumo_estado.to_csv("dashboard_resumo_estado.csv", index=False)

print("Arquivos gerados:")
print("- dashboard_resumo_geral.csv")
print("- dashboard_resumo_periodo.csv")
print("- dashboard_resumo_aeroporto.csv")
print("- dashboard_resumo_estado.csv")