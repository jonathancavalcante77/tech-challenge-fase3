from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Tech Challenge Fase 3",
    layout="wide",
)


CLASSIFICACAO = {
    "Regressão logística": {"roc_auc": 0.6348, "f1": 0.6056},
    "Random forest": {"roc_auc": 0.7015, "f1": 0.6554},
}

REGRESSAO = {
    "Ridge": {"rmse": 55.96, "mae": 33.82, "r2": 0.0144},
    "Random forest regressor": {"rmse": 54.48, "mae": 32.70, "r2": 0.0659},
}

CLUSTERING = {
    "aeroportos": 403,
    "intervalo_k": "2 a 10",
    "melhor_k": 2,
    "melhor_silhouette": 0.3867,
    "k_adotado": 4,
    "silhouette_k4": 0.3101,
    "pca_2d": 83.5,
    "anomalias": 3000,
    "pct_anomalias": 2.0,
}

ARQUIVOS_RESUMO = {
    "geral": "dashboard_resumo_geral.csv",
    "periodo": "dashboard_resumo_periodo.csv",
    "aeroporto": "dashboard_resumo_aeroporto.csv",
    "estado": "dashboard_resumo_estado.csv",
}


def validar_arquivos() -> None:
    ausentes = [arquivo for arquivo in ARQUIVOS_RESUMO.values() if not Path(arquivo).exists()]
    if ausentes:
        st.error(
            "Arquivos de resumo não encontrados. Execute `python preparar_dados_dashboard.py` antes de abrir o painel."
        )
        st.stop()


@st.cache_data(show_spinner="Carregando dados do painel...")
def carregar_resumos():
    resumo_geral = pd.read_csv(ARQUIVOS_RESUMO["geral"])
    resumo_periodo = pd.read_csv(ARQUIVOS_RESUMO["periodo"])
    resumo_aeroporto = pd.read_csv(ARQUIVOS_RESUMO["aeroporto"])
    resumo_estado = pd.read_csv(ARQUIVOS_RESUMO["estado"])
    airports = pd.read_csv("airports.csv")
    airlines = pd.read_csv("airlines.csv")
    return resumo_geral, resumo_periodo, resumo_aeroporto, resumo_estado, airports, airlines


def filtrar(df: pd.DataFrame, mes, companhias):
    base = df.copy()
    if mes != "Todos":
        base = base[base["MONTH"] == int(mes)]
    if companhias:
        base = base[base["AIRLINE"].isin(companhias)]
    return base


def consolidar_taxa(df: pd.DataFrame, grupo: list[str]) -> pd.DataFrame:
    resumo = df.groupby(grupo, as_index=False).agg(
        total_voos=("total_voos", "sum"),
        total_atrasados=("total_atrasados", "sum"),
    )
    resumo["pct_atraso"] = resumo["total_atrasados"] / resumo["total_voos"]
    return resumo


validar_arquivos()
resumo_geral, resumo_periodo, resumo_aeroporto, resumo_estado, airports, airlines = carregar_resumos()

airline_map = dict(zip(airlines["IATA_CODE"], airlines["AIRLINE"]))

st.title("Tech Challenge Fase 3 - Análise de atrasos de voos")
st.markdown(
    "Painel analítico baseado em agregações geradas a partir da base completa de voos não cancelados."
)


with st.sidebar:
    st.header("Recortes analíticos")
    opcoes_mes = ["Todos"] + [str(m) for m in sorted(resumo_geral["MONTH"].dropna().unique())]
    mes = st.selectbox("Mês", options=opcoes_mes, index=0)

    opcoes_companhia = sorted(resumo_geral["AIRLINE"].dropna().unique().tolist())
    companhias = st.multiselect(
        "Companhias aéreas",
        options=opcoes_companhia,
        format_func=lambda x: f"{x} - {airline_map.get(x, x)}",
    )


geral_filtrado = filtrar(resumo_geral, mes, companhias)
periodo_filtrado = filtrar(resumo_periodo, mes, companhias)
aeroporto_filtrado = filtrar(resumo_aeroporto, mes, companhias)
estado_filtrado = filtrar(resumo_estado, mes, companhias)

if geral_filtrado.empty:
    st.warning("Nenhum dado encontrado para os recortes selecionados.")
    st.stop()


total_voos = int(geral_filtrado["total_voos"].sum())
total_atrasados = int(geral_filtrado["total_atrasados"].sum())
taxa_atraso = total_atrasados / total_voos * 100
atraso_medio = geral_filtrado["soma_delay"].sum() / total_voos

limiar_voos = 1000 if mes == "Todos" and not companhias else 100

aeroporto_resumo = aeroporto_filtrado.groupby("ORIGIN_AIRPORT", as_index=False).agg(
    total_voos=("total_voos", "sum"),
    total_atrasados=("total_atrasados", "sum"),
    soma_delay=("soma_delay", "sum"),
)
aeroporto_resumo["pct_atraso"] = aeroporto_resumo["total_atrasados"] / aeroporto_resumo["total_voos"]
aeroporto_resumo = aeroporto_resumo[aeroporto_resumo["total_voos"] >= limiar_voos].sort_values(
    "pct_atraso", ascending=False
)

periodo_resumo = consolidar_taxa(periodo_filtrado, ["PERIOD_OF_DAY"])
periodo_resumo["PERIOD_OF_DAY"] = pd.Categorical(
    periodo_resumo["PERIOD_OF_DAY"],
    categories=["Madrugada", "Manhã", "Tarde", "Noite"],
    ordered=True,
)
periodo_resumo = periodo_resumo.sort_values("PERIOD_OF_DAY")

mensal_resumo = consolidar_taxa(geral_filtrado, ["MONTH"])
companhia_resumo = consolidar_taxa(geral_filtrado, ["AIRLINE"])
companhia_resumo["companhia"] = companhia_resumo["AIRLINE"].map(airline_map)
companhia_resumo = companhia_resumo.sort_values("pct_atraso", ascending=False)

estado_resumo = consolidar_taxa(estado_filtrado, ["STATE"])
estado_resumo = estado_resumo[estado_resumo["total_voos"] >= limiar_voos].sort_values(
    "pct_atraso", ascending=False
)

mapa_aeroportos = aeroporto_resumo.merge(
    airports,
    left_on="ORIGIN_AIRPORT",
    right_on="IATA_CODE",
    how="left",
).dropna(subset=["LATITUDE", "LONGITUDE"])


col1, col2, col3, col4 = st.columns(4)
col1.metric("Voos analisados", f"{total_voos:,}")
col2.metric("Taxa de atraso", f"{taxa_atraso:.2f}%")
col3.metric("Atraso médio", f"{atraso_medio:.2f} min")
if not aeroporto_resumo.empty:
    col4.metric(
        "Aeroporto com maior taxa",
        aeroporto_resumo.iloc[0]["ORIGIN_AIRPORT"],
        f"{aeroporto_resumo.iloc[0]['pct_atraso'] * 100:.1f}%",
    )
else:
    col4.metric("Aeroporto com maior taxa", "N/D")


st.subheader("Distribuição temporal dos atrasos")
graf1, graf2 = st.columns(2)

fig_periodo = px.bar(
    periodo_resumo,
    x="PERIOD_OF_DAY",
    y="pct_atraso",
    labels={"PERIOD_OF_DAY": "Período do dia", "pct_atraso": "Taxa de atraso"},
    title="Taxa de atraso por período do dia",
    color="PERIOD_OF_DAY",
)
fig_periodo.update_yaxes(tickformat=".0%")
graf1.plotly_chart(fig_periodo, width="stretch")

fig_mes = px.bar(
    mensal_resumo,
    x="MONTH",
    y="pct_atraso",
    labels={"MONTH": "Mês", "pct_atraso": "Taxa de atraso"},
    title="Taxa de atraso por mês",
)
fig_mes.update_yaxes(tickformat=".0%")
graf2.plotly_chart(fig_mes, width="stretch")


st.subheader("Aeroportos com maior taxa de atraso")
if not aeroporto_resumo.empty:
    top_aeroportos = aeroporto_resumo.head(15).sort_values("pct_atraso")
    fig_aero = px.bar(
        top_aeroportos,
        x="pct_atraso",
        y="ORIGIN_AIRPORT",
        orientation="h",
        labels={"pct_atraso": "Taxa de atraso", "ORIGIN_AIRPORT": "Aeroporto"},
        title="Top 15 aeroportos",
    )
    fig_aero.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_aero, width="stretch")
else:
    st.info("O recorte selecionado não atingiu o volume mínimo definido para esta visualização.")


st.subheader("Companhias aéreas e estados")
graf3, graf4 = st.columns(2)

top_companhias = companhia_resumo.head(10).sort_values("pct_atraso")
fig_companhia = px.bar(
    top_companhias,
    x="pct_atraso",
    y="companhia",
    orientation="h",
    labels={"pct_atraso": "Taxa de atraso", "companhia": "Companhia aérea"},
    title="Companhias com maior taxa de atraso",
)
fig_companhia.update_xaxes(tickformat=".0%")
graf3.plotly_chart(fig_companhia, width="stretch")

top_estados = estado_resumo.head(10).sort_values("pct_atraso")
if not top_estados.empty:
    fig_estado = px.bar(
        top_estados,
        x="pct_atraso",
        y="STATE",
        orientation="h",
        labels={"pct_atraso": "Taxa de atraso", "STATE": "Estado"},
        title="Estados com maior taxa de atraso",
    )
    fig_estado.update_xaxes(tickformat=".0%")
    graf4.plotly_chart(fig_estado, width="stretch")
else:
    graf4.info("O recorte selecionado não gerou estados com volume mínimo para comparação.")


st.subheader("Distribuição geográfica dos aeroportos")
if not mapa_aeroportos.empty:
    fig_mapa = px.scatter_geo(
        mapa_aeroportos,
        lat="LATITUDE",
        lon="LONGITUDE",
        size="total_voos",
        color=mapa_aeroportos["pct_atraso"] * 100,
        hover_name="ORIGIN_AIRPORT",
        hover_data={
            "AIRPORT": True,
            "CITY": True,
            "STATE": True,
            "total_voos": True,
            "LATITUDE": False,
            "LONGITUDE": False,
        },
        projection="albers usa",
        title="Mapa dos aeroportos por taxa de atraso e volume de voos",
        labels={"color": "Taxa de atraso (%)", "total_voos": "Volume de voos"},
    )
    st.plotly_chart(fig_mapa, width="stretch")
else:
    st.info("O recorte selecionado não gerou aeroportos suficientes para o mapa com o volume mínimo definido.")


st.subheader("Síntese dos modelos")
tab1, tab2, tab3 = st.tabs(["Classificação", "Regressão", "Não supervisionado"])

with tab1:
    df_class = pd.DataFrame(
        [
            {"Modelo": nome, "ROC-AUC": vals["roc_auc"], "F1": vals["f1"]}
            for nome, vals in CLASSIFICACAO.items()
        ]
    )
    st.dataframe(df_class, width="stretch", hide_index=True)
    st.markdown(
        "**Leitura técnica:** o melhor resultado de classificação foi obtido com random forest. "
        "A variável de maior relevância foi o horário programado da partida."
    )

with tab2:
    df_reg = pd.DataFrame(
        [
            {"Modelo": nome, "RMSE": vals["rmse"], "MAE": vals["mae"], "R²": vals["r2"]}
            for nome, vals in REGRESSAO.items()
        ]
    )
    st.dataframe(df_reg, width="stretch", hide_index=True)
    st.markdown(
        "**Leitura técnica:** o random forest regressor superou o ridge, mas o R² permaneceu baixo. "
        "Isso indica limitação para estimar com precisão os minutos exatos de atraso."
    )

with tab3:
    df_unsup = pd.DataFrame(
        [
            {"Indicador": "Aeroportos analisados", "Valor": CLUSTERING["aeroportos"]},
            {"Indicador": "Intervalo avaliado para k", "Valor": CLUSTERING["intervalo_k"]},
            {"Indicador": "Melhor silhouette", "Valor": f"{CLUSTERING['melhor_silhouette']:.4f} com k={CLUSTERING['melhor_k']}"},
            {"Indicador": "Solução adotada", "Valor": f"k={CLUSTERING['k_adotado']} com silhouette {CLUSTERING['silhouette_k4']:.4f}"},
            {"Indicador": "PCA em 2 componentes", "Valor": f"{CLUSTERING['pca_2d']:.1f}% da variância"},
            {"Indicador": "Anomalias", "Valor": f"{CLUSTERING['anomalias']:,} voos ({CLUSTERING['pct_anomalias']:.2f}%)"},
        ]
    )
    st.dataframe(df_unsup, width="stretch", hide_index=True)
    st.markdown(
        "**Leitura técnica:** a melhor separação estatística apareceu com k=2. "
        "A configuração com k=4 foi mantida na interpretação final por oferecer maior detalhamento dos perfis de aeroportos, "
        "sem comprometer completamente a coerência do agrupamento."
    )


st.subheader("Síntese analítica")
st.markdown(
    f"""
    - **Aeroporto com maior taxa de atraso no recorte atual:** {aeroporto_resumo.iloc[0]['ORIGIN_AIRPORT'] if not aeroporto_resumo.empty else 'N/D'}.
    - **Variável mais relevante no melhor classificador:** horário programado da partida.
    - **Período do dia com maior incidência de atraso:** {periodo_resumo.sort_values('pct_atraso', ascending=False).iloc[0]['PERIOD_OF_DAY']}.
    - **Estrutura de agrupamento adotada na análise não supervisionada:** {CLUSTERING['k_adotado']} grupos com apoio de PCA.
    - **Capacidade preditiva do melhor classificador:** ROC-AUC de {CLASSIFICACAO['Random forest']['roc_auc']:.4f}.
    """
)