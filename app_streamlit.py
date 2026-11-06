# App Streamlit - Previsão de carga alta (Projeto Final FA-BC / EVCOMX)
# Grupo: Bernardo, Fernando, Gabriela e Rafael
#
# Para rodar:  streamlit run app_streamlit.py
# O app usa o modelo e os resultados que a pipeline salva na pasta "saida"
# (python -m carga_alta tudo).

import os

import joblib
import pandas as pd
import streamlit as st

from carga_alta import config
from carga_alta import modelo as M

st.set_page_config(page_title="Carga Alta - EVCOMX", page_icon="🔥", layout="wide")

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQ_MODELO = os.path.join(PASTA, "saida", "modelo", "modelo_votacao.joblib")
ARQ_PREVISOES = os.path.join(PASTA, "saida", "previsoes", "previsoes.csv")
ARQ_COMPARACAO = os.path.join(PASTA, "saida", "avaliacao", "comparacao_alternativas.csv")


# ------------------------------------------------------------
# Carregando o modelo e os dados
# ------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return joblib.load(ARQ_MODELO)


@st.cache_data
def carregar_previsoes():
    # previsões do mês de prova (os 352 cestos que o modelo não viu no treino)
    df = pd.read_csv(ARQ_PREVISOES)
    df["data"] = pd.to_datetime(df["dt_ini"].str[:19])
    df["resultado"] = df["is_carga_alta_cesto"].map({1: "Deu carga alta", 0: "Não deu"})
    return df


@st.cache_data
def carregar_comparacao():
    return pd.read_csv(ARQ_COMPARACAO)


if not os.path.exists(ARQ_MODELO):
    st.error("Modelo não encontrado. Rode antes a pipeline: python -m carga_alta tudo")
    st.stop()

modelo = carregar_modelo()
previsoes = carregar_previsoes()
comparacao = carregar_comparacao()


def mostrar_nivel(nivel):
    # mostra o nível de risco com a cor certa e a ação recomendada
    acao = config.ACOES[nivel]
    if nivel == "vermelho":
        st.error(f"🔴 **VERMELHO** - {acao}")
    elif nivel == "amarelo":
        st.warning(f"🟡 **AMARELO** - {acao}")
    else:
        st.success(f"🟢 **VERDE** - {acao}")


# ------------------------------------------------------------
# Menu
# ------------------------------------------------------------
st.sidebar.title("🔥 Carga Alta")
st.sidebar.write("Projeto Final FA-BC - EVCOMX")
pagina = st.sidebar.radio(
    "Menu",
    ["Início", "Simular um cesto", "Mês de prova", "Prever arquivo CSV", "Sobre o projeto"],
)
st.sidebar.divider()
st.sidebar.write("**Níveis de risco**")
st.sidebar.write(f"🟢 Verde: nota abaixo de {config.CORTE_AMARELO}")
st.sidebar.write(f"🟡 Amarelo: {config.CORTE_AMARELO} a {config.CORTE_VERMELHO}")
st.sidebar.write(f"🔴 Vermelho: {config.CORTE_VERMELHO} ou mais")


# ------------------------------------------------------------
# Página: Início
# ------------------------------------------------------------
if pagina == "Início":
    st.title("Previsão de carga alta no forno")
    st.write(
        "Este app dá uma **nota de risco de 0 a 100** para cada cesto antes de ele entrar no forno. "
        "Quanto maior a nota, maior a chance de o cesto dar carga alta e parar o forno."
    )

    # resultados no mês de prova (59 alertas para todas as alternativas)
    mes = comparacao[comparacao["avaliacao"] == "mês de prova"]
    nossa = mes[mes["alternativa"] == "Votação suave (solução)"].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Paradas antecipadas", f"{int(nossa['paradas_antecipadas'])} de {int(nossa['paradas'])}")
    col2.metric("F1-score", f"{nossa['f1']:.2f}")
    col3.metric("Ganho no mês", f"R$ {nossa['reais_liquidos'] / 1000:.0f} mil")

    st.subheader("Comparação com as outras alternativas")
    st.write("Todas com o mesmo número de alertas (59) no mês de prova, que teve 57 paradas.")
    grafico = pd.DataFrame({
        "Alternativa": ["Hoje (sem alerta)", "Regra simples", "Modelo anterior", "Votação suave"],
        "Paradas antecipadas": [0] + list(mes["paradas_antecipadas"]),
    })
    st.bar_chart(grafico, x="Alternativa", y="Paradas antecipadas")

    tabela = pd.DataFrame({
        "Alternativa": ["Regra simples", "Modelo anterior", "Votação suave"],
        "Paradas antecipadas": mes["paradas_antecipadas"].values,
        "F1-score": mes["f1"].round(2).values,
        "Minutos ganhos": mes["min_liquidos"].round(0).values,
        "R$ ganhos no mês": ((mes["reais_liquidos"] / 1000).round(0).astype(int).astype(str) + " mil").values,
    })
    st.dataframe(tabela, hide_index=True)
    st.caption("Minutos ganhos = minutos de parada nos cestos alertados que pararam, menos 1 minuto por alarme falso. "
               "Usamos R$ 2.000 por minuto de forno parado.")


# ------------------------------------------------------------
# Página: Simular um cesto
# ------------------------------------------------------------
elif pagina == "Simular um cesto":
    st.title("Simular um cesto")
    st.write("Escolha um cesto do mês de prova e mude os principais sinais para ver como a nota muda.")

    previsoes["nome"] = (previsoes["cd_corrida"] + " - cesto " + previsoes["nu_carregamento"].astype(str)
                         + " - " + previsoes["data"].dt.strftime("%d/%m %H:%M"))
    escolha = st.selectbox("Cesto", previsoes["nome"])
    cesto = previsoes[previsoes["nome"] == escolha].iloc[0]

    st.subheader("Principais sinais")
    col1, col2 = st.columns(2)
    with col1:
        corridas_24h = st.slider("Corridas nas últimas 24 h (ritmo do forno)", 0, 30, int(cesto["corridas_24h"]))
        frac_pesado = st.slider("Fração do volume com material pesado", 0.0, 1.0,
                                float(round(cesto["frac_vol_pesado"], 2)), 0.01)
        volume = st.number_input("Volume do cesto (m³)", min_value=1.0, value=float(cesto["volume_m3"]))
    with col2:
        y_ant = st.checkbox("A corrida anterior deu carga alta neste cesto?", value=bool(cesto["y_ant"] == 1))
        taxa_ult20 = st.slider("Taxa de carga alta nas últimas 20 corridas", 0.0, 1.0,
                               float(round(cesto["taxa_ult20"], 2)), 0.05)
        peso = st.number_input("Peso do cesto (t)", min_value=1.0, value=float(cesto["peso_t"]))

    # os outros sinais continuam iguais aos do cesto escolhido
    entrada = cesto[config.VARIAVEIS].to_frame().T
    entrada["corridas_24h"] = corridas_24h
    if frac_pesado != round(cesto["frac_vol_pesado"], 2):  # o slider arredonda, então só troca se foi mexido
        entrada["frac_vol_pesado"] = frac_pesado
    entrada["volume_m3"] = volume
    entrada["peso_t"] = peso
    entrada["dens_media"] = peso / volume
    entrada["y_ant"] = 1 if y_ant else 0
    entrada["taxa_ult20"] = taxa_ult20

    nota = M.nota_de_risco(modelo, entrada)[0]
    nivel = M.nivel([nota])[0]

    st.divider()
    col1, col2 = st.columns([1, 2])
    col1.metric("Nota de risco", f"{nota:.1f}")
    with col2:
        mostrar_nivel(nivel)
    st.write("**O que mais aumenta a nota:**")
    for motivo in M.motivos(modelo, entrada)[0].split("; "):
        st.write("-", motivo)

    if cesto["is_carga_alta_cesto"] == 1:
        st.info(f"Na realidade este cesto deu carga alta ({int(cesto['segundos_parada_cesto'])} s de parada). "
                f"Nota original: {cesto['nota_risco']:.1f}")
    else:
        st.info(f"Na realidade este cesto não deu carga alta. Nota original: {cesto['nota_risco']:.1f}")


# ------------------------------------------------------------
# Página: Mês de prova
# ------------------------------------------------------------
elif pagina == "Mês de prova":
    st.title("Resultados no mês de prova")
    st.write(f"{len(previsoes)} cestos de 03/07 a 04/08/2026, que o modelo não viu no treino.")

    vermelhos = previsoes[previsoes["nivel"] == "vermelho"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Cestos", len(previsoes))
    col2.metric("Deram carga alta", int(previsoes["is_carga_alta_cesto"].sum()))
    col3.metric("Cestos no vermelho", len(vermelhos))

    st.subheader("Nota de cada cesto ao longo do mês")
    st.scatter_chart(previsoes, x="data", y="nota_risco", color="resultado")

    st.subheader("Carga alta em cada nível")
    resumo = previsoes.groupby("nivel").agg(
        cestos=("nota_risco", "count"),
        carga_alta=("is_carga_alta_cesto", "sum"),
    )
    resumo = resumo.reindex(["verde", "amarelo", "vermelho"])
    resumo["% dos cestos"] = (100 * resumo["cestos"] / resumo["cestos"].sum()).round(0)
    resumo["% com carga alta"] = (100 * resumo["carga_alta"] / resumo["cestos"]).round(0)
    st.dataframe(resumo)

    st.subheader("Tabela dos cestos")
    niveis = st.multiselect("Filtrar por nível", ["verde", "amarelo", "vermelho"], default=["vermelho"])
    filtro = previsoes[previsoes["nivel"].isin(niveis)].sort_values("nota_risco", ascending=False)
    colunas = ["cd_corrida", "nu_carregamento", "data", "nota_risco", "nivel", "resultado", "motivos"]
    st.dataframe(filtro[colunas], hide_index=True)
    st.download_button("Baixar tabela (CSV)", filtro[colunas].to_csv(index=False), "cestos_mes_de_prova.csv")


# ------------------------------------------------------------
# Página: Prever arquivo CSV
# ------------------------------------------------------------
elif pagina == "Prever arquivo CSV":
    st.title("Prever um arquivo de cestos")
    st.write("Envie um CSV com uma linha por cesto e as colunas dos 27 sinais (lista na página *Sobre o projeto*). "
             "Use ponto como separador decimal.")

    exemplo = os.path.join(PASTA, "exemplos", "cestos_exemplo.csv")
    if os.path.exists(exemplo):
        with open(exemplo, "rb") as f:
            st.download_button("Baixar arquivo de exemplo", f.read(), "cestos_exemplo.csv")

    arquivo = st.file_uploader("Arquivo CSV", type="csv")
    if arquivo is not None:
        dados = pd.read_csv(arquivo, sep=None, engine="python")  # aceita vírgula ou ponto e vírgula

        faltando = [v for v in config.VARIAVEIS if v not in dados.columns]
        if faltando:
            st.warning("Colunas que faltaram (o modelo usa a mediana do treino): " + ", ".join(faltando))

        dados["nota_risco"] = M.nota_de_risco(modelo, dados)
        dados["nivel"] = M.nivel(dados["nota_risco"])
        st.write(dados["nivel"].value_counts())
        st.dataframe(dados, hide_index=True)
        st.download_button("Baixar resultado", dados.to_csv(index=False), "previsoes.csv")


# ------------------------------------------------------------
# Página: Sobre o projeto
# ------------------------------------------------------------
else:
    st.title("Sobre o projeto")
    st.write("""
O modelo é uma **votação suave** entre dois modelos treinados com os mesmos 27 sinais de cada cesto:

- **Regressão logística** (C = 0,5), que dá um peso para cada sinal;
- **Gradient Boosting** com 300 árvores, que pega combinações entre os sinais.

A nota é a média das duas probabilidades vezes 100. Ela serve para **ordenar** os cestos por risco,
não é a chance exata de dar carga alta.

Avaliação: o modelo foi testado em 5 períodos ao longo do ano e no último mês dos dados (mês de prova),
sempre com o mesmo número de alertas para comparar com a regra simples e com o modelo anterior.
""")

    st.subheader("O que fazer em cada nível")
    st.table(pd.DataFrame({
        "Nível": ["Verde", "Amarelo", "Vermelho"],
        "Nota": [f"abaixo de {config.CORTE_AMARELO}", f"{config.CORTE_AMARELO} a {config.CORTE_VERMELHO}",
                 f"{config.CORTE_VERMELHO} ou mais"],
        "Ação": [config.ACOES["verde"], config.ACOES["amarelo"], config.ACOES["vermelho"]],
    }))

    with st.expander("Ver os 27 sinais usados pelo modelo"):
        st.table(pd.DataFrame({
            "Coluna": config.VARIAVEIS,
            "Descrição": [config.DESCRICAO_VARIAVEIS[v] for v in config.VARIAVEIS],
        }))

    st.write("**Grupo:** Bernardo, Fernando, Gabriela e Rafael")
