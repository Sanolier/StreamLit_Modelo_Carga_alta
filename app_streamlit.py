"""App Streamlit da nota de risco de carga alta (EVCOMX).

Rodar a partir da pasta do projeto:
    streamlit run app_streamlit.py
ou pelo Docker:
    docker compose up --build      (app em http://localhost:8501)

Usa o mesmo pacote da pipeline (carga_alta). Se ainda não houver dados processados, modelo ou
avaliação em ./saida, o app gera tudo na primeira abertura (cerca de 20 segundos).
"""
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carga_alta import config as C  # noqa: E402
from carga_alta import modelo as M  # noqa: E402
from carga_alta import preparar  # noqa: E402

# ---------------------------------------------------------------- aparência
COR = dict(verde='#3F8F5F', amarelo='#E8A317', vermelho='#D62F1F', laranja='#FF4400',
           grafite='#2B313A', cinza='#8A919B', cinza_claro='#E9E9E9')
DIAS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']

ROTULO = {
    'cesto2': 'Cesto 2', 'n_camadas': 'Camadas', 'peso_t': 'Peso do cesto (t)', 'volume_m3': 'Volume do cesto (m³)',
    'dens_media': 'Densidade média (t/m³)', 'n_zero': 'Camadas com peso zero', 'n_materiais': 'Materiais diferentes',
    'frac_vol_leve': 'Fração leve', 'frac_vol_misto': 'Fração mista', 'frac_vol_pesado': 'Fração pesada',
    'max_vol_camada': 'Maior camada (m³)', 'vol_topo': 'Volume no topo (m³)', 'frac_vol_topo': 'Fração no topo',
    'vol_leve_topo': 'Leve no topo (m³)', 'frac_vol_inf': 'Fração na metade de baixo',
    'vol_c1': 'Volume do cesto 1 (m³)', 'peso_c1': 'Peso do cesto 1 (t)', 'y_c1': 'Cesto 1 deu carga alta',
    'dur_carga_min': 'Duração da montagem (min)', 'hora': 'Hora de início', 'dia_sem': 'Dia da semana',
    'y_ant': 'Corrida anterior deu carga alta', 'taxa_ult20': 'Taxa nas últimas 20 corridas',
    'horas_desde_anterior': 'Horas desde a corrida anterior', 'pausa_longa': 'Pausa de mais de 6 h',
    'corridas_24h': 'Corridas nas últimas 24 h', 'frac_fora_baia': 'Fração fora da baia usual',
}
GRUPOS = {
    'Composição do cesto': ['n_camadas', 'peso_t', 'volume_m3', 'n_zero', 'n_materiais', 'max_vol_camada',
                            'frac_vol_leve', 'frac_vol_misto', 'frac_vol_pesado'],
    'Topo do cesto': ['vol_topo', 'frac_vol_topo', 'vol_leve_topo', 'frac_vol_inf'],
    'Cesto 1 da mesma corrida': ['vol_c1', 'peso_c1', 'y_c1'],
    'Horário e montagem': ['hora', 'dia_sem', 'dur_carga_min'],
    'Histórico recente': ['y_ant', 'taxa_ult20'],
    'Ritmo do forno e baia': ['corridas_24h', 'horas_desde_anterior', 'pausa_longa', 'frac_fora_baia'],
}
INTEIRAS = {'n_camadas', 'n_zero', 'n_materiais', 'hora', 'corridas_24h'}
BINARIAS = {'y_c1', 'y_ant', 'pausa_longa'}
FRACOES = {'frac_vol_leve', 'frac_vol_misto', 'frac_vol_pesado', 'frac_vol_topo', 'frac_vol_inf',
           'taxa_ult20', 'frac_fora_baia'}
NOMES_ALTERNATIVAS = {
    'Regra simples (corrida anterior parou)': 'Regra simples',
    'Modelo anterior (regressão logística)': 'Modelo anterior',
    'Votação suave (solução)': 'Nossa solução',
}

plt.rcParams.update({'figure.facecolor': '#FBFBFA', 'axes.facecolor': '#FBFBFA', 'savefig.facecolor': '#FBFBFA',
                     'font.family': 'sans-serif'})

st.set_page_config(page_title='Carga alta · nota de risco', page_icon='🔥', layout='wide')
st.markdown("""
<style>
.nota-grande {font-size: 4.2rem; font-weight: 800; line-height: 1; margin: 0.2rem 0 0.4rem 0;}
.nivel {display: inline-block; padding: 0.25rem 0.9rem; border-radius: 999px; color: white;
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;}
.acao {font-size: 1.05rem; margin-top: 0.6rem;}
</style>
""", unsafe_allow_html=True)


def br(x, casas=0):
    """Número no formato brasileiro (1.234,5)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'{x:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def selo(nivel):
    return f'<span class="nivel" style="background:{COR[nivel]}">{nivel}</span>'


# ---------------------------------------------------------------- dados e modelo (com cache)
@st.cache_data(show_spinner='Preparando os dados dos cestos...')
def dados():
    t = preparar.carregar_cestos()
    treino, teste, _ = preparar.dividir(t)
    medianas = M.preparar_entrada(treino[C.VARIAVEIS]).median()
    return t, treino, teste, medianas


@st.cache_resource(show_spinner='Carregando o modelo...')
def modelo():
    """Carrega o modelo salvo pela pipeline. Se não existir, ou se foi salvo com outra versão do
    scikit-learn, treina de novo (cerca de 3 segundos) com o mesmo protocolo da pipeline."""
    from carga_alta import treinar
    aviso = None
    try:
        m, meta = M.carregar()
        if meta.get('versao_sklearn') != sklearn.__version__:
            aviso = (f'Modelo salvo com scikit-learn {meta.get("versao_sklearn")}; '
                     f'retreinado com a versão instalada ({sklearn.__version__}).')
            raise RuntimeError(aviso)
    except Exception:  # noqa: BLE001
        treinar.executar()
        m, meta = M.carregar()
    return m, meta, aviso


@st.cache_data(show_spinner='Calculando as notas do mês de prova...')
def notas_mes_de_prova(_modelo, chave):
    _, _, teste, _ = dados()
    out = teste.copy()
    out['nota_risco'] = M.nota_de_risco(_modelo, teste)
    out['nivel'] = M.nivel(out.nota_risco)
    out['acao'] = out.nivel.map(C.ACOES)
    out['motivos'] = M.motivos(_modelo, teste)
    return out


@st.cache_data(show_spinner='Avaliando o modelo (5 provas e mês de prova, cerca de 10 segundos)...')
def avaliacao():
    arquivos = ['comparacao_alternativas', 'niveis_de_risco', 'economia_anual', 'provas_validacao']
    if not all((C.PASTA_AVALIACAO / f'{a}.csv').exists() for a in arquivos):
        from carga_alta import avaliar
        avaliar.executar()
    return {a: pd.read_csv(C.PASTA_AVALIACAO / f'{a}.csv') for a in arquivos}


def contribuicoes(m, x):
    """Contribuição de cada sinal na parte linear da votação (coeficiente x valor padronizado)."""
    lr = m.named_estimators_['lr']
    z = lr[:-1].transform(M.preparar_entrada(x))[0]
    return pd.Series(z * lr[-1].coef_[0], index=C.VARIAVEIS)


# ---------------------------------------------------------------- gráficos
def estilo(ax):
    for lado in ['top', 'right']:
        ax.spines[lado].set_visible(False)
    ax.spines['left'].set_color(COR['cinza_claro'])
    ax.spines['bottom'].set_color(COR['cinza_claro'])
    ax.tick_params(colors=COR['grafite'], labelsize=9)


def grafico_regua(nota):
    fig, ax = plt.subplots(figsize=(6, 1.15))
    for ini, fim, n in [(0, C.CORTE_AMARELO, 'verde'), (C.CORTE_AMARELO, C.CORTE_VERMELHO, 'amarelo'),
                        (C.CORTE_VERMELHO, 100, 'vermelho')]:
        ax.barh(0, fim - ini, left=ini, height=0.55, color=COR[n], alpha=0.9)
    ax.plot([nota, nota], [-0.42, 0.42], color=COR['grafite'], lw=3, solid_capstyle='round')
    ax.annotate(br(nota, 1), (nota, 0.42), xytext=(0, 4), textcoords='offset points', ha='center',
                va='bottom', fontsize=11, fontweight='bold', color=COR['grafite'])
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 0.95)
    ax.set_xticks([0, C.CORTE_AMARELO, C.CORTE_VERMELHO, 100])
    ax.set_yticks([])
    for lado in ['top', 'right', 'left']:
        ax.spines[lado].set_visible(False)
    ax.spines['bottom'].set_color(COR['cinza_claro'])
    ax.tick_params(colors=COR['grafite'], labelsize=9)
    fig.tight_layout()
    return fig


def valor_legivel(v, x):
    if v in BINARIAS or v == 'cesto2':
        return 'sim' if x >= 0.5 else 'não'
    if v == 'dia_sem':
        return DIAS[int(round(x)) % 7].lower()
    return br(x, 0 if v in INTEIRAS else 2)


def grafico_contribuicoes(contrib, valores, n=8):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:n])[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.42 * n + 0.6))
    cores = [COR['vermelho'] if v > 0 else COR['verde'] for v in top.values]
    rotulos = [f'{ROTULO[v]} = {valor_legivel(v, valores[v])}' for v in top.index]
    ax.barh(rotulos, top.values, color=cores, height=0.6)
    ax.axvline(0, color=COR['grafite'], lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel('← reduz o risco      aumenta o risco →', fontsize=9, color=COR['grafite'])
    estilo(ax)
    fig.tight_layout()
    return fig


def grafico_notas_no_tempo(prev):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axhspan(0, C.CORTE_AMARELO, color=COR['verde'], alpha=0.07)
    ax.axhspan(C.CORTE_AMARELO, C.CORTE_VERMELHO, color=COR['amarelo'], alpha=0.10)
    ax.axhspan(C.CORTE_VERMELHO, 100, color=COR['vermelho'], alpha=0.08)
    dt = pd.to_datetime(prev.dt_ini).dt.tz_localize(None)
    nao = prev[C.ALVO] == 0
    ax.scatter(dt[nao], prev.nota_risco[nao], s=14, color=COR['cinza'], alpha=0.55, label='Não deu carga alta')
    ax.scatter(dt[~nao], prev.nota_risco[~nao], s=26, color=COR['vermelho'], edgecolor='white', lw=0.5,
               label='Deu carga alta')
    ax.set_ylim(0, 100); ax.set_ylabel('Nota de risco', fontsize=9, color=COR['grafite'])
    ax.set_yticks([0, C.CORTE_AMARELO, C.CORTE_VERMELHO, 100])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    estilo(ax)
    fig.tight_layout()
    return fig


def grafico_niveis(tab):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(3)
    ax.bar(x - 0.2, 100 * tab.frac_cestos, width=0.38, color=COR['cinza_claro'], label='% dos cestos')
    ax.bar(x + 0.2, 100 * tab.taxa_carga_alta, width=0.38, color=[COR[n] for n in tab.nivel],
           label='% que deu carga alta')
    for i, (a, b) in enumerate(zip(tab.frac_cestos, tab.taxa_carga_alta)):
        ax.text(i - 0.2, 100 * a + 1, f'{100 * a:.0f}%', ha='center', fontsize=9, color=COR['grafite'])
        ax.text(i + 0.2, 100 * b + 1, f'{100 * b:.0f}%', ha='center', fontsize=9, fontweight='bold',
                color=COR['grafite'])
    ax.set_xticks(x, [n.capitalize() for n in tab.nivel])
    ax.set_yticks([])
    ax.set_ylim(0, 100 * max(tab.frac_cestos.max(), tab.taxa_carga_alta.max()) * 1.15)
    ax.spines['left'].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    estilo(ax)
    fig.tight_layout()
    return fig


def grafico_comparacao(comp):
    m = comp[comp.avaliacao == 'mês de prova'].copy()
    m['nome'] = m.alternativa.map(NOMES_ALTERNATIVAS)
    nomes = ['Hoje (sem alerta)'] + list(m.nome)
    paradas = [0] + list(m.paradas_antecipadas)
    total = int(m.paradas.iloc[0])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    cores = [COR['cinza_claro']] * (len(nomes) - 1) + [COR['laranja']]
    ax.barh(nomes[::-1], paradas[::-1], color=cores[::-1], height=0.6)
    for i, v in enumerate(paradas[::-1]):
        ax.text(v + 0.6, i, f'{v}', va='center', fontsize=10, fontweight='bold', color=COR['grafite'])
    ax.set_xlim(0, total)
    ax.set_xlabel(f'Paradas antecipadas no mês de prova (de {total})', fontsize=9, color=COR['grafite'])
    estilo(ax)
    fig.tight_layout()
    return fig


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def csv_para_excel(df):
    """CSV no padrão do Excel em português: ponto e vírgula e vírgula decimal."""
    return df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')


def tabela_resultados(df):
    cols = {'cd_corrida': 'Corrida', 'nu_carregamento': 'Cesto', 'dt_ini': 'Início', 'nota_risco': 'Nota',
            'nivel': 'Nível', 'acao': 'Ação', 'motivos': 'Motivos (o que mais eleva a nota)',
            C.ALVO: 'Deu carga alta', C.DURACAO: 'Segundos de parada'}
    out = df[[c for c in cols if c in df]].rename(columns=cols)
    if 'Início' in out:
        out['Início'] = pd.to_datetime(out['Início']).dt.strftime('%d/%m/%Y %H:%M')
    if 'Deu carga alta' in out:
        out['Deu carga alta'] = out['Deu carga alta'].map({1: 'Sim', 0: 'Não'})
    return out


# ---------------------------------------------------------------- páginas
def pagina_simular(m, prev):
    st.title('Simular um cesto')
    st.caption('Parta de um cesto real do mês de prova (ou de valores típicos) e mude os sinais para ver '
               'como a nota de risco reage. A nota é recalculada a cada mudança.')
    _, _, _, medianas = dados()

    opcoes = ['Valores típicos (mediana do treino)']
    prev = prev.sort_values('dt_ini')
    rotulos = [f'{r.cd_corrida} · cesto {r.nu_carregamento} · {pd.to_datetime(r.dt_ini):%d/%m %H:%M} · '
               f'nota {br(r.nota_risco, 1)} ({r.nivel})' for r in prev.itertuples()]
    escolha = st.selectbox('Ponto de partida', opcoes + rotulos,
                           help='Digite o código da corrida para procurar.')
    if escolha == opcoes[0]:
        base, chave = medianas.copy(), 'mediana'
        real = None
    else:
        linha = prev.iloc[rotulos.index(escolha)]
        base = M.preparar_entrada(linha[C.VARIAVEIS].to_frame().T).iloc[0].fillna(medianas)
        chave = f'{linha.cd_corrida}_{linha.nu_carregamento}'
        real = linha

    esq, dir_ = st.columns([3, 2], gap='large')
    x = {}
    with esq:
        cesto = st.radio('Qual cesto da corrida?', ['Cesto 1', 'Cesto 2'], horizontal=True,
                         index=int(base['cesto2'] >= 0.5), key=f'cesto2_{chave}')
        x['cesto2'] = 1.0 if cesto == 'Cesto 2' else 0.0
        for grupo, variaveis in GRUPOS.items():
            so_cesto2 = grupo.startswith('Cesto 1 da mesma')
            with st.expander(grupo, expanded=grupo == 'Composição do cesto'):
                if so_cesto2 and x['cesto2'] == 0:
                    st.caption('Só vale para o cesto 2. No cesto 1 estes sinais são zero.')
                    for v in variaveis:
                        x[v] = 0.0
                    continue
                cols = st.columns(3)
                for i, v in enumerate(variaveis):
                    with cols[i % 3]:
                        x[v] = entrada(v, base[v], chave)
        x['dens_media'] = x['peso_t'] / x['volume_m3'] if x['volume_m3'] > 0 else np.nan
        st.caption(f'Densidade média calculada: {br(x["dens_media"], 3)} t/m³ (peso ÷ volume). '
                   'Os demais sinais não se recalculam sozinhos: ao mexer em um, confira os relacionados.')

    X = pd.DataFrame([x])[C.VARIAVEIS]
    nota = float(M.nota_de_risco(m, X)[0])
    nivel = str(M.nivel([nota])[0])
    with dir_:
        st.markdown(f'<div class="nota-grande" style="color:{COR[nivel]}">{br(nota, 1)}</div>'
                    f'{selo(nivel)}<div class="acao"><b>Ação:</b> {C.ACOES[nivel]}</div>',
                    unsafe_allow_html=True)
        mostrar(grafico_regua(nota))
        if real is not None:
            if real[C.ALVO] == 1:
                st.error(f'Na realidade, este cesto **deu carga alta** ({int(real[C.DURACAO])} s de parada). '
                         f'Nota original: {br(real.nota_risco, 1)}.')
            else:
                st.success(f'Na realidade, este cesto **não deu carga alta**. Nota original: {br(real.nota_risco, 1)}.')
        st.markdown('**O que mais eleva a nota**')
        texto = M.motivos(m, X)[0]
        st.markdown('\n'.join(f'- {t}' for t in texto.split('; ')) if texto else '_Nenhum sinal eleva a nota._')
        st.markdown('**Peso de cada sinal na nota** (parte linear do modelo)')
        mostrar(grafico_contribuicoes(contribuicoes(m, X), M.preparar_entrada(X).iloc[0].fillna(medianas)))
        st.caption('Vermelho aumenta o risco, verde reduz. Mostra o que o modelo usa para dar a nota; '
                   'não prova causa.')


def entrada(v, valor, chave):
    rotulo, ajuda, k = ROTULO[v], C.DESCRICAO_VARIAVEIS[v], f'{v}_{chave}'
    valor = float(valor) if pd.notna(valor) else 0.0
    if v == 'dia_sem':
        return float(DIAS.index(st.selectbox(rotulo, DIAS, index=int(round(valor)) % 7, key=k, help=ajuda)))
    if v in BINARIAS:
        return 1.0 if st.checkbox(rotulo, value=bool(valor >= 0.5), key=k, help=ajuda) else 0.0
    if v in FRACOES:
        return float(st.slider(rotulo, min_value=0.0, max_value=1.0, value=float(np.clip(valor, 0, 1)),
                               step=0.01, key=k, help=ajuda))
    if v in INTEIRAS:
        maximo = 23 if v == 'hora' else 1000
        return float(st.number_input(rotulo, min_value=0, max_value=maximo, value=int(np.clip(round(valor), 0, maximo)),
                                     step=1, key=k, help=ajuda))
    # sem limite inferior: há registros reais com valores negativos (ex.: horas desde a corrida anterior)
    return float(st.number_input(rotulo, value=valor, step=0.5, format='%.2f', key=k, help=ajuda))


def pagina_mes(prev):
    st.title('Mês de prova')
    ini, fim = pd.to_datetime(prev.dt_ini).min(), pd.to_datetime(prev.dt_ini).max()
    st.caption(f'{len(prev)} cestos de {ini:%d/%m/%Y} a {fim:%d/%m/%Y}, que o modelo nunca viu no treino. '
               'Mostra como a nota teria funcionado no pátio.')
    y = prev[C.ALVO]
    verm = prev.nivel == 'vermelho'
    c = st.columns(4)
    c[0].metric('Cestos', br(len(prev)))
    c[1].metric('Deram carga alta', br(int(y.sum())), help='Paradas reais no mês')
    c[2].metric('Cestos no vermelho', f'{int(verm.sum())} ({verm.mean():.0%})')
    c[3].metric('Paradas pegas no vermelho', f'{int(y[verm].sum())} de {int(y.sum())}',
                help=f'{y[verm].mean():.0%} dos cestos vermelhos deram carga alta')

    st.subheader('Notas ao longo do mês')
    mostrar(grafico_notas_no_tempo(prev))

    esq, dir_ = st.columns([2, 3], gap='large')
    with esq:
        st.subheader('Por nível')
        tab = pd.DataFrame([dict(nivel=n, frac_cestos=(prev.nivel == n).mean(),
                                 taxa_carga_alta=y[prev.nivel == n].mean() if (prev.nivel == n).any() else 0)
                            for n in ['verde', 'amarelo', 'vermelho']])
        mostrar(grafico_niveis(tab))
    with dir_:
        st.subheader('Cestos')
        f1, f2, f3 = st.columns([2, 1, 1])
        niveis = f1.multiselect('Nível', ['verde', 'amarelo', 'vermelho'], default=['vermelho', 'amarelo'])
        so_alta = f2.checkbox('Só os que deram carga alta')
        busca = f3.text_input('Corrida', placeholder='ex.: O7172')
        sel = prev[prev.nivel.isin(niveis)]
        if so_alta:
            sel = sel[sel[C.ALVO] == 1]
        if busca:
            sel = sel[sel.cd_corrida.str.contains(busca.strip(), case=False)]
        sel = sel.sort_values('nota_risco', ascending=False)
        st.dataframe(tabela_resultados(sel), hide_index=True, height=360)
        st.download_button('Baixar tabela (Excel)', csv_para_excel(tabela_resultados(sel)),
                           'notas_mes_de_prova.csv', 'text/csv')


def ler_csv_enviado(arquivo):
    """Aceita CSV com vírgula ou ponto e vírgula, e vírgula decimal (padrão do Excel em português)."""
    bruto = arquivo.getvalue()
    df = pd.read_csv(io.BytesIO(bruto), sep=None, engine='python', encoding='utf-8-sig')
    for v in C.VARIAVEIS:
        if v in df and not pd.api.types.is_numeric_dtype(df[v]):
            texto = df[v].astype(str).str.strip()
            if texto.str.contains(',', regex=False).any():  # vírgula decimal (1.234,5)
                texto = texto.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[v] = pd.to_numeric(texto, errors='coerce')
    return df


def pagina_arquivo(m):
    st.title('Prever um arquivo de cestos')
    st.markdown('Envie um **CSV com uma linha por cesto** e as colunas dos 27 sinais '
                '(veja a página *Sobre o modelo*). Colunas a mais são mantidas; sinais que faltarem '
                'são preenchidos pela mediana do treino. Aceita CSV salvo pelo Excel em português.')
    modelo_csv = (Path(__file__).resolve().parent / 'exemplos' / 'cestos_exemplo.csv')
    if modelo_csv.exists():
        st.download_button('Baixar arquivo de exemplo', modelo_csv.read_bytes(), 'cestos_exemplo.csv', 'text/csv')
    arquivo = st.file_uploader('Arquivo CSV', type=['csv'])
    if arquivo is None:
        st.info('Nenhum arquivo enviado ainda.')
        return
    try:
        df = ler_csv_enviado(arquivo)
    except Exception as e:  # noqa: BLE001
        st.error(f'Não consegui ler o arquivo: {e}')
        return
    faltando = [v for v in C.VARIAVEIS if v not in df]
    if len(faltando) == len(C.VARIAVEIS):
        st.error('O arquivo não tem nenhuma das colunas esperadas. Confira os nomes na página *Sobre o modelo*.')
        return
    if faltando:
        st.warning(f'{len(faltando)} sinais não estão no arquivo e foram preenchidos pela mediana do treino: '
                   + ', '.join(faltando))
    out = df.copy()
    out['nota_risco'] = M.nota_de_risco(m, df)
    out['nivel'] = M.nivel(out.nota_risco)
    out['acao'] = out.nivel.map(C.ACOES)
    out['motivos'] = M.motivos(m, df)
    c = st.columns(4)
    c[0].metric('Cestos', br(len(out)))
    for i, n in enumerate(['verde', 'amarelo', 'vermelho'], start=1):
        c[i].metric(n.capitalize(), f'{int((out.nivel == n).sum())} ({(out.nivel == n).mean():.0%})')
    st.dataframe(tabela_resultados(out.sort_values('nota_risco', ascending=False)), hide_index=True)
    b1, b2 = st.columns(2)
    b1.download_button('Baixar resultado (Excel)', csv_para_excel(out), 'previsoes.csv', 'text/csv')
    b2.download_button('Baixar resultado (CSV padrão)', out.to_csv(index=False).encode('utf-8'),
                       'previsoes_padrao.csv', 'text/csv')


def pagina_desempenho():
    st.title('Desempenho do modelo')
    st.caption('Mesmo protocolo da apresentação: 5 provas ao longo do ano (aprende com o passado e prevê os '
               '~2 meses seguintes) e o mês de prova, nunca visto no treino. Todas as alternativas com o mesmo '
               'número de alertas (16,8% dos cestos; 59 no mês de prova).')
    av = avaliacao()
    comp = av['comparacao_alternativas']
    v = comp[comp.avaliacao.str.startswith('5 provas')].set_index('alternativa')
    mp = comp[comp.avaliacao == 'mês de prova'].set_index('alternativa')
    sol = 'Votação suave (solução)'
    c = st.columns(4)
    anterior = 'Modelo anterior (regressão logística)'
    c[0].metric('Paradas antecipadas no mês de prova',
                f'{int(mp.loc[sol, "paradas_antecipadas"])} de {int(mp.loc[sol, "paradas"])}',
                f'+{int(mp.loc[sol, "paradas_antecipadas"] - mp.loc[anterior, "paradas_antecipadas"])} sobre o anterior')
    c[1].metric('F1-score no mês de prova', br(mp.loc[sol, 'f1'], 2),
                f'+{br(mp.loc[sol, "f1"] - mp.loc[anterior, "f1"], 2)} sobre o anterior')
    c[2].metric('Ganho no mês de prova', f'R$ {br(mp.loc[sol, "reais_liquidos"] / 1000)} mil',
                help=f'{br(mp.loc[sol, "min_liquidos"])} minutos de forno, já descontados os alarmes falsos')
    c[3].metric('AUC no mês de prova', br(mp.loc[sol, 'auc'], 3))

    esq, dir_ = st.columns([1, 1], gap='large')
    with esq:
        st.subheader('Contra as alternativas')
        mostrar(grafico_comparacao(comp))
    with dir_:
        st.subheader('Tabela')
        tab = pd.DataFrame({
            'Alternativa': [NOMES_ALTERNATIVAS[a] for a in v.index],
            'F1 nas 5 provas': [br(x, 2) for x in v.f1],
            'F1 no mês de prova': [br(x, 2) for x in mp.loc[v.index, 'f1']],
            'Paradas antecipadas': [f'{int(a)} de {int(b)}' for a, b in zip(mp.loc[v.index, 'paradas_antecipadas'],
                                                                           mp.loc[v.index, 'paradas'])],
            'Minutos líquidos': [br(x) for x in mp.loc[v.index, 'min_liquidos']],
            'R$ no mês': [f'R$ {br(x / 1000)} mil' for x in mp.loc[v.index, 'reais_liquidos']],
        })
        st.dataframe(tab, hide_index=True)
        st.caption('Minutos líquidos: minutos de parada nos cestos alertados que de fato pararam, '
                   'menos 1 minuto por alarme falso. R$ 2.000 por minuto de forno.')

    st.subheader('Níveis de risco')
    niv = av['niveis_de_risco']
    e1, e2 = st.columns(2, gap='large')
    for col, periodo in [(e1, '5 provas'), (e2, 'mês de prova')]:
        with col:
            st.markdown(f'**{periodo.capitalize()}**')
            mostrar(grafico_niveis(niv[niv.periodo == periodo].reset_index(drop=True)))

    st.subheader('Economia agindo só no vermelho')
    eco = av['economia_anual']
    st.dataframe(pd.DataFrame({
        'Período': eco.periodo,
        'Alertas por dia': [br(x, 1) for x in eco.alertas_por_dia],
        'Paradas antecipadas por mês': [br(x) for x in eco.paradas_antecipadas_por_mes],
        'Horas de forno por ano': [br(x) for x in eco.horas_por_ano],
        'R$ por ano': [f'R$ {br(x / 1e6, 1)} mi' for x in eco.reais_por_ano],
        'R$ por ano (conservador)': [f'R$ {br(x / 1e6, 1)} mi' for x in eco.reais_por_ano_conservador],
    }), hide_index=True)
    st.caption('Alarmes falsos já descontados. Conservador: a ação no pátio evita só metade da parada.')


def pagina_sobre(meta):
    st.title('Sobre o modelo')
    st.markdown("""
**Votação suave**: dois modelos olham o mesmo cesto e a nota é a média das duas probabilidades × 100.

- **Regressão logística** (C = 0,5): dá um peso a cada sinal; estável ao longo do ano.
- **Gradient Boosting** (300 árvores): capta combinações entre sinais.

A nota **ordena os cestos por risco**; ela não é a chance de carga alta. O que vale é o nível.
""")
    st.dataframe(pd.DataFrame({
        'Nível': ['Verde', 'Amarelo', 'Vermelho'],
        'Nota': [f'abaixo de {C.CORTE_AMARELO}', f'{C.CORTE_AMARELO} a {C.CORTE_VERMELHO}',
                 f'{C.CORTE_VERMELHO} ou mais'],
        'Ação': [C.ACOES['verde'], C.ACOES['amarelo'], C.ACOES['vermelho']],
    }), hide_index=True)
    st.subheader('Os 27 sinais')
    grupo_de = {v: g for g, vs in GRUPOS.items() for v in vs}
    grupo_de.update(cesto2='Composição do cesto', dens_media='Composição do cesto')
    st.dataframe(pd.DataFrame({'Coluna no CSV': C.VARIAVEIS,
                               'Grupo': [grupo_de[v] for v in C.VARIAVEIS],
                               'Descrição': [C.DESCRICAO_VARIAVEIS[v] for v in C.VARIAVEIS]}),
                 hide_index=True, height=420)
    st.subheader('Modelo carregado')
    st.json({k: meta.get(k) for k in ['modelo', 'treinado_em', 'dados_de_treino', 'periodo_treino',
                                      'cestos_treino', 'versao_sklearn']})


# ---------------------------------------------------------------- navegação
PAGINAS = ['Simular um cesto', 'Mês de prova', 'Prever arquivo CSV', 'Desempenho do modelo', 'Sobre o modelo']


def main():
    with st.sidebar:
        st.markdown(f'<div style="font-weight:800;letter-spacing:0.3em;color:{COR["grafite"]}">E V C O M X</div>',
                    unsafe_allow_html=True)
        st.markdown('### Nota de risco de carga alta')
        pagina = st.radio('Página', PAGINAS, label_visibility='collapsed')
        st.divider()
        st.markdown(f'{selo("verde")} &nbsp;nota < {C.CORTE_AMARELO}<br><br>'
                    f'{selo("amarelo")} &nbsp;{C.CORTE_AMARELO} a {C.CORTE_VERMELHO}<br><br>'
                    f'{selo("vermelho")} &nbsp;≥ {C.CORTE_VERMELHO}', unsafe_allow_html=True)
        st.divider()
        st.caption('Projeto Final FA-BC · Bernardo, Fernando, Gabriela e Rafael')

    m, meta, aviso = modelo()
    if aviso:
        st.sidebar.info(aviso)
    prev = notas_mes_de_prova(m, meta.get('treinado_em'))

    if pagina == PAGINAS[0]:
        pagina_simular(m, prev)
    elif pagina == PAGINAS[1]:
        pagina_mes(prev)
    elif pagina == PAGINAS[2]:
        pagina_arquivo(m)
    elif pagina == PAGINAS[3]:
        pagina_desempenho()
    else:
        pagina_sobre(meta)


main()
