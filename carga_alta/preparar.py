import numpy as np
import pandas as pd

from . import config as C

CLASSE_MATERIAL = {
    'CAVA': 'leve', 'ESBS': 'leve', 'ESPS': 'leve', 'MARP': 'leve', 'PACB': 'leve', 'RETL': 'leve', 'SHRE': 'leve',
    'CAFO': 'misto', 'ESPR': 'misto', 'MISB': 'misto', 'MISM': 'misto', 'OXIC': 'misto', 'PAEB': 'misto',
    'PAEP': 'misto',
    'FOFO': 'pesado', 'GUSL': 'pesado', 'GUSS': 'pesado', 'RECA': 'pesado', 'SUGU': 'pesado',
    '118441': 'pesado', '194032': 'pesado',
}


def classe_por_densidade(dens):
    if pd.isna(dens):
        return 'nao_classificado'
    return 'leve' if dens <= 0.6 else ('misto' if dens < 2.0 else 'pesado')


# ---------------------------------------------------------------- leitura dos brutos
def ler_materiais(pasta=C.DADOS_BRUTOS):
    m = pd.read_csv(pasta / C.ARQ_MATERIAIS, dtype=str, na_values=['null'])
    m.columns = ['cd_codigo_material', 'energia_kwh_t', 'densidade_t_m3', 'rendimento_metalico']
    for col in ['energia_kwh_t', 'densidade_t_m3', 'rendimento_metalico']:
        m[col] = pd.to_numeric(m[col].str.replace(',', '.', regex=False), errors='coerce')
    # linhas de insumos não metálicos (cal, coque) não têm densidade nem rendimento
    m = m.dropna(subset=['densidade_t_m3', 'rendimento_metalico'])
    return m.drop_duplicates('cd_codigo_material', keep='last').reset_index(drop=True)


def ler_camadas(pasta=C.DADOS_BRUTOS):
    c = pd.read_csv(pasta / C.ARQ_CAMADAS, na_values=['null'], dtype={'cd_codigo_material': str})
    c = c.drop_duplicates().reset_index(drop=True)
    c['nu_carregamento'] = c.nu_carregamento.astype(int)
    c['nu_camada'] = c.nu_camada.astype(int)
    c['cd_baia'] = pd.to_numeric(c.cd_baia, errors='coerce')
    c['dt_hora_consumo'] = pd.to_datetime(c.dt_hora_consumo, utc=True, errors='coerce')
    return c


def ler_resposta(pasta=C.DADOS_BRUTOS):
    return pd.read_csv(pasta / C.ARQ_RESPOSTA)


# ---------------------------------------------------------------- camadas enriquecidas
def enriquecer_camadas(cam, mat):
    c = cam.merge(mat, on='cd_codigo_material', how='left')
    c['peso_t'] = c.qt_peso_carregamento / 1000
    c['peso_zero'] = (c.qt_peso_carregamento == 0).astype(int)
    c['tipo_registro'] = np.where(c.densidade_t_m3.notna(), 'metalico', 'sem_dicionario')
    classe = c.cd_codigo_material.map(CLASSE_MATERIAL)
    sem_classe = classe.isna()
    classe[sem_classe] = c.loc[sem_classe, 'densidade_t_m3'].map(classe_por_densidade)
    c['classe'] = classe
    c['volume_m3'] = c.peso_t / c.densidade_t_m3
    c['energia_estimada_kwh'] = c.peso_t * c.energia_kwh_t
    c['massa_metalica_util_t'] = c.peso_t * c.rendimento_metalico
    chave = ['cd_corrida', 'nu_carregamento']
    c['pos_relativa'] = c.nu_camada / c.groupby(chave).nu_camada.transform('max')
    return c


# ---------------------------------------------------------------- uma linha por cesto
def tabela_por_cesto(cam, resp):
    """Agrega as camadas dos cestos 1 e 2 (o carregamento 3 é o gusa líquido) e cria o histórico."""
    chave = ['cd_corrida', 'nu_carregamento']
    c = cam[cam.nu_carregamento.isin([1, 2])].sort_values(chave + ['nu_camada'], kind='mergesort').copy()
    for k in ['leve', 'misto', 'pesado']:
        c['vol_' + k] = np.where(c.classe == k, c.volume_m3, 0)
    n_cam = c.groupby(chave).nu_camada.transform('count')
    pos = c.groupby(chave).cumcount()
    topo = pos >= n_cam - 3
    c['vol_topo'] = np.where(topo, c.volume_m3, 0)
    c['vol_leve_topo'] = np.where(topo & (c.classe == 'leve'), c.volume_m3, 0)
    c['vol_inf'] = np.where(pos < n_cam / 2, c.volume_m3, 0)

    t = c.groupby(chave).agg(
        pk=('pk_aci_corrida', 'first'), n_camadas=('nu_camada', 'count'), peso_t=('peso_t', 'sum'),
        volume_m3=('volume_m3', 'sum'), n_zero=('peso_zero', 'sum'), n_materiais=('cd_codigo_material', 'nunique'),
        vol_leve=('vol_leve', 'sum'), vol_misto=('vol_misto', 'sum'), vol_pesado=('vol_pesado', 'sum'),
        vol_topo=('vol_topo', 'sum'), vol_leve_topo=('vol_leve_topo', 'sum'), vol_inf=('vol_inf', 'sum'),
        max_vol_camada=('volume_m3', 'max'), dt_ini=('dt_hora_consumo', 'min'), dt_fim=('dt_hora_consumo', 'max'),
    ).reset_index()
    t['dens_media'] = t.peso_t / t.volume_m3
    for k in ['leve', 'misto', 'pesado']:
        t['frac_vol_' + k] = t['vol_' + k] / t.volume_m3
    t['frac_vol_topo'] = t.vol_topo / t.volume_m3
    t['frac_vol_inf'] = t.vol_inf / t.volume_m3
    t['dur_carga_min'] = (t.dt_fim - t.dt_ini).dt.total_seconds() / 60

    # alvo e duração da parada de cada cesto
    rot = pd.concat([resp[['cd_corrida']].assign(
        nu_carregamento=k, **{C.ALVO: resp[f'is_carga_alta_carregamento_{k}'],
                              C.DURACAO: resp[f'qt_segundos_duracao_parada_carregamento_{k}']}) for k in [1, 2]])
    t = t.merge(rot, on=chave, how='left')

    # o que o cesto 1 deixou no forno (só faz sentido para o cesto 2)
    c1 = t[t.nu_carregamento == 1].set_index('cd_corrida')
    e_c2 = t.nu_carregamento == 2
    t['vol_c1'] = t.cd_corrida.map(c1.volume_m3).where(e_c2, 0)
    t['peso_c1'] = t.cd_corrida.map(c1.peso_t).where(e_c2, 0)
    t['y_c1'] = t.cd_corrida.map(c1[C.ALVO]).where(e_c2, 0)

    # ordem das corridas (pela chave da corrida) e calendário
    cor = t[t.nu_carregamento == 1][['cd_corrida', 'pk']].sort_values('pk').reset_index(drop=True)
    cor['ordem'] = np.arange(len(cor))
    t = t.merge(cor[['cd_corrida', 'ordem']], on='cd_corrida')
    local = t.dt_ini.dt.tz_convert(C.FUSO)
    t['hora'] = local.dt.hour
    t['dia_sem'] = local.dt.dayofweek
    t['cesto2'] = (t.nu_carregamento == 2).astype(int)

    # histórico do mesmo cesto: só corridas anteriores
    t = t.sort_values(['nu_carregamento', 'ordem'])
    t['y_ant'] = t.groupby('nu_carregamento')[C.ALVO].shift(1)
    t['taxa_ult20'] = t.groupby('nu_carregamento')[C.ALVO].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    return t.sort_values(['ordem', 'nu_carregamento']).reset_index(drop=True)


def corte_teste(t):
    """Ordem da primeira corrida do mês de prova (últimas 10% das corridas)."""
    return int((t.ordem.max() + 1) * (1 - C.FRACAO_TESTE))


def ritmo_e_pausas(t):
    """Ritmo do forno e pausas, olhando só corridas anteriores (horário de início do cesto 1)."""
    cor = t[t.nu_carregamento == 1][['cd_corrida', 'ordem', 'dt_ini']].sort_values('ordem').reset_index(drop=True)
    cor['dt_ini'] = cor.dt_ini.ffill()  # corridas sem horário herdam o da anterior
    cor['horas_desde_anterior'] = cor.dt_ini.diff().dt.total_seconds() / 3600
    cor['pausa_longa'] = (cor.horas_desde_anterior > 6).astype(float).where(cor.horas_desde_anterior.notna())
    ts = cor.dt_ini.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy().astype('datetime64[ns]')
    cor['corridas_24h'] = np.arange(len(ts)) - np.searchsorted(ts, ts - np.timedelta64(24, 'h'), side='left')
    return t.merge(cor[['cd_corrida', 'horas_desde_anterior', 'pausa_longa', 'corridas_24h']],
                   on='cd_corrida', how='left')


def calcular_baia_usual(cam, t, corte):
    """Baia mais frequente de cada material, usando só as corridas de treino."""
    ordem = t.drop_duplicates('cd_corrida').set_index('cd_corrida').ordem
    b = cam[cam.nu_carregamento.isin([1, 2]) & cam.cd_baia.notna()].copy()
    b['ordem'] = b.cd_corrida.map(ordem)
    b = b[b.ordem < corte]
    return (b.groupby('cd_codigo_material').cd_baia.agg(lambda s: s.mode().iloc[0])
            .rename('baia_usual').reset_index())


def fora_da_baia(cam, t, baia_usual):
    b = cam[cam.nu_carregamento.isin([1, 2]) & cam.cd_baia.notna()].copy()
    usual = b.cd_codigo_material.map(baia_usual.set_index('cd_codigo_material').baia_usual)
    b['fora'] = (b.cd_baia != usual).astype(float)  # material sem baia usual conta como fora
    f = b.groupby(['cd_corrida', 'nu_carregamento']).fora.mean().rename('frac_fora_baia').reset_index()
    return t.merge(f, on=['cd_corrida', 'nu_carregamento'], how='left')


def montar_cestos(cam, resp, baia_usual=None):
    """Pipeline completa de variáveis. Se baia_usual não for dada, é calculada no período de treino."""
    t = tabela_por_cesto(cam, resp)
    t = ritmo_e_pausas(t)
    if baia_usual is None:
        baia_usual = calcular_baia_usual(cam, t, corte_teste(t))
    t = fora_da_baia(cam, t, baia_usual)
    t = t.sort_values(['ordem', 'nu_carregamento']).reset_index(drop=True)
    t['dt_ini'] = t.dt_ini.dt.tz_convert(C.FUSO)
    t['dt_fim'] = t.dt_fim.dt.tz_convert(C.FUSO)
    ids = ['cd_corrida', 'nu_carregamento', 'pk', 'ordem', 'dt_ini', 'dt_fim', C.ALVO, C.DURACAO]
    return t[ids + C.VARIAVEIS], baia_usual


def executar():
    C.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    C.PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    mat = ler_materiais()
    cam = enriquecer_camadas(ler_camadas(), mat)
    resp = ler_resposta()
    cestos, baia_usual = montar_cestos(cam, resp)

    cam.to_csv(C.ARQ_CAMADAS_PROC, index=False)
    cestos.to_csv(C.ARQ_CESTOS, index=False)
    baia_usual.to_csv(C.ARQ_BAIA_USUAL, index=False)

    corte = corte_teste(cestos)
    print(f'[preparar] {len(cam):,} camadas, {cestos.cd_corrida.nunique():,} corridas, {len(cestos):,} cestos')
    print(f'[preparar] carga alta em {cestos[C.ALVO].mean():.1%} dos cestos; '
          f'treino = corridas 0 a {corte - 1}, mês de prova = corridas {corte} a {cestos.ordem.max()}')
    print(f'[preparar] salvo em {C.ARQ_CESTOS}')
    return cestos


# ---------------------------------------------------------------- uso pelas outras etapas
def carregar_cestos():
    """Lê cestos.csv; se ainda não existir, roda a preparação."""
    if not C.ARQ_CESTOS.exists():
        return executar()
    t = pd.read_csv(C.ARQ_CESTOS, dtype={'cd_corrida': str})
    t['dt_ini'] = pd.to_datetime(t.dt_ini, utc=True).dt.tz_convert(C.FUSO)
    return t.sort_values(['ordem', 'nu_carregamento']).reset_index(drop=True)


def dividir(t):
    """Treino (90% iniciais das corridas), mês de prova (10% finais) e as 5 provas de validação
    temporal: o treino é cortado em 6 fatias seguidas; a prova i aprende com as fatias anteriores
    e prevê a fatia i (cerca de 2 meses)."""
    corte = corte_teste(t)
    treino, teste = t[t.ordem < corte], t[t.ordem >= corte]
    blocos = np.array_split(np.arange(corte), C.N_BLOCOS)
    provas = [(treino[treino.ordem.isin(np.concatenate(blocos[:i]))], treino[treino.ordem.isin(blocos[i])])
              for i in range(1, C.N_BLOCOS)]
    return treino, teste, provas
