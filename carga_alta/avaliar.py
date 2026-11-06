import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import modelo as M
from .preparar import carregar_cestos, dividir


# ---------------------------------------------------------------- alternativas comparadas
def regra_simples(a, b):
    return (b.y_ant.fillna(0) + b.taxa_ult20.fillna(a[C.ALVO].mean()) / 10).values


def com_modelo(criar):
    def prever(a, b):
        m = criar().fit(M.preparar_entrada(a[C.VARIAVEIS]), a[C.ALVO].astype(int))
        return m.predict_proba(M.preparar_entrada(b[C.VARIAVEIS]))[:, 1]
    return prever


ALTERNATIVAS = {
    'Regra simples (corrida anterior parou)': regra_simples,
    'Modelo anterior (regressão logística)': com_modelo(M.regressao_logistica),
    'Votação suave (solução)': com_modelo(M.criar_modelo),
}


def alertas_top(score, k):
    s = np.asarray(score, float) + np.random.default_rng(0).uniform(0, 1e-9, len(score))  # desempate fixo
    alerta = np.zeros(len(s), bool)
    alerta[np.argsort(-s, kind='stable')[:k]] = True
    return alerta


def resultado(y, seg, alerta):
    y, seg = np.asarray(y).astype(int), np.asarray(seg, float)
    vp = int((alerta & (y == 1)).sum()); fp = int((alerta & (y == 0)).sum())
    min_cob = seg[alerta & (y == 1)].sum() / 60
    return dict(cestos=len(y), paradas=int(y.sum()), alertas=int(alerta.sum()), paradas_antecipadas=vp,
                alarmes_falsos=fp, min_parada_total=seg.sum() / 60, min_cobertos=min_cob,
                min_liquidos=min_cob - C.CUSTO_ALARME_MIN * fp)


def resumir(r):
    r = dict(r)
    r['precisao'] = r['paradas_antecipadas'] / r['alertas'] if r['alertas'] else np.nan
    r['recall'] = r['paradas_antecipadas'] / r['paradas'] if r['paradas'] else np.nan
    r['f1'] = 2 * r['paradas_antecipadas'] / (r['alertas'] + r['paradas'])
    r['reais_liquidos'] = r['min_liquidos'] * C.REAIS_POR_MINUTO
    return r


def dias(b):
    return (b.dt_ini.max() - b.dt_ini.min()).total_seconds() / 86400


def somar(linhas):
    chaves = ['cestos', 'paradas', 'alertas', 'paradas_antecipadas', 'alarmes_falsos',
              'min_parada_total', 'min_cobertos', 'min_liquidos']
    return {k: sum(l[k] for l in linhas) for k in chaves}


def tabela_niveis(y, seg, nota, periodo):
    niv = M.nivel(nota)
    linhas = []
    for n in ['verde', 'amarelo', 'vermelho']:
        m = niv == n
        linhas.append(dict(periodo=periodo, nivel=n, cestos=int(m.sum()), frac_cestos=m.mean(),
                           taxa_carga_alta=np.asarray(y)[m].mean() if m.any() else np.nan,
                           min_parada_medio_por_cesto=np.asarray(seg)[m].mean() / 60 if m.any() else np.nan,
                           acao=C.ACOES[n]))
    return linhas


def economia_vermelho(y, seg, nota, n_dias, periodo):
    alerta = np.asarray(nota) >= C.CORTE_VERMELHO
    r = resultado(y, seg, alerta)
    fator = 365 / n_dias
    return dict(periodo=periodo, dias=round(n_dias, 1), alertas_vermelhos=r['alertas'],
                alertas_por_dia=r['alertas'] / n_dias, paradas_antecipadas=r['paradas_antecipadas'],
                paradas_antecipadas_por_mes=r['paradas_antecipadas'] / n_dias * 30,
                alarmes_falsos=r['alarmes_falsos'], min_liquidos=r['min_liquidos'],
                horas_por_ano=r['min_liquidos'] * fator / 60,
                reais_por_ano=r['min_liquidos'] * fator * C.REAIS_POR_MINUTO,
                reais_por_ano_conservador=(r['min_cobertos'] / 2 - C.CUSTO_ALARME_MIN * r['alarmes_falsos'])
                * fator * C.REAIS_POR_MINUTO)


def executar():
    C.PASTA_AVALIACAO.mkdir(parents=True, exist_ok=True)
    t = carregar_cestos()
    t = t[t[C.ALVO].notna()]
    treino, teste, provas = dividir(t)
    k_teste = int(round(C.FRACAO_ALERTAS * len(teste)))

    comp, por_prova, scores_val, score_teste = [], [], {}, {}
    for nome, prever in ALTERNATIVAS.items():
        linhas, aucs, aps, svs = [], [], [], []
        for i, (a, b) in enumerate(provas, start=1):
            s = prever(a, b)
            r = resultado(b[C.ALVO], b[C.DURACAO], alertas_top(s, int(round(C.FRACAO_ALERTAS * len(b)))))
            auc, ap = roc_auc_score(b[C.ALVO], s), average_precision_score(b[C.ALVO], s)
            por_prova.append(dict(alternativa=nome, prova=i,
                                  periodo=f'{b.dt_ini.min():%d/%m/%y} a {b.dt_ini.max():%d/%m/%y}',
                                  auc=auc, **resumir(r)))
            linhas.append(r); aucs.append(auc); aps.append(ap); svs.append(s)
        scores_val[nome] = np.concatenate(svs)
        val = resumir(somar(linhas))
        s = prever(treino, teste)
        score_teste[nome] = s
        tst = resumir(resultado(teste[C.ALVO], teste[C.DURACAO], alertas_top(s, k_teste)))
        tst.update(auc=roc_auc_score(teste[C.ALVO], s), ap=average_precision_score(teste[C.ALVO], s))
        comp.append(dict(alternativa=nome, avaliacao='5 provas (validação temporal)', auc=np.mean(aucs),
                         ap=np.mean(aps), **val))
        comp.append(dict(alternativa=nome, avaliacao='mês de prova', **tst))
        print(f'[avaliar] {nome:42s} F1 provas {val["f1"]:.2f} | mês de prova: F1 {tst["f1"]:.2f}, '
              f'{tst["paradas_antecipadas"]} de {tst["paradas"]} paradas, {tst["min_liquidos"]:.0f} min líquidos')

    comp = pd.DataFrame(comp)
    por_prova = pd.DataFrame(por_prova)


    sol = 'Votação suave (solução)'
    val_pool = pd.concat([b for _, b in provas])
    nota_val = scores_val[sol] * 100
    nota_teste = score_teste[sol] * 100
    niveis = pd.DataFrame(tabela_niveis(val_pool[C.ALVO], val_pool[C.DURACAO], nota_val, '5 provas')
                          + tabela_niveis(teste[C.ALVO], teste[C.DURACAO], nota_teste, 'mês de prova'))
    dias_val = sum(dias(b) for _, b in provas)
    economia = pd.DataFrame([economia_vermelho(val_pool[C.ALVO], val_pool[C.DURACAO], nota_val, dias_val, '5 provas'),
                             economia_vermelho(teste[C.ALVO], teste[C.DURACAO], nota_teste, dias(teste), 'mês de prova')])
    perda_atual = t[C.DURACAO].sum() / 60 * 365 / dias(t) * C.REAIS_POR_MINUTO

    prev = teste[['cd_corrida', 'nu_carregamento', 'dt_ini', C.ALVO, C.DURACAO]].copy()
    prev['nota_risco'] = np.round(nota_teste, 1)
    prev['nivel'] = M.nivel(nota_teste)
    prev['alerta_59'] = alertas_top(nota_teste, k_teste).astype(int)

    comp.to_csv(C.PASTA_AVALIACAO / 'comparacao_alternativas.csv', index=False)
    por_prova.to_csv(C.PASTA_AVALIACAO / 'provas_validacao.csv', index=False)
    niveis.to_csv(C.PASTA_AVALIACAO / 'niveis_de_risco.csv', index=False)
    economia.to_csv(C.PASTA_AVALIACAO / 'economia_anual.csv', index=False)
    prev.to_csv(C.PASTA_AVALIACAO / 'previsoes_mes_de_prova.csv', index=False)
    relatorio(comp, niveis, economia, perda_atual, t, teste, provas, k_teste)
    print(f'[avaliar] resultados em {C.PASTA_AVALIACAO}')
    return comp


def _br(x, casas=0):
    return f'{x:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def _mi(x):
    return f'R$ {_br(x / 1e6, 1)} mi'


def relatorio(comp, niveis, economia, perda_atual, t, teste, provas, k_teste):
    v = comp[comp.avaliacao == '5 provas (validação temporal)'].set_index('alternativa')
    m = comp[comp.avaliacao == 'mês de prova'].set_index('alternativa')
    L = ['# Avaliação do modelo de carga alta', '',
         'Modelo: **votação suave** (média das probabilidades de uma regressão logística e de um '
         'Gradient Boosting com 300 árvores), com 27 sinais por cesto.', '',
         f'- Dados: {t.cd_corrida.nunique():,} corridas, {len(t):,} cestos, '
         f'{t.dt_ini.min():%d/%m/%Y} a {t.dt_ini.max():%d/%m/%Y}.'.replace(',', '.'),
         f'- 5 provas ao longo do ano: {sum(len(b) for _, b in provas):,} cestos.'.replace(',', '.'),
         f'- Mês de prova: {len(teste)} cestos, {int(teste[C.ALVO].sum())} paradas, '
         f'{teste.dt_ini.min():%d/%m/%Y} a {teste.dt_ini.max():%d/%m/%Y}; {k_teste} alertas para todas as alternativas.',
         f'- Perda atual com carga alta: cerca de {_mi(perda_atual)} por ano (R$ {_br(C.REAIS_POR_MINUTO)} por minuto).',
         '', '## Comparação com as alternativas (mesmo número de alertas)', '',
         '| Alternativa | F1 nas 5 provas | F1 no mês de prova | AUC no mês de prova | Paradas antecipadas | Minutos líquidos | R$ no mês |',
         '|---|---|---|---|---|---|---|']
    for nome in v.index:
        L.append(f'| {nome} | {_br(v.loc[nome, "f1"], 2)} | {_br(m.loc[nome, "f1"], 2)} | {_br(m.loc[nome, "auc"], 3)} | '
                 f'{int(m.loc[nome, "paradas_antecipadas"])} de {int(m.loc[nome, "paradas"])} | '
                 f'{_br(m.loc[nome, "min_liquidos"])} | R$ {_br(m.loc[nome, "reais_liquidos"] / 1000)} mil |')
    L += ['', 'Minutos líquidos = minutos de parada nos cestos alertados que de fato pararam, menos 1 minuto '
          'por alarme falso.', '', '## Níveis de risco (nota de 0 a 100)', '',
          f'Verde: nota abaixo de {C.CORTE_AMARELO} · Amarelo: {C.CORTE_AMARELO} a {C.CORTE_VERMELHO} · '
          f'Vermelho: {C.CORTE_VERMELHO} ou mais.', '',
          '| Período | Nível | % dos cestos | % com carga alta | Min. de parada por cesto | Ação |', '|---|---|---|---|---|---|']
    for _, r in niveis.iterrows():
        L.append(f'| {r.periodo} | {r.nivel} | {_br(100 * r.frac_cestos)}% | {_br(100 * r.taxa_carga_alta)}% | '
                 f'{_br(r.min_parada_medio_por_cesto, 1)} | {r.acao} |')
    L += ['', '## Economia agindo só no vermelho (levada para um ano)', '',
          '| Período | Alertas por dia | Paradas antecipadas por mês | Horas de forno por ano | R$ por ano | R$ por ano (conservador) |',
          '|---|---|---|---|---|---|']
    for _, r in economia.iterrows():
        L.append(f'| {r.periodo} | {_br(r.alertas_por_dia, 1)} | {_br(r.paradas_antecipadas_por_mes)} | '
                 f'{_br(r.horas_por_ano)} | {_mi(r.reais_por_ano)} | {_mi(r.reais_por_ano_conservador)} |')
    L += ['', 'Conservador: a ação no pátio evita só metade da parada. Alarmes falsos já descontados.', '']
    (C.PASTA_AVALIACAO / 'relatorio.md').write_text('\n'.join(L), encoding='utf-8')
