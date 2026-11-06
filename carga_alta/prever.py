from pathlib import Path

import pandas as pd

from . import config as C
from . import modelo as M
from .preparar import carregar_cestos, dividir


def pontuar(modelo, cestos):
    out = cestos.copy()
    out['nota_risco'] = M.nota_de_risco(modelo, cestos)
    out['nivel'] = M.nivel(out.nota_risco)
    out['acao'] = out.nivel.map(C.ACOES)
    out['motivos'] = M.motivos(modelo, cestos)
    return out


def executar(entrada=None, saida=None):
    modelo, meta = M.carregar()
    if entrada:
        cestos = pd.read_csv(entrada)
        origem = entrada
    else:
        _, cestos, _ = dividir(carregar_cestos())
        origem = 'cestos do mês de prova (demonstração)'
    saida = Path(saida) if saida else C.PASTA_PREVISOES / 'previsoes.csv'
    saida.parent.mkdir(parents=True, exist_ok=True)

    out = pontuar(modelo, cestos)
    ids = [c for c in ['cd_corrida', 'nu_carregamento', 'dt_ini'] if c in out]
    extras = [c for c in [C.ALVO, C.DURACAO] if c in out]
    cols = ids + ['nota_risco', 'nivel', 'acao', 'motivos'] + extras
    out[cols + [c for c in out.columns if c not in cols]].to_csv(saida, index=False)

    print(f'[prever] {len(out)} cestos de: {origem}')
    print(f'[prever] modelo treinado com: {meta.get("dados_de_treino", "?")}')
    resumo = out.groupby('nivel').size().reindex(['verde', 'amarelo', 'vermelho'], fill_value=0)
    for n, q in resumo.items():
        linha = f'[prever]   {n:9s} {q:5d} cestos ({q / len(out):.0%})'
        if C.ALVO in out and out[C.ALVO].notna().any():
            taxa = out.loc[out.nivel == n, C.ALVO].mean()
            linha += f' · carga alta real em {taxa:.0%}' if q else ''
        print(linha)
    print(f'[prever] salvo em {saida}')
    return out
