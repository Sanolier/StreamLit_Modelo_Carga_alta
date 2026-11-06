from datetime import datetime, timezone

from . import config as C
from . import modelo as M
from .preparar import carregar_cestos, dividir


def executar(todos=False):
    t = carregar_cestos()
    treino, teste, _ = dividir(t)
    base = t if todos else treino
    base = base[base[C.ALVO].notna()]

    modelo = M.criar_modelo().fit(M.preparar_entrada(base[C.VARIAVEIS]), base[C.ALVO].astype(int))
    meta = dict(
        modelo='Votação suave: regressão logística (C=0,5) + Gradient Boosting (300 árvores)',
        treinado_em=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        dados_de_treino='todas as corridas' if todos else 'período de treino (90% iniciais das corridas)',
        cestos_treino=int(len(base)), corridas_treino=int(base.cd_corrida.nunique()),
        taxa_carga_alta_treino=round(float(base[C.ALVO].mean()), 4),
        periodo_treino=f'{base.dt_ini.min():%d/%m/%Y} a {base.dt_ini.max():%d/%m/%Y}',
        periodo_mes_de_prova=f'{teste.dt_ini.min():%d/%m/%Y} a {teste.dt_ini.max():%d/%m/%Y}',
        primeira_corrida_mes_de_prova=int(teste.ordem.min()),
    )
    M.salvar(modelo, meta)
    print(f'[treinar] {meta["modelo"]}')
    print(f'[treinar] {meta["cestos_treino"]:,} cestos ({meta["periodo_treino"]}), '
          f'carga alta em {meta["taxa_carga_alta_treino"]:.1%}')
    print(f'[treinar] modelo salvo em {C.ARQ_MODELO}')
    return modelo
