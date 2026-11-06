import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def regressao_logistica():
    """Modelo anterior do projeto: mediana nos faltantes, padronização e regressão logística."""
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight='balanced', max_iter=5000))


def gradient_boosting():
    """300 árvores rasas; trata faltantes sozinho."""
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                          l2_regularization=1.0, class_weight='balanced', random_state=0)


def criar_modelo():
    return VotingClassifier([('lr', regressao_logistica()), ('hgb', gradient_boosting())], voting='soft')


# ---------------------------------------------------------------- nota, nível e motivos
def nota_de_risco(modelo, X):
    X = preparar_entrada(X)
    return np.round(modelo.predict_proba(X)[:, 1] * 100, 1)


def nivel(nota):
    nota = np.asarray(nota, float)
    return np.select([nota >= C.CORTE_VERMELHO, nota >= C.CORTE_AMARELO], ['vermelho', 'amarelo'], 'verde')


def preparar_entrada(X):
    """Garante as 27 colunas na ordem do treino; colunas ausentes viram faltantes (o modelo imputa)."""
    X = pd.DataFrame(X).copy()
    for col in C.VARIAVEIS:
        if col not in X:
            X[col] = np.nan
    return X[C.VARIAVEIS].apply(pd.to_numeric, errors='coerce').astype(float)


BINARIAS = {'cesto2', 'y_c1', 'y_ant', 'pausa_longa'}
SO_CESTO2 = {'vol_c1', 'peso_c1', 'y_c1'}   # valem 0 no cesto 1 por construção: não explicam nada nele


def motivos(modelo, X, n=3):
    """Sinais que mais elevam o risco de cada cesto, pela parte linear da votação
    (coeficiente x valor padronizado), com o sentido do sinal (alto/baixo, sim/não).
    Serve para explicar o alerta ao operador; não é uma prova de causa."""
    X = preparar_entrada(X)
    lr = modelo.named_estimators_['lr']
    z = lr[:-1].transform(X)
    contrib = z * lr[-1].coef_[0]
    valores = lr[0].transform(X)  # valores com faltantes preenchidos pela mediana
    i_c2 = C.VARIAVEIS.index('cesto2')
    saida = []
    for linha, zl, val in zip(contrib, z, valores):
        textos = []
        for i in np.argsort(-linha):
            nome = C.VARIAVEIS[i]
            if linha[i] <= 0 or len(textos) == n:
                break
            if nome in SO_CESTO2 and val[i_c2] == 0:
                continue
            desc = C.DESCRICAO_VARIAVEIS[nome]
            if nome in BINARIAS:
                textos.append(f'{desc}: {"sim" if val[i] >= 0.5 else "não"}')
            else:
                textos.append(f'{desc}: {"alto" if zl[i] > 0 else "baixo"} ({val[i]:.2f})')
        saida.append('; '.join(textos))
    return saida


# ---------------------------------------------------------------- salvar e carregar
def salvar(modelo, metadados):
    C.PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, C.ARQ_MODELO)
    metadados = dict(metadados, versao_sklearn=sklearn.__version__, variaveis=C.VARIAVEIS,
                     corte_amarelo=C.CORTE_AMARELO, corte_vermelho=C.CORTE_VERMELHO)
    C.ARQ_METADADOS.write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding='utf-8')


def carregar():
    if not C.ARQ_MODELO.exists():
        raise FileNotFoundError(f'Modelo não encontrado em {C.ARQ_MODELO}. Rode antes: python -m carga_alta treinar')
    meta = json.loads(C.ARQ_METADADOS.read_text(encoding='utf-8')) if C.ARQ_METADADOS.exists() else {}
    return joblib.load(C.ARQ_MODELO), meta
