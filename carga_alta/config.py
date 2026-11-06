import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_BRUTOS = Path(os.environ.get('DADOS_BRUTOS', RAIZ / 'dados' / 'brutos'))
SAIDA = Path(os.environ.get('SAIDA', RAIZ / 'saida'))

# ---------------------------------------------------------------- arquivos de entrada
ARQ_CAMADAS = 'infos_camadas.csv'
ARQ_MATERIAIS = 'infos_materiais.csv'
ARQ_RESPOSTA = 'VariavelResposta.csv'

# ---------------------------------------------------------------- arquivos de saída
PASTA_DADOS = SAIDA / 'dados'
PASTA_MODELO = SAIDA / 'modelo'
PASTA_AVALIACAO = SAIDA / 'avaliacao'
PASTA_PREVISOES = SAIDA / 'previsoes'

ARQ_CAMADAS_PROC = PASTA_DADOS / 'camadas.csv'
ARQ_CESTOS = PASTA_DADOS / 'cestos.csv'
ARQ_BAIA_USUAL = PASTA_MODELO / 'baia_usual.csv'
ARQ_MODELO = PASTA_MODELO / 'modelo_votacao.joblib'
ARQ_METADADOS = PASTA_MODELO / 'metadados.json'

# ---------------------------------------------------------------- problema
ALVO = 'is_carga_alta_cesto'          # 1 = o cesto deu carga alta
DURACAO = 'segundos_parada_cesto'     # segundos de forno parado por carga alta naquele cesto
FUSO = 'America/Sao_Paulo'            # mesmo fuso usado no projeto (UTC-3)

# 27 sinais usados pela votação suave (mesmos do modelo anterior do projeto)
VARIAVEIS = [
    # composição do cesto
    'cesto2', 'n_camadas', 'peso_t', 'volume_m3', 'dens_media', 'n_zero', 'n_materiais',
    'frac_vol_leve', 'frac_vol_misto', 'frac_vol_pesado', 'max_vol_camada',
    # topo do cesto
    'vol_topo', 'frac_vol_topo', 'vol_leve_topo', 'frac_vol_inf',
    # o que o cesto 1 deixou no forno
    'vol_c1', 'peso_c1', 'y_c1',
    # horário e duração da montagem
    'dur_carga_min', 'hora', 'dia_sem',
    # histórico recente
    'y_ant', 'taxa_ult20',
    # ritmo do forno, pausas e baia
    'horas_desde_anterior', 'pausa_longa', 'corridas_24h', 'frac_fora_baia',
]

DESCRICAO_VARIAVEIS = {
    'cesto2': 'É o cesto 2 da corrida (1) ou o cesto 1 (0)',
    'n_camadas': 'Número de camadas do cesto',
    'peso_t': 'Peso total do cesto (t)',
    'volume_m3': 'Volume estimado do cesto (m³)',
    'dens_media': 'Densidade média do cesto (t/m³)',
    'n_zero': 'Camadas com peso zero',
    'n_materiais': 'Número de materiais diferentes',
    'frac_vol_leve': 'Fração do volume em material leve',
    'frac_vol_misto': 'Fração do volume em material misto',
    'frac_vol_pesado': 'Fração do volume em material pesado',
    'max_vol_camada': 'Maior volume de uma camada (m³)',
    'vol_topo': 'Volume das 3 camadas do topo (m³)',
    'frac_vol_topo': 'Fração do volume nas 3 camadas do topo',
    'vol_leve_topo': 'Volume de material leve no topo (m³)',
    'frac_vol_inf': 'Fração do volume na metade inferior',
    'vol_c1': 'Volume do cesto 1 da mesma corrida (só para o cesto 2)',
    'peso_c1': 'Peso do cesto 1 da mesma corrida (só para o cesto 2)',
    'y_c1': 'O cesto 1 da mesma corrida deu carga alta (só para o cesto 2)',
    'dur_carga_min': 'Duração da montagem do cesto (min)',
    'hora': 'Hora de início da montagem (0-23)',
    'dia_sem': 'Dia da semana (0 = segunda, 6 = domingo)',
    'y_ant': 'Corrida anterior deu carga alta neste cesto',
    'taxa_ult20': 'Taxa de carga alta nas últimas 20 corridas (mesmo cesto)',
    'horas_desde_anterior': 'Horas desde o início da corrida anterior',
    'pausa_longa': 'Pausa de mais de 6 h desde a corrida anterior',
    'corridas_24h': 'Corridas iniciadas nas 24 h anteriores (ritmo do forno)',
    'frac_fora_baia': 'Fração das camadas com material fora da baia usual',
}

# ---------------------------------------------------------------- divisão temporal
FRACAO_TESTE = 0.10   # últimas 10% das corridas = mês de prova (nunca usado no treino)
N_BLOCOS = 6          # treino dividido em 6 fatias -> 5 provas de validação temporal

# ---------------------------------------------------------------- operação
N_ALERTAS_TESTE = 59                  # orçamento de alertas do projeto no mês de prova
FRACAO_ALERTAS = N_ALERTAS_TESTE / 352  # 16,8% dos cestos
CUSTO_ALARME_MIN = 1.0                # cada alarme falso custa 1 minuto
REAIS_POR_MINUTO = 2000               # R$ por minuto de forno parado

# Níveis de risco sobre a nota (0 a 100)
CORTE_AMARELO = 45
CORTE_VERMELHO = 70
ACOES = {
    'verde': 'Carregar normalmente',
    'amarelo': 'Conferir o topo antes de içar e nivelar se sobrar, sem atrasar a carga',
    'vermelho': 'Parar 1 minuto: reacomodar o topo, tirar excesso de leve e avisar a sala do forno',
}
