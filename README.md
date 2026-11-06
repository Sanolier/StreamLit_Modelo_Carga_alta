# Nota de risco de carga alta · pipeline com Docker

Projeto Final FA-BC · EVCOMX · Bernardo, Fernando, Gabriela e Rafael

Pipeline que reproduz, a partir dos arquivos brutos da usina, o modelo apresentado na
**solução para a carga alta**: uma **nota de risco de 0 a 100 para cada cesto**, calculada antes
de ele entrar no forno, com o nível (verde, amarelo ou vermelho), a ação no pátio e os motivos
do risco. Inclui um **app Streamlit** para usar o modelo pelo navegador e uma API HTTP.

## O modelo

**Votação suave**: dois modelos olham o mesmo cesto e a nota é a média das duas probabilidades × 100.

| Membro | Configuração |
|---|---|
| Regressão logística | mediana nos faltantes, padronização, `C=0.5`, `class_weight='balanced'` |
| Gradient Boosting | `HistGradientBoostingClassifier`, 300 árvores, `learning_rate=0.04`, `max_leaf_nodes=15`, `min_samples_leaf=30`, `l2_regularization=1.0`, `class_weight='balanced'` |

Os dois usam os mesmos **27 sinais** por cesto (lista e descrição em `carga_alta/config.py`):
o que tem no cesto (peso, volume, camadas, mistura, topo), como está o forno (ritmo, pausas,
cesto 1 da corrida) e o que houve antes (corrida anterior, últimas 20 corridas).

| Nível | Nota | Ação |
|---|---|---|
| Verde | abaixo de 45 | Carregar normalmente |
| Amarelo | 45 a 70 | Conferir o topo antes de içar e nivelar se sobrar, sem atrasar a carga |
| Vermelho | 70 ou mais | Parar 1 minuto: reacomodar o topo, tirar excesso de leve e avisar a sala do forno |

A nota ordena os cestos por risco; ela não é a chance de carga alta. O que vale é o nível.

## Como rodar com Docker

Pré-requisito: Docker Desktop (Windows/macOS) ou Docker Engine (Linux). Na pasta deste projeto:

**Opção 1 · tudo de uma vez (pipeline + API)**

```bash
docker compose up --build
```

Roda a pipeline completa, grava os resultados em `./saida` e depois sobe o **app Streamlit** em
<http://localhost:8501> e a API em <http://localhost:8000/saude>. `Ctrl+C` para parar.

**Opção 2 · só a pipeline**

```bash
docker build -t carga-alta:1.0 .
docker run --rm -v "$(pwd)/saida:/app/saida" carga-alta:1.0 tudo
```

No Windows (PowerShell) troque `$(pwd)` por `${PWD}`, ou dê dois cliques em `rodar_pipeline.bat`.
No Linux/macOS há também `./rodar_pipeline.sh`.

**Etapas avulsas** (mesma imagem, trocando o comando final):

| Comando | O que faz |
|---|---|
| `tudo` | preparar → treinar → avaliar → prever |
| `preparar` | brutos → uma linha por cesto com os 27 sinais |
| `treinar` | ajusta e salva a votação suave (período de treino, como na apresentação) |
| `treinar --todos` | retreino com todas as corridas (rotina mensal da implantação) |
| `avaliar` | 5 provas ao longo do ano + mês de prova, comparação com as alternativas |
| `prever --entrada /app/exemplos/cestos_exemplo.csv` | nota, nível, ação e motivos para cada cesto de um CSV |
| `servir --porta 8000` | API HTTP (publicar a porta com `-p 8000:8000`) |
| `app --porta 8501` | app Streamlit (publicar a porta com `-p 8501:8501`) |

Testes automáticos (pipeline inteira em pasta temporária e todas as páginas do app, ~40 s):

```bash
docker run --rm --entrypoint python carga-alta:1.0 -m unittest -v
```

> Se a pasta estiver no Google Drive (`G:\Meu Drive\...`) e o Docker não conseguir montar o volume,
> copie a pasta para um disco local (por exemplo `C:\entrega_carga_alta`) e rode de lá.

### Sem Docker

Python 3.11 ou mais novo (de preferência num ambiente separado, para não mexer nas versões do seu Anaconda):

```bash
python -m venv .venv            # Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m carga_alta tudo
streamlit run app_streamlit.py
```

No Windows, `rodar_app_sem_docker.bat` faz isso sozinho (cria o ambiente em `%LOCALAPPDATA%\carga_alta_venv`).

## App Streamlit

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/saida:/app/saida" carga-alta:1.0 app
```

Abra <http://localhost:8501>. No Windows, dê dois cliques em `rodar_app.bat` (com Docker) ou
`rodar_app_sem_docker.bat` (sem Docker). Na primeira abertura, se `./saida` estiver vazia, o app
prepara os dados, treina e avalia o modelo sozinho (cerca de 20 segundos).

| Página | O que mostra |
|---|---|
| Simular um cesto | Parte de um cesto real do mês de prova (ou de valores típicos); cada sinal pode ser alterado e a nota, o nível, a ação e o peso de cada sinal são recalculados na hora |
| Mês de prova | As notas dos 352 cestos ao longo do mês, a carga alta real em cada nível e a tabela filtrável, com download para Excel |
| Prever arquivo CSV | Envia um CSV de cestos (aceita o formato do Excel em português) e baixa as notas |
| Desempenho do modelo | Comparação com a regra simples e o modelo anterior, níveis de risco e economia anual |
| Sobre o modelo | Como a nota é calculada, os níveis e os 27 sinais esperados |

## Fluxo da pipeline

```
dados/brutos/infos_camadas.csv ─┐
dados/brutos/infos_materiais.csv ├─ preparar ─► saida/dados/cestos.csv ─ treinar ─► saida/modelo/
dados/brutos/VariavelResposta.csv┘                                   │                    │
                                                                       └─ avaliar ─► saida/avaliacao/
                                                                                         prever / servir ◄┘
```

| Etapa | Arquivo | Saídas |
|---|---|---|
| preparar | `carga_alta/preparar.py` | `saida/dados/camadas.csv`, `saida/dados/cestos.csv`, `saida/modelo/baia_usual.csv` |
| treinar | `carga_alta/treinar.py` | `saida/modelo/modelo_votacao.joblib`, `saida/modelo/metadados.json` |
| avaliar | `carga_alta/avaliar.py` | `saida/avaliacao/relatorio.md` e CSVs (comparação, provas, níveis, economia, previsões) |
| prever | `carga_alta/prever.py` | `saida/previsoes/previsoes.csv` |
| servir | `carga_alta/servir.py` | API HTTP |
| app | `app_streamlit.py` | App Streamlit |

Sem vazamento de informação do futuro: o histórico (corrida anterior, últimas 20 corridas, ritmo
e pausas) usa só corridas anteriores, e a baia usual de cada material é calculada só com o
período de treino.

## Avaliação

Mesmo protocolo da apresentação:

- **Mês de prova:** últimas 10% das corridas (180 corridas, 352 cestos, 57 paradas, 03/07 a 04/08/2026), nunca usadas no aprendizado.
- **5 provas ao longo do ano:** o treino é cortado em 6 fatias seguidas; cada prova aprende com as anteriores e prevê os ~2 meses seguintes (2.622 cestos).
- **Comparação justa:** todas as alternativas com o mesmo número de alertas (16,8% dos cestos; 59 no mês de prova).
- **Minutos líquidos:** minutos de parada nos cestos alertados que de fato pararam, menos 1 minuto por alarme falso, a R$ 2.000 por minuto.

Resultado gerado por esta pipeline (`saida/avaliacao/relatorio.md`):

| Alternativa | F1 nas 5 provas | F1 no mês de prova | Paradas antecipadas | Minutos líquidos | R$ no mês |
|---|---|---|---|---|---|
| Regra simples (corrida anterior parou) | 0,35 | 0,40 | 23 de 57 | 61 | R$ 123 mil |
| Modelo anterior (regressão logística) | 0,44 | 0,59 | 34 de 57 | 102 | R$ 204 mil |
| **Votação suave (solução)** | **0,45** | **0,66** | **38 de 57** | **117** | **R$ 234 mil** |

| Nível (5 provas) | % dos cestos | % com carga alta |
|---|---|---|
| Verde | 73% | 9% |
| Amarelo | 16% | 23% |
| Vermelho | 10% | 53% |

Agindo só no vermelho: cerca de 1 alerta por dia e 16 paradas antecipadas por mês, de 10 a 21 horas
de forno por ano (R$ 1,2 a 2,6 milhões), já descontados os alarmes falsos.

## Publicar no Streamlit Community Cloud

1. Suba esta pasta para um repositório no GitHub (com `dados/brutos`, `saida/modelo` e `.streamlit`).
2. Em <https://share.streamlit.io>, **Create app** → escolha o repositório e a branch `main`.
3. Em **Main file path**, use **`app_streamlit.py`**.
4. **Deploy**. O `requirements.txt` instala tudo; na primeira abertura o app prepara os dados (cerca de 20 s).

Se o app já tiver sido publicado com `carga_alta/__main__.py` como arquivo principal, ele também funciona:
quando é executado pelo Streamlit, esse arquivo abre o `app_streamlit.py`.

## API

```bash
curl http://localhost:8000/saude
curl http://localhost:8000/variaveis
curl -X POST http://localhost:8000/prever -H "Content-Type: application/json" -d @exemplos/requisicao.json
```

No PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/prever -ContentType "application/json" -InFile exemplos/requisicao.json
```

Resposta (um item por cesto):

```json
{
  "resultados": [
    {
      "cd_corrida": "O6904",
      "nu_carregamento": 2,
      "nota_risco": 72.0,
      "nivel": "vermelho",
      "acao": "Parar 1 minuto: reacomodar o topo, tirar excesso de leve e avisar a sala do forno",
      "motivos": "Volume estimado do cesto (m³): baixo (37.00); Fração do volume em material pesado: baixo (0.04); ..."
    }
  ],
  "avisos": []
}
```

Sinais não enviados são preenchidos pela mediana do treino e listados em `avisos`.

## Observações

- **Motivos** vêm da parte linear da votação (coeficiente × valor padronizado). Explicam o que o modelo usa para dar a nota; não provam causa.
- **Retreino mensal:** `treinar --todos` usa todas as corridas com resultado conhecido. Os cortes 45 e 70 devem ser revistos no período em paralelo.
- **Reprodutibilidade:** versões fixas em `requirements.txt` e `random_state=0`. A pipeline recalcula os sinais a partir dos brutos; diferenças numéricas mínimas em relação ao dataset intermediário do projeto (da ordem de 10⁻¹⁴) mudam a terceira casa de algumas métricas das 5 provas (F1 0,45 aqui, 0,44 na apresentação). Os números do mês de prova são idênticos.
- Materiais novos que não estejam no dicionário de classes são classificados pela densidade (`carga_alta/preparar.py`).
