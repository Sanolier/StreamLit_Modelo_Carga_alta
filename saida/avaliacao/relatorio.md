# Avaliação do modelo de carga alta

Modelo: **votação suave** (média das probabilidades de uma regressão logística e de um Gradient Boosting com 300 árvores), com 27 sinais por cesto.

- Dados: 1.799 corridas. 3.501 cestos. 14/08/2025 a 04/08/2026.
- 5 provas ao longo do ano: 2.622 cestos.
- Mês de prova: 352 cestos, 57 paradas, 03/07/2026 a 04/08/2026; 59 alertas para todas as alternativas.
- Perda atual com carga alta: cerca de R$ 4,5 mi por ano (R$ 2.000 por minuto).

## Comparação com as alternativas (mesmo número de alertas)

| Alternativa | F1 nas 5 provas | F1 no mês de prova | AUC no mês de prova | Paradas antecipadas | Minutos líquidos | R$ no mês |
|---|---|---|---|---|---|---|
| Regra simples (corrida anterior parou) | 0,35 | 0,40 | 0,819 | 23 de 57 | 61 | R$ 123 mil |
| Modelo anterior (regressão logística) | 0,44 | 0,59 | 0,869 | 34 de 57 | 102 | R$ 204 mil |
| Votação suave (solução) | 0,45 | 0,66 | 0,887 | 38 de 57 | 117 | R$ 234 mil |

Minutos líquidos = minutos de parada nos cestos alertados que de fato pararam, menos 1 minuto por alarme falso.

## Níveis de risco (nota de 0 a 100)

Verde: nota abaixo de 45 · Amarelo: 45 a 70 · Vermelho: 70 ou mais.

| Período | Nível | % dos cestos | % com carga alta | Min. de parada por cesto | Ação |
|---|---|---|---|---|---|
| 5 provas | verde | 73% | 9% | 0,4 | Carregar normalmente |
| 5 provas | amarelo | 16% | 23% | 0,9 | Conferir o topo antes de içar e nivelar se sobrar, sem atrasar a carga |
| 5 provas | vermelho | 10% | 53% | 2,1 | Parar 1 minuto: reacomodar o topo, tirar excesso de leve e avisar a sala do forno |
| mês de prova | verde | 68% | 4% | 0,1 | Carregar normalmente |
| mês de prova | amarelo | 13% | 18% | 0,4 | Conferir o topo antes de içar e nivelar se sobrar, sem atrasar a carga |
| mês de prova | vermelho | 19% | 57% | 2,1 | Parar 1 minuto: reacomodar o topo, tirar excesso de leve e avisar a sala do forno |

## Economia agindo só no vermelho (levada para um ano)

| Período | Alertas por dia | Paradas antecipadas por mês | Horas de forno por ano | R$ por ano | R$ por ano (conservador) |
|---|---|---|---|---|---|
| 5 provas | 1,0 | 16 | 10 | R$ 1,2 mi | R$ 0,4 mi |
| mês de prova | 2,2 | 37 | 21 | R$ 2,6 mi | R$ 1,0 mi |

Conservador: a ação no pátio evita só metade da parada. Alarmes falsos já descontados.
