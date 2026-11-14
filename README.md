# SVM na base Heart Failure Prediction

Trabalho da disciplina Aprendizado de Máquina (PPGCC - UFPI). Um SVM prevê **doença cardíaca**
(`HeartDisease`) a partir de 11 atributos clínicos, com experimentos organizados em rodadas,
como nos projetos `DecisionTree`, `RandomForest` e `Glass-Identification-Neural-Network`.

- Base: [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
  (fedesoriano). 918 pacientes de 5 bases clássicas (Cleveland, Hungarian, Switzerland, Long
  Beach VA, Statlog). Alvo binário, ~55% positivos.
- Atenção ao nome: o alvo é doença cardíaca, não infarto especificamente.

## 1. Preparar

```bash
pip install -r requirements.txt
# baixar o heart.csv no Kaggle e colocar em data/raw/heart.csv
python -m src.make_split
```

O `make_split` gera os dois splits (estratificados, seed 42) e mostra os problemas da base:

| Split | Uso |
|---|---|
| `data/split/70_30/` | **principal**: todas as rodadas (R0-R10) e o teste final |
| `data/split/80_20/` | **só o R11**: a config final é refeita aqui para ver se o resultado muda |

O 80/20 não escolhe nada. Escolher o split pelo melhor teste seria usar o teste para decidir.

## 2. Rotina de cada rodada

```bash
# 1. trocar as linhas "# ATUALIZAR" do YAML pelos vencedores anteriores
python run_experiments.py experiments/r1_escalonamento.yaml
# 2. olhar tudo
python summarize.py
# 3. fechar a rodada (teste t corrigido de Nadeau e Bengio, um contra todos)
python compare.py --rodada R1_ --salvar
python compare.py --rodada R1_ --metric recall --salvar
# 4. anotar a decisão em docs/anotacoes_artigo.md; o vencedor vira o baseline da próxima
```

Regra de decisão (igual aos trabalhos anteriores): entre as configs **sem diferença
significativa** para a melhor, fica a **mais simples**. A ordem de simplicidade de cada rodada
está no cabeçalho do YAML.

## 3. Protocolo

- **CV:** `RepeatedStratifiedKFold` 5 folds x 5 repetições, seed 42, sempre os mesmos folds
  (as execuções são comparadas fold a fold).
- **Métricas:** `roc_auc` é a principal; `recall` (sensibilidade) é a segunda, porque deixar
  de identificar um doente (falso negativo) é o erro mais caro. Também: specificity, accuracy,
  precision, f1, balanced_accuracy e, depois da calibração, brier e log_loss.
- **Sem vazamento:** tudo é etapa do pipeline e aprende só no treino de cada fold (imputação,
  one-hot, escalonamento, seleção, limiar, calibração). O teste só é lido pelo `final_test.py`.

## 4. O pipeline

Toda execução passa pelas mesmas etapas, nesta ordem. Cada uma recebe uma opção; `none` pula.

| Etapa | Opções | O que faz |
|---|---|---|
| `zeros` | none, **to_nan** | Cholesterol = 0 (~19%) e RestingBP = 0 (1) viram ausentes |
| `missing_flags` | **none**, flags | acrescenta `Cholesterol_missing` ("was_missing") |
| `imputer` | **median**, median_by_group, knn_scaled, iterative, none | preenche os ausentes (iterative = MICE). Grupo: `Sex` ou `age_band`, nunca o alvo |
| `features` | **none**, oldpeak_abs, st_elevation | Oldpeak negativo (R2) |
| `drop_cols` | **none**, cholesterol | tira a coluna Cholesterol (R2_atalho; a flag, se houver, fica) |
| `encoding` | **onehot** | categóricas viram colunas 0/1 (obrigatória) |
| `scaler` | none, **standard**, minmax, robust | só nas contínuas (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) |
| `selection` | **none**, rfe_linear, kbest | seleção de colunas (R5) |
| `model` | **svm**, logreg, gboost, dummy | com os hiperparâmetros no YAML |
| `calibration` (opcional) | sigmoid, isotonic | probabilidades (R8) |
| `threshold` (opcional) | tuned (scoring: f1, f2, balanced_accuracy) | limiar de decisão (R6) |

Em negrito, o ponto de partida. Exemplo de config:

```yaml
baseline:
  zeros: to_nan
  missing_flags: flags
  imputer: {name: median_by_group, group: Sex}
  features: none
  drop_cols: none
  encoding: onehot
  scaler: standard
  selection: none
  model: {name: svm, kernel: rbf, C: 10, gamma: 0.01}
  threshold: {name: tuned, scoring: f2}
```

`sweep` gera uma execução por combinação: `{model.C: [0.1, 1, 10], model.gamma: [0.01, 0.1]}`
gera 6. O `override` **substitui a etapa inteira** (para mudar só o C, repita o model todo).

## 5. Rodadas

| Rodada | Pergunta | Arquivo / script |
|---|---|---|
| R0 | O que se consegue sem o SVM? (dummy, regressão logística, gradient boosting) | `r0_baselines.yaml` |
| R1 | Escalonar importa? | `r1_escalonamento.yaml` |
| R2 | Como tratar os zeros? A flag é um atalho? E o Oldpeak negativo? | `r2_dados.yaml` |
| R3 | Qual kernel? (só elimina os piores) | `r3_kernel.yaml` |
| R4 | C x gamma, grau; escolha final do kernel | `r4_hiperparametros.yaml` + `heatmap_c_gamma.py` |
| R5 | Menos colunas ajuda? | `r5_selecao.yaml` |
| R6 | Como subir o recall? (peso de classe, limiar) | `r6_recall.yaml` |
| R7 | Tudo junto + ablação + reajuste de C/gamma | `r7_combinacao.yaml` |
| R8 | Probabilidades calibradas | `r8_calibracao.yaml` |
| R9 | O que o modelo usa? (teste de sanidade) | `importances_permutation.csv` e `coeficientes_lineares.csv` da execução |
| R10 | Robustez: outras sementes da CV e subgrupos | `robustez.py --run <vencedor>` |
| Final | Vencedor e baselines no teste 70/30 | `final_test.py --run <vencedor> --baselines ...` |
| R11 | O resultado muda no 80/20? | `r11_split_80_20.yaml` + `final_test.py` + `split_check.py` |

**R2, o atalho do colesterol:** Cholesterol = 0 tem 88% de doentes, contra 48% nos
registrados, porque os zeros vêm quase todos de um hospital (Suíça). Uma flag de ausência pode
subir o roc_auc ensinando "veio daquele hospital". O `R2_atalho` separa os efeitos (sem a
coluna; só a flag) e a regra de leitura está no YAML e em `docs/anotacoes_artigo.md`.

**R9, teste de sanidade:** na literatura e na cardiologia, ST_Slope, ChestPainType e Oldpeak
são os sinais mais fortes nesta base. Se as importâncias da execução escolhida não se parecerem
com isso, procure vazamento ou erro de codificação antes de seguir.

## 6. Figura para a apresentação: fronteira em 2D

```bash
python fronteira_2d.py --run <pasta>                    # hiperparâmetros da execução
python fronteira_2d.py --run <pasta> --C 0.1 1 100      # efeito do C (margem)
python fronteira_2d.py --run <pasta> --gamma 0.01 0.1 1 # efeito do gamma (curvatura)
```

Projeta os pacientes do treino em 2 componentes principais e treina um SVM nesse plano:
fronteira, margens e vetores de suporte. É uma **ilustração** (2 componentes guardam ~40% da
variância): não é o modelo avaliado, e isso deve ser dito na apresentação.

## 7. O que cada execução grava (`results/runs/<data>_<hash>_<nome>/`)

| Arquivo | Conteúdo |
|---|---|
| `config.yaml`, `meta.json` | config resolvida; split, hash, versões, limiar escolhido |
| `metrics.json`, `fold_scores.csv` | média/desvio e as notas dos 25 folds (com % de vetores de suporte) |
| `pipeline.joblib` | o modelo treinado no treino inteiro (o que o `final_test.py` avalia) |
| `svm_stats.json` | kernel, C, gamma efetivo, vetores de suporte, limiar |
| `importances_permutation.csv`, `importances.png` | queda do roc_auc ao embaralhar cada coluna original |
| `coeficientes_lineares.csv` | só no kernel linear e na logreg |

Outras saídas: `results/summary.csv`, `results/comparacoes/`, `results/figuras/` (mapas de
calor e fronteira 2D), `results/robustez/`, `results/final/` (métricas, matriz de confusão e curva ROC no teste).

## 8. Estrutura

```
SVM-Infarto/
├── data/raw/heart.csv            nunca alterar
├── data/split/70_30/, 80_20/     gerados uma vez pelo make_split
├── src/
│   ├── config.py      caminhos, colunas, SEED, CV 5x5, métricas
│   ├── data.py, make_split.py
│   ├── steps/         zeros, imputers, features, encoding, scalers, selection, models, wrappers
│   ├── registry.py    nome no YAML -> função
│   ├── builder.py     config -> Pipeline
│   ├── runner.py      validação cruzada + gravação
│   └── evaluate.py    métricas, matriz de confusão e curva ROC do teste
├── experiments/       um YAML por rodada
├── docs/anotacoes_artigo.md      decisões, achados e limitações
├── results/
├── run_experiments.py, summarize.py, compare.py
├── heatmap_c_gamma.py, fronteira_2d.py, robustez.py, final_test.py, split_check.py
└── CLAUDE.md
```
