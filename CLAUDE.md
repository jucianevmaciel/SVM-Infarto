# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projeto de mestrado (PPGCC - UFPI, Aprendizado de Máquina): SVM (scikit-learn) na base Heart
Failure Prediction (fedesoriano, Kaggle; alvo `HeartDisease`, doença cardíaca), com um sistema
declarativo de experimentos em rodadas. Segue a mesma arquitetura dos projetos `DecisionTree`,
`RandomForest` e `Glass-Identification-Neural-Network`, ao lado deste. A documentação de uso para
humanos está em `README.md`, em português; mantenha o código, os comentários e as mensagens em
português.

## Comandos

Todos rodam a partir da raiz, com o `.venv`:

```bash
pip install -r requirements.txt
python -m src.make_split                                             # data/raw/heart.csv -> data/split/70_30 e 80_20 (uma vez)
python correlacao.py [--split 70_30] [--limiar 0.5]                  # correlação entre as variáveis (antes do R0, só treino)
python run_experiments.py experiments/r0_baselines.yaml              # uma rodada
python run_experiments.py experiments/r3_kernel_hiperparametros.yaml --only R3_linear   # um experimento (ou um sweep inteiro)
python run_experiments.py experiments/r1_escalonamento.yaml --force  # repete, cria pasta nova
python summarize.py [--metric recall] [--split 70_30]                # regenera results/summary.csv
python compare.py <run_A> <run_B> [--metric roc_auc]                 # teste t corrigido (Nadeau-Bengio) fold a fold
python compare.py --rodada R1_ [--metric roc_auc|recall|brier] [--ref <pasta>] [--incluir <pasta> ...] [--salvar]
python heatmap_c_gamma.py --rodada R3_rbf [--incluir <pasta>] [--metric recall]
python robustez.py --run <pasta>                                     # R9 (só treino)
python fronteira_2d.py --run <pasta> [--C ...] [--gamma ...]         # figura didática (PCA 2D, só treino)
python final_test.py --run <pasta> [--baselines <pasta R0> ...]      # teste final, uma vez
python split_check.py --run70 <pasta> --run80 <pasta>                # R10
```

Não há testes, linter nem build. Para validar uma mudança, rode os experimentos afetados com
`--only` numa CÓPIA do projeto (não grave resultados de teste em `results/` do usuário). Para
validar um YAML sem rodar nada, use `load_experiments()` + `build_pipeline()` de
`run_experiments.py` num script temporário.

## Arquitetura

O fluxo é YAML → `resolve` → `build_pipeline` → `cross_validate` → pasta imutável em `results/runs/`.

- **Etapas fixas e ordenadas** (`src/registry.py:STEP_ORDER`): `zeros → missing_flags → imputer →
  features → drop_cols → encoding → scaler → selection`, seguidas do `model`. Cada etapa recebe uma opção pelo
  nome em `REGISTRY`; `none` pula a etapa (exceto `encoding`, obrigatória: só `onehot`).
  Depois, as opcionais `calibration` e `threshold` envolvem o pipeline inteiro, nessa ordem.
- **Colunas** (`src/config.py`): `NUM_COLS` (6), `CAT_COLS` (5), `ZERO_COLS` (Cholesterol,
  RestingBP: 0 = ausente). O `scaler` só escalona as contínuas (`steps/scalers.py:SCALE_COLS`);
  dummies, flags e FastingBS (0/1) ficam como estão. Os imputadores só atuam nas numéricas.
- **`split`** (chave de topo opcional na config, não é etapa): `70_30` (padrão, sai da config)
  ou `80_20`. Diz qual `data/split/<split>/train.csv` usar; vai para `meta.json`. O YAML lê
  `80_20` sem aspas como o número 8020: o `resolve` aceita as duas formas.
- **Model:** `{name: svm | logreg | gboost | dummy, **hiperparâmetros}` (sem `name`, vale `svm`).
  O SVC fica com `probability=False`: o `roc_auc` usa o `decision_function`; `brier` e
  `log_loss` são NaN até a calibração (R7). O `compare.py` pula execuções sem a métrica.
- **Normalização da config (`src/builder.py:resolve`):** baseline + override (o override
  **substitui a etapa inteira**, inclusive `model`); parâmetros do model iguais ao padrão da
  fábrica são removidos (mesmo modelo = mesmo hash); opcionais com `none` e `split: 70_30` saem.
- **Validações do builder:** `zeros: to_nan` exige imputer; `missing_flags` exige `zeros: to_nan`.
- **Método novo:** função-fábrica em `src/steps/<etapa>.py` + uma linha no `REGISTRY`.
  Transformadores que aprendem algo dos dados calculam no `fit` (anti-vazamento).
- **Não altere o comportamento de uma opção existente:** o hash não inclui o código. Crie uma
  opção nova.
- **Runner (`src/runner.py`):** hash = config + seed + CV + métricas + sha256 do `train.csv`;
  hash existente é pulado. Por fold: métricas, `n_sv`/`pct_sv` (vetores de suporte) e
  importância por permutação (`roc_auc`, colunas ORIGINAIS) → `fold_scores.csv`,
  `fold_importances.csv`. Modelo final no treino inteiro: `pipeline.joblib`, `svm_stats.json`
  (kernel, C, gamma, vetores de suporte, limiar), `coeficientes_lineares.csv` (só kernel linear
  / logreg), `importances_permutation.csv`, `importances.png`. Grava em `.tmp_*` e renomeia no
  fim; nunca apaga nem edita execuções existentes.
- **Métricas** (`src/config.py`): `SCORING` = accuracy, precision, recall, specificity, f1,
  balanced_accuracy, roc_auc, brier, log_loss. `PRIMARY_METRIC = roc_auc`,
  `SECONDARY_METRIC = recall` (coluna extra no `compare.py`). Mudar o `SCORING` muda o hash de
  todas as configs.
- **`builder.unwrap/model_of/svm_of`:** desembrulham calibração e limiar; `svm_of` devolve
  `None` nos baselines.

## Rodadas de experimentos

Um YAML por rodada em `experiments/`; o vencedor de uma rodada vira o baseline da seguinte.
Todas no split 70/30, CV 5x5 estratificada (`SEED=42`).

| Arquivo | Pergunta | Critério |
|---|---|---|
| `correlacao.py` (antes do R0) | correlação/associação entre as variáveis e com o alvo (Pearson/Spearman, V de Cramér, eta, informação mútua, VIF) | exploratória: não escolhe nada |
| `r0_baselines.yaml` | dummy, logreg, gboost | não há escolha |
| `r1_escalonamento.yaml` | `scaler` | roc_auc; none → standard → minmax → robust |
| `r2_dados.yaml` | `R2_ausentes` (imputer median/median_by_group/knn_scaled/iterative x flag; zeros crus), `R2_atalho` (`drop_cols`: sem o colesterol / só a flag / sem o FastingBS / sem os dois) e `R2_oldpeak` (`features`) | roc_auc; fechadas separadamente; regra de leitura do atalho no YAML |
| `r3_kernel_hiperparametros.yaml` | 4 kernels já ajustados: C x gamma (rbf), C (linear), grau x C (poly), C x gamma (sigmoid) | roc_auc; escolhe o kernel; linear → rbf → poly → sigmoid, menor C/gamma no empate |
| `r4_selecao.yaml` | `rfe_linear` / `kbest` | roc_auc; menos colunas no empate |
| `r5_recall.yaml` | `class_weight` e `threshold: tuned` (f1, f2, balanced_accuracy) | recall com especificidade mínima (registrar antes) |
| `r6_combinacao.yaml` | ablação + reajuste de C/gamma | roc_auc e recall contra a combinação |
| `r7_calibracao.yaml` | sigmoid x isotonic | brier |
| R8 (sem YAML) | importâncias e coeficientes: ST_Slope, ChestPainType, Oldpeak no topo? | teste de sanidade |
| R9 (`robustez.py`) | outras sementes da CV e subgrupos (Sex, colesterol ausente, idade) | — |
| teste final (`final_test.py`) | vencedor + baselines no teste 70/30 | — |
| `r10_split_80_20.yaml` | config final + baselines no 80/20 (`split_check.py`) | análise de sensibilidade |

- Linhas com `# ATUALIZAR: vencedor de Rx` são provisórias e devem ser trocadas pelo usuário
  antes de rodar. Não preencha vencedores por conta própria.
- O baseline de cada rodada é a config do vencedor anterior, então o runner o pula pelo hash:
  a referência no `compare.py --rodada` vem por `--incluir <vencedor anterior>` (e no
  `heatmap_c_gamma.py`, o ponto C=1/gamma=scale também).

## Restrições importantes

- **O conjunto de teste é isolado:** `load_test()` só é chamado pelo `final_test.py`. Não rode
  sem o usuário pedir. O 80/20 não escolhe nada (decisão do grupo: opção B).
- **Zeros impossíveis:** tratados DENTRO do pipeline (`zeros → missing_flags → imputer`), nunca
  na base inteira. Imputação por grupo só por ATRIBUTO (Sex, faixa de idade), nunca pelo alvo.
- **Colesterol ausente é um atalho em potencial:** Cholesterol = 0 tem 88% de doentes (contra
  48%), quase todos da base suíça. A flag pode ensinar o hospital de origem. Ver `R2_atalho`.
- **Oldpeak negativo:** é supradesnivelamento do ST (informação clínica). Não remover linhas.
- **Não regere `data/split/`** depois de rodar experimentos (muda o sha256 e quebra as
  comparações).
- **Paralelismo:** `parallel_config(backend="threading")`; o loky quebra no Python 3.14 com
  acento no caminho. O config do sklearn é por thread: use `sklearn.utils.parallel`.
