# Anotações para o artigo

Decisões, achados e limitações, registrados ao longo das rodadas. Critérios de leitura que
dependem de um valor escolhido pelo grupo (ex.: especificidade mínima do R5) são registrados
ANTES de rodar.

## Decisões de projeto (antes das rodadas)

- **Alvo:** `HeartDisease` (doença cardíaca), não infarto especificamente.
- **Splits:** 70/30 estratificado, seed 42, como nos trabalhos anteriores (DecisionTree e
  RandomForest). O 80/20 é usado só no R10, como análise de sensibilidade (opção B do grupo):
  não escolhe nada, e os dois resultados são reportados.
- **Validação:** CV estratificada 5 folds x 5 repetições (com ~640 pacientes no treino, um
  único split varia vários pontos só por acaso). Comparação pelo teste t corrigido de Nadeau e
  Bengio (2003); entre as configs sem diferença significativa, fica a mais simples.
- **Sem vazamento:** imputação, one-hot, escalonamento, seleção, limiar e calibração são etapas
  do pipeline, ajustadas só no treino de cada fold.
- **Métricas:** roc_auc (principal) e recall (segunda). Contexto médico: o falso negativo
  (doente classificado como saudável) é o erro mais caro. A matriz de confusão é reportada no
  teste final.
- **Zeros impossíveis:** Cholesterol = 0 e RestingBP = 0 são ausentes codificados como 0. São
  tratados como NaN e imputados (R2 compara as formas). Imputação por grupo apenas por atributo
  (Sex, faixa de idade), nunca pelo alvo.
- **Oldpeak negativo:** mantido. É supradesnivelamento do ST, sinal clínico real (inclusive de
  infarto). Remover as linhas mudaria a população e deixaria as comparações desiguais. O R2
  verifica alternativas nos mesmos pacientes.
- **Desbalanceamento:** pequeno (~55/45); não há rodada de reamostragem. O R5 trata o
  deslocamento da decisão a favor do recall (peso de classe e limiar).

- **Colesterol ausente não é aleatório (achado da análise exploratória):** Cholesterol = 0 tem
  88,4% de doentes (152 de 172), contra 47,7% entre os registrados (746). Os zeros vêm quase
  todos da base suíça, onde a maioria dos pacientes era doente (e 161 dos 172 são homens).
  Uma flag de ausência pode subir o roc_auc por um atalho: o modelo aprende o hospital de
  origem, não uma relação fisiológica, e esse ganho não se transferiria para outro hospital.
  O R2_atalho separa os dois efeitos.

## Critérios a registrar antes de rodar

- R2_atalho: (1) se `so_flag` empatar com `median + flag`, o valor do colesterol não ajuda e o
  ganho é da ausência (atalho): achado e limitação. (2) se `sem_colesterol` empatar com o
  vencedor do R2_ausentes, fica sem o colesterol (regra da mais simples). (3) se a flag vencer,
  conferir no R9 o desempenho no subgrupo "colesterol registrado".
- R2_atalho__sem_fastingbs (acrescentado depois da análise de correlação, antes de rodar o
  R2): comparar com o baseline (mesma config com o FastingBS). Se empatar, o FastingBS sai
  (regra da mais simples). Se piorar significativamente, fica, e a dúvida (informação clínica
  ou origem) vai para o R8 e as limitações.
- R2_atalho__sem_colesterol_fastingbs (acrescentado depois de rodar o R2 e antes de fechá-lo,
  porque sem_colesterol e sem_fastingbs empataram cada um sozinho, o segundo no limite,
  p = 0,085): comparar com R2_atalho__sem_colesterol. Se empatar, os dois saem (regra da mais
  simples). Se piorar significativamente, sai só o Cholesterol e o FastingBS fica.
- R5: especificidade mínima aceitável = **0,70** (decidido em 17/11, antes de rodar o R5).
  Critério: média da especificidade na CV 5x5 >= 0,70; entre as configs que cumprem, vence a
  de maior recall; no empate de recall (sem diferença significativa), a mais simples: sem peso
  nem limiar -> class_weight -> limiar ajustado. Ver justificativa na seção do R5.

## Rodadas

### Análise de correlação (antes do R0, só treino)

`python correlacao.py` (treino do 70/30, 642 pacientes; conferido no treino do 80/20, mesmo
retrato). Zeros impossíveis como ausentes e a flag `Cholesterol_missing`, só para a análise.
Medidas: V de Cramér (discreta x discreta), ponto-bisserial (contínua x alvo), eta (mista),
Pearson/Spearman (contínuas), informação mútua (IM, em nats). Saídas em
`results/correlacao/70_30/`.

**Cada variável contra o alvo (HeartDisease)**, ordenado pela força da associação (todos com
p < 0,05):

| Variável | Medida | Valor | IM |
|---|---|---|---|
| ST_Slope | V de Cramér | 0,63 | 0,213 |
| ChestPainType | V de Cramér | 0,53 | 0,154 |
| ExerciseAngina | V de Cramér | 0,49 | 0,132 |
| MaxHR | ponto-bisserial | -0,40 | 0,101 |
| Oldpeak | ponto-bisserial | 0,37 | 0,093 |
| Cholesterol_missing | V de Cramér | 0,31 | 0,057 |
| FastingBS | V de Cramér | 0,28 | 0,042 |
| Sex | V de Cramér | 0,27 | 0,038 |
| Age | ponto-bisserial | 0,24 | 0,027 |
| RestingECG | V de Cramér | 0,09 | 0,006 |
| RestingBP | ponto-bisserial | 0,08 | 0,051 |
| Cholesterol (registrados, n = 529) | ponto-bisserial | 0,08 | 0,042 |

- **Sinais fortes, coerentes com a cardiologia:** ST_Slope (Flat 82% e Down 84% de doentes,
  contra 19% no Up), ChestPainType (assintomático, ASY: 79% de doentes; ATA: 14%),
  ExerciseAngina (85% contra 35%), MaxHR (doentes atingem frequência máxima menor: 127 contra
  148 bpm) e Oldpeak (1,24 contra 0,45). É o "esperado" do teste de sanidade do R8.
- **Pouco associadas:** RestingECG, RestingBP e o VALOR do colesterol. Entre os registrados,
  doentes e saudáveis têm colesterol quase igual (média 250 contra 241 mg/dL, desvio ~60); por
  quartis, a % de doentes sobe só de 42% para 56%.
- **O atalho do colesterol aparece com força:** a AUSÊNCIA do exame (V = 0,31) diz mais que o
  valor (0,08): 89% de doentes entre os ausentes (113) contra 48% entre os registrados (529).
  Leitura: a flag funciona como indicador do hospital de origem (base suíça), não como
  fisiologia. Hipótese a testar no R2_atalho.
- **FastingBS também pode carregar a origem:** é a variável mais associada à flag do
  colesterol (V = 0,36); 43% dos pacientes com FastingBS = 1 não têm colesterol registrado,
  contra 10% dos com FastingBS = 0. Parte da associação de FastingBS com o alvo (81% contra 48%
  de doentes) pode vir do mesmo efeito de hospital. Registrar como limitação; conferir no R8 se
  FastingBS aparece mais importante do que a clínica justificaria.
- **Pouca redundância:** só um par passa de 0,5 (Oldpeak x ST_Slope, eta = 0,52, os dois
  medem o segmento ST). Os seguintes: ExerciseAngina x ST_Slope (0,48), ChestPainType x
  ExerciseAngina (0,44), Oldpeak x ExerciseAngina (0,40). Entre as contínuas, o maior é Age x
  MaxHR (Spearman -0,38). VIF das contínuas entre 1,02 e 1,29: sem multicolinearidade.
  Implicações: não há motivo para tirar colunas antes do R4; no R8, Oldpeak e ST_Slope podem
  dividir a importância entre si.
- **IM instável para as contínuas fracas:** a IM do colesterol varia entre os treinos (0,042
  no 70/30, 0,018 no 80/20), assim como a de RestingBP. O estimador por vizinhos tem viés e
  variância com ~600 pacientes; para as fracas, confiar mais no ponto-bisserial. O ranking das
  fortes é o mesmo nos dois treinos.

### R0 - Baselines

Execuções: `20261116_162243_9c3dbf_R0_dummy`, `20261116_162259_aba988_R0_logreg`,
`20261116_162319_f642d9_R0_gboost`. Pré-processamento: zeros -> NaN, mediana, one-hot,
padronização. CV 5x5 no treino 70/30 (média ± desvio entre os 25 folds).

| Modelo | roc_auc | recall | especificidade | acurácia | brier |
|---|---|---|---|---|---|
| dummy | 0,500 | 1,000 | 0,000 | 0,553 | 0,447 |
| logreg | 0,916 ± 0,028 | 0,853 ± 0,054 | 0,810 ± 0,048 | 0,834 ± 0,041 | 0,115 |
| gboost | 0,911 ± 0,026 | 0,873 ± 0,034 | 0,811 ± 0,044 | 0,845 ± 0,024 | 0,115 |

- O dummy reproduz o piso esperado (classifica todos como doentes; acurácia = prevalência de
  55%): confere o protocolo.
- A regressão logística já chega a roc_auc 0,916: régua alta para o SVM. Se o SVM ajustado
  apenas empatar com ela, a complexidade a mais não se justifica nesta base (discutir no artigo).

### R1 - Escalonamento

Execuções: baseline `20261116_162631_a42e0c_R1_baseline` (standard) e o sweep
`R1_escalonamento__scaler={none, minmax, robust}`. SVM com os padrões do scikit-learn (rbf,
C = 1, gamma = scale). Comparação: teste t corrigido (Nadeau-Bengio), referência = maior média
(robust).

| Scaler | roc_auc | recall | especificidade | % vetores de suporte | p (roc_auc) | p (recall) |
|---|---|---|---|---|---|---|
| none | 0,735 ± 0,036 | 0,766 ± 0,043 | 0,569 ± 0,072 | 79,1 | < 0,001 | < 0,001 |
| standard | 0,910 ± 0,026 | 0,895 ± 0,037 | 0,779 ± 0,049 | 46,6 | 0,73 | 0,63 |
| minmax | 0,908 ± 0,027 | 0,886 ± 0,042 | 0,767 ± 0,060 | 40,3 | 0,74 | 0,54 |
| robust | 0,910 ± 0,025 | 0,898 ± 0,034 | 0,774 ± 0,049 | 46,0 | referência | referência |

(% de vetores de suporte do modelo final, treinado no treino inteiro.)

- **Escalonar é indispensável:** sem escalonamento, o roc_auc cai 0,175 (t = -9,95) e 79% dos
  pacientes viram vetores de suporte. As colunas de números grandes (Cholesterol, MaxHR) dominam
  a distância do kernel RBF e o SVM quase não separa as classes. Confirma a hipótese do YAML.
- **standard, minmax e robust empatam** (roc_auc e recall, p > 0,5; diferenças muito menores
  que o desvio entre folds, ~0,026).
- **Decisão: `standard`** (a mais simples entre as empatadas: none -> standard -> minmax ->
  robust). É a referência do R2 (`--incluir 20261116_162631_a42e0c_R1_baseline`).
- O SVM com os padrões (0,910) fica pouco abaixo da logreg do R0 (0,916) e troca especificidade
  por recall (recall 0,895 contra 0,853; especificidade 0,779 contra 0,810). A conferir:
  `python compare.py 20261116_162631_a42e0c_R1_baseline 20261116_162259_aba988_R0_logreg`.

### R2 - Zeros e Oldpeak

Base: vencedor do R1 (`20261116_162631_a42e0c_R1_baseline`: zeros -> NaN, mediana, sem flag,
standard, SVM rbf padrão; roc_auc 0,910). Teste t corrigido; referência = maior média de cada
grupo. CSVs em `results/comparacoes/R2_*`.

**R2_ausentes (imputador x flag): tudo empata.** roc_auc entre 0,910 e 0,912 nas 10 variantes
(median, median_by_group por Sex e por faixa de idade, knn_scaled, iterative; com e sem flag),
todos com p > 0,35; recall entre 0,895 e 0,903, p > 0,29. A flag dá ~+0,002 de roc_auc, sem
significância. KNN e MICE não ganham da mediana, como previsto (o colesterol quase não se
correlaciona com os outros exames, ver a análise de correlação).
- **Decisão: mediana, sem flag** (a mais simples; é o próprio baseline).

**R2_zeros: zeros crus empatam, pelo motivo errado.** roc_auc 0,915 (crus) contra 0,910
(to_nan), p = 0,23. Com o 0 cru, o Cholesterol vira a mais importante das colunas fracas
(importância por permutação 0,0135, contra -0,002 no baseline): o 0 funciona como uma flag
escondida de "exame ausente", o mesmo atalho do hospital.
- **Decisão: manter to_nan.** Decisão metodológica, não de desempenho: deixar o 0 cru ensinaria
  ao modelo que "colesterol 0" indica doença.

**R2_oldpeak: tudo empata.** roc_auc 0,910 (none), 0,910 (oldpeak_abs, p = 0,79), 0,910
(st_elevation, p = 0,53); recall p > 0,46.
- **Decisão: none** (manter o Oldpeak com sinal). Confirma a hipótese do grupo.

**R2_atalho: o atalho do colesterol se confirmou.** roc_auc (recall):

| Execução | roc_auc | recall | Comparação registrada | p | Leitura |
|---|---|---|---|---|---|
| so_flag (sem o valor, com a flag) | 0,917 | 0,899 | x median + flag (0,912) | 0,069 | empata |
| sem_colesterol | 0,915 | 0,894 | x baseline (0,910) | 0,14 | empata |
| sem_fastingbs | 0,905 | 0,888 | x baseline (0,910) | 0,085 | empata (no limite) |
| sem_colesterol_fastingbs | 0,909 | 0,885 | x sem_colesterol (0,915) | 0,078 | empata (no limite) |

- **so_flag empata com median + flag:** o VALOR do colesterol não acrescenta nada além da
  AUSÊNCIA. As importâncias confirmam: no baseline, o valor tem importância -0,002 (inútil);
  com a flag, a coluna passa a pesar 0,007, só pela ausência. Achado e limitação: o ganho
  possível vem do hospital de origem (base suíça), não da fisiologia. O so_flag teve a maior
  média da rodada, mas não é candidato: adotá-lo seria adotar o atalho.
- **FastingBS também carrega a origem:** com a flag no modelo, a importância do FastingBS cai
  pela metade (0,0084 -> 0,0046), coerente com a associação dele com a ausência do colesterol
  (V = 0,36, análise de correlação).
- **sem_colesterol empata:** pela regra registrada, o Cholesterol sai.
- **FastingBS (caminho C):** sem_fastingbs empatou no limite (p = 0,085) e a remoção dos dois
  juntos não tinha sido testada. O experimento sem_colesterol_fastingbs foi acrescentado
  DEPOIS de rodar o R2 e ANTES de fechá-lo, com a regra registrada antes de rodar (comparar com
  sem_colesterol). Resultado: empata (p = 0,078; recall p = 0,21; especificidade p = 0,44) ->
  **os dois saem**. Sem os dois, o desempenho fica igual ao do baseline com todas as colunas
  (0,909 x 0,910), com duas colunas a menos e sem depender das variáveis ligadas à origem.
- Importâncias na configuração sem os dois: ST_Slope muito à frente (0,146), depois
  ChestPainType (0,028), Oldpeak (0,025), Sex (0,014) e MaxHR (0,012).
- **Ressalva (limitações):** é a segunda vez que tirar o FastingBS dá uma perda pequena
  (~0,005 de roc_auc) no limite da significância (p = 0,085 e 0,078). Leitura honesta: o
  FastingBS pode trazer um ganho pequeno que a CV com 642 pacientes não consegue confirmar; a
  regra da mais simples decide. Conferir de novo na ablação do R6 (devolver o FastingBS na
  configuração final).

**Configuração vencedora do R2 (baseline do R3):** zeros: to_nan (ainda trata o RestingBP = 0),
missing_flags: none, imputer: median, features: none, drop_cols: cholesterol_fastingbs,
scaler: standard. Execução: `20261116_221432_b974c7_R2_atalho__sem_colesterol_fastingbs`
(roc_auc 0,909 ± 0,025; recall 0,885).

### R3 - Kernel e hiperparâmetros

**Status: fechada** (com as grades estendidas nas pontas; decisão no fim da seção).

Base: vencedor do R2 (`20261116_221432_b974c7_R2_atalho__sem_colesterol_fastingbs`: to_nan,
mediana, sem flag, sem Cholesterol e FastingBS, standard; rbf C = 1, gamma = scale; roc_auc
0,909). 44 execuções novas (rbf 5 C x 5 gamma; linear 5 C; poly 3 graus x 3 C; sigmoid 3 C x 2
gamma), mais o baseline por `--incluir`. CSVs em `results/comparacoes/R3_*` e mapas de calor em
`results/figuras/R3_rbf_heatmap_*`.

**Melhores de cada kernel** (roc_auc; teste t corrigido contra a maior média):

| Config | roc_auc | recall | especificidade | % SV | p |
|---|---|---|---|---|---|
| poly grau 2, C = 0,1 | 0,915 ± 0,022 | 0,878 | 0,803 | 62 | referência |
| poly grau 3, C = 0,1 | 0,914 | 0,843 | 0,841 | 67 | 0,79 |
| poly grau 2, C = 1 | 0,914 | 0,872 | 0,808 | 42 | 0,80 |
| rbf C = 100, gamma = 0,01 | 0,911 | 0,868 | 0,796 | 38 | 0,53 |
| rbf C = 1, gamma = 0,1 | 0,910 | 0,882 | 0,785 | 45 | 0,41 |
| rbf C = 1, gamma = scale (baseline) | 0,909 | 0,885 | 0,778 | 47 | 0,37 |
| linear C = 0,01 | 0,903 | 0,878 | 0,774 | 59 | 0,052 |
| sigmoid C = 1, gamma = 0,01 | 0,903 | 0,878 | 0,774 | 59 | 0,051 |

(% SV: média dos 25 folds, do compare.py.)

Observações:
- **A fronteira é quase linear:** o melhor kernel ganha só ~0,012 de roc_auc sobre o linear, e
  o linear C = 0,01 empata no limite (p = 0,052). O sigmoid com gamma pequeno dá praticamente o
  mesmo resultado do linear (0,903; recall 0,878; especificidade 0,774), como esperado: com gamma
  pequeno, o kernel sigmoide se comporta como um linear. Também coerente com o R0: a regressão
  logística (linear) já chegava a 0,916 (com Cholesterol e FastingBS).
- **O SVM não supera os baselines do R0 até aqui:** o melhor (0,915) fica no nível da logreg
  (0,916) e do gboost (0,911). Ponto para a discussão: o ganho do SVM, se houver, não vem do
  kernel.
- **Mapa de calor do rbf: uma "crista" diagonal.** As melhores células ficam em (C = 1,
  gamma = 0,1) 0,910, (C = 10, gamma = 0,01) 0,907 e (C = 100, gamma = 0,01) 0,911: aumentar C
  e diminuir gamma juntos mantém o desempenho (margem mais estreita compensada por fronteira
  mais suave). Fora da crista:
  - gamma = 1 sobreajusta: 85-94% de vetores de suporte e roc_auc 0,855-0,886;
  - C grande com gamma grande (C = 100, gamma = 0,1 ou scale): roc_auc 0,854-0,861.
- **No poly, grau maior e C maior pioram:** grau 4 e C = 10 ficam entre 0,863 e 0,901. Graus
  mais altos trocam recall por especificidade (grau 4, C = 0,1: recall 0,819, especificidade
  0,857).
- **Configurações degeneradas (recall = 1 é falso):** com regularização forte, o SVM classifica
  todos (ou quase todos) como doentes. Especificidade 0 em todo o rbf com C = 0,01 e no rbf
  (C = 0,1, gamma = 0,001); entre 0,05 e 0,29 no rbf (C = 0,1, gamma = 1), no linear C = 0,001 e
  no sigmoid (C = 0,1, gamma = 0,01). O roc_auc dessas configs continua ~0,885 (a ORDEM dos
  pacientes é boa); o que quebra é o limiar de decisão (0 da decision_function). Por isso a
  tabela de recall do R3 (`R3_recall.csv`) não serve para escolher: a referência dela é uma
  config degenerada. Recall só deve ser lido junto com a especificidade.

**Pendência antes de fechar (regra registrada no YAML):** "se o melhor cair na PONTA de uma
grade, acrescentar um valor naquela direção". Aconteceu duas vezes:
- poly: o melhor (C = 0,1) está na ponta de baixo da grade de C -> acrescentar C = 0,01
  (graus 2, 3 e 4);
- rbf: o melhor (C = 100, gamma = 0,01) está na ponta de cima da grade de C -> seguir a crista
  com C = 1000 (gamma = 0,001 e 0,01).

**Leitura prevista (registrada ANTES das execuções extras):** refazer o compare.py com as
execuções extras. Pela regra da mais simples (linear -> rbf -> poly -> sigmoid; menor C; no
rbf, menor gamma), se o linear C = 0,01 continuar sem diferença significativa para a nova
referência, ele vence. Se deixar de empatar, vence o mais simples entre os empatados. Pelos
resultados atuais, o vencedor seria o linear C = 0,01, decidido no limite (p = 0,052).

**Execuções extras (R3_poly_ext e R3_rbf_ext):**

| Config | roc_auc | recall | especificidade | % SV | p contra o melhor |
|---|---|---|---|---|---|
| poly grau 2, C = 0,01 | 0,892 | 0,996 | 0,167 | 90 | < 0,001 |
| poly grau 3, C = 0,01 | 0,889 | 0,997 | 0,146 | 90 | < 0,001 |
| poly grau 4, C = 0,01 | 0,893 | 0,997 | 0,091 | 90 | 0,002 |
| rbf C = 1000, gamma = 0,001 | 0,906 | 0,857 | 0,776 | 38 | 0,068 |
| rbf C = 1000, gamma = 0,01 | 0,891 | 0,859 | 0,778 | 35 | 0,036 |

- poly com C = 0,01: as três configs são degeneradas (quase todos classificados como doentes) e
  significativamente piores. O melhor poly continua sendo C = 0,1: não há ganho além da ponta.
- rbf C = 1000: a crista não sobe além de (C = 100, gamma = 0,01). O máximo do rbf fica dentro
  da grade.
- A referência não mudou: poly grau 2, C = 0,1 (0,915).

**Decisão: SVM linear, C = 0,01** (`20261117_150444_6c0342_R3_linear__C=0.01`; roc_auc 0,903 ±
0,024; recall 0,878; especificidade 0,774; 59% de vetores de suporte). Continua sem diferença
significativa para a referência (p = 0,052) e é o mais simples pela ordem registrada (linear ->
rbf -> poly -> sigmoid; menor C).
- Ponto interior da grade do linear: C = 0,001 é degenerado (especificidade 0,29); de C = 0,1
  em diante, o roc_auc cai significativamente (0,898 a 0,901).
- **Ressalvas (discussão):** o empate é no limite (p = 0,052); o melhor kernel (poly grau 2)
  tem ~0,012 a mais de roc_auc e especificidade um pouco maior (0,803 contra 0,774), diferenças
  que a CV não confirma. Conclusão para o artigo: nesta base, com estas colunas, a fronteira é
  essencialmente linear; o kernel não traz ganho demonstrável.
- **Consequências:** baseline do R4 com `model: {name: svm, kernel: linear, C: 0.01}`; no R5, o
  R5_peso repete esse model + class_weight; no R6, o reajuste vira um sweep só de C em volta de
  0,01 (ex.: 0,003, 0,01, 0,03); no R8, o modelo linear grava `coeficientes_lineares.csv` (peso e
  sinal de cada coluna).

### R4 - Seleção de atributos

**Decisão de projeto (antes de rodar):** o RFE ranqueia as colunas com um `LinearSVC` próprio,
que por padrão usa C = 1. Como o vencedor do R3 é um SVM linear com C = 0,01, o RFE passou a
usar a MESMA regularização (`selection: {name: rfe_linear, C: 0.01}`): a ordem de eliminação
das colunas fica coerente com o modelo que será usado. (Com C = 1, regularização ~100x mais
fraca, a ordem poderia diferir da que o modelo final faria; a forma com C = 1 também seria
defensável, como no RFE clássico de Guyon et al., 2002.) O kbest não usa modelo (ANOVA F) e
fica como estava.

Base: vencedor do R3 (`20261117_150444_6c0342_R3_linear__C=0.01`), com 16 colunas depois do
one-hot (4 numéricas + 12 dummies). Grade: n_features / k = 6, 9, 12, 15 (16 = todas = o
baseline, `selection: none`, que entra por `--incluir`).

**Status: fechada** (com a grade estendida para 3 e 4 colunas; decisão com DESVIO do plano, registrado no fim da seção).

**Resultados** (roc_auc; teste t corrigido contra a maior média, rfe com 12 colunas):

| Config | Colunas | roc_auc | recall | especificidade | p (roc_auc) |
|---|---|---|---|---|---|
| todas (baseline, R3 linear C = 0,01) | 16 | 0,903 | 0,878 | 0,774 | 0,45 |
| rfe 15 / kbest 15 | 15 | 0,903 / 0,903 | 0,878 / 0,878 | 0,776 / 0,775 | 0,32 / 0,52 |
| rfe 12 / kbest 12 | 12 | 0,904 / 0,904 | 0,875 / 0,877 | 0,776 / 0,782 | referência / 0,75 |
| rfe 9 / kbest 9 | 9 | 0,903 / 0,902 | 0,872 / 0,873 | 0,778 / 0,778 | 0,69 / 0,39 |
| kbest 6 | 6 | 0,899 | 0,867 | 0,778 | 0,20 |
| rfe 6 | 6 | 0,894 | 0,859 | 0,771 | **0,025** |

(Recall e especificidade: nenhuma diferença significativa, p > 0,13 em todas.)

- **A seleção não melhora o modelo:** de 9 a 15 colunas, tudo fica em ~0,903, igual a usar as 16.
- **kbest com 6 colunas empata** (p = 0,20): pela regra registrada (no empate, a de menos
  colunas), é o vencedor provisório. As 6 colunas do modelo final, com o coeficiente do SVM
  linear: ST_Slope_Up (-0,73), ST_Slope_Flat (+0,44), ChestPainType_ASY (+0,42),
  ExerciseAngina_Y (+0,37), MaxHR (-0,21), ChestPainType_ATA (-0,17). Todos os sinais coerentes
  com a cardiologia (segmento ST ascendente, angina atípica e frequência máxima maior -> menos
  doença; ST plano, dor assintomática e angina no esforço -> mais doença).
- **rfe com 6 colunas é significativamente pior** (0,894; p = 0,025): escolhe quase as mesmas
  colunas, mas troca o MaxHR pelo Sex_M.
- **O Oldpeak sai com 6 colunas** (volta com 9). Ele era esperado entre os mais fortes (R8), mas
  é redundante com o ST_Slope (eta = 0,52 na análise de correlação; os dois medem o segmento ST).
  Ponto para o R8: a ausência do Oldpeak no modelo reduzido é redundância, não irrelevância.
- Coeficientes do modelo com as 16 colunas (R3 linear C = 0,01), para o R8: ST_Slope_Up -0,61,
  ST_Slope_Flat +0,50, ChestPainType_ASY +0,31, ExerciseAngina_Y +0,30, Sex_M +0,18, MaxHR -0,18,
  ChestPainType_ATA -0,18, Oldpeak +0,18, ChestPainType_NAP -0,11, ST_Slope_Down +0,11; as demais
  (Age, RestingBP, RestingECG, ChestPainType_TA) abaixo de 0,07 em módulo.

**Pendência antes de fechar:** o vencedor provisório (6 colunas) está na menor opção da grade.
Se 6 empatam com 16, menos colunas também podem empatar. Acrescentar R4_rfe_ext
(rfe_linear, C = 0,01, n_features = 3 e 4) e R4_kbest_ext (k = 3 e 4).

**Leitura prevista (registrada ANTES das execuções extras):** refazer o compare.py com as
extras; mesma regra da rodada: entre as configs sem diferença significativa para a referência
(roc_auc), vence a de menos colunas (no empate de número de colunas, kbest antes de rfe, por
ser o filtro mais simples, sem modelo). Se nenhuma das extras empatar, fica o kbest com 6.

**Execuções extras (R4_rfe_ext e R4_kbest_ext):**

| Config | Colunas do modelo final (coeficiente) | roc_auc | recall | especificidade | p (roc_auc) | p (recall) |
|---|---|---|---|---|---|---|
| rfe 3 | ST_Slope_Up (-0,97), ChestPainType_ASY (+0,61), ExerciseAngina_Y (+0,43) | 0,897 | 0,808 | 0,849 | 0,29 | < 0,001 |
| rfe 4 | as mesmas 3 + Sex_M (+0,01) | 0,892 | 0,829 | 0,801 | 0,031 | 0,016 |
| kbest 4 | ST_Slope_Up, ST_Slope_Flat, ChestPainType_ASY, ExerciseAngina_Y | 0,890 | 0,850 | 0,787 | 0,016 | 0,078 |
| kbest 3 | ST_Slope_Up, ST_Slope_Flat, ChestPainType_ASY | 0,867 | 0,842 | 0,787 | 0,004 | 0,051 |

(p de recall: contra a maior média de recall, kbest 15 = 0,878, igual ao baseline.)

Pela regra registrada (roc_auc; no empate, a de menos colunas), o vencedor seria o **rfe com 3
colunas**. Três problemas com ele:
1. **Empate frágil:** o rfe 4 é praticamente o mesmo modelo (as mesmas 3 colunas + Sex_M com
   peso 0,01), mas "perde" (p = 0,031) enquanto o rfe 3 "empata" (p = 0,29). Uma coluna de peso
   quase nulo não deveria mudar a conclusão: nessa faixa, a fronteira entre empatar e perder é
   ruído.
2. **Recall significativamente pior:** 0,808 contra 0,878 (p < 0,001), 7 pontos a menos na
   métrica secundária do protocolo, a do erro mais caro (doente classificado como saudável).
3. **Modelo grosso demais:** com 3 colunas 0/1 só existem 8 escores possíveis. O roc_auc é
   calculado sobre uma escada de 8 degraus (o que explica o deslocamento causado pelo peso de
   0,01), e o ajuste de limiar do R5 teria só 8 limiares possíveis.
(Ponto positivo para a discussão: o modelo de 3 colunas equivale a um escore clínico de três
perguntas sim/não, com roc_auc 0,897 e especificidade 0,849.)

**DESVIO DO PLANO (decidido depois de ver os resultados das extras):** acrescentado um critério
de desempate pela métrica secundária, já declarada no protocolo (recall): entre as configs sem
diferença significativa no roc_auc, são descartadas as significativamente piores em recall; entre
as restantes, vence a de menos colunas. Motivos: os três acima; sem esse critério, a escolha
dependeria de uma coluna com peso 0,01. Deve ser reportado no artigo como decisão posterior aos
dados.

**Decisão: kbest com 6 colunas** (`20261117_153215_ec5fa1_R4_kbest__k=6`; roc_auc 0,899 ± 0,025,
p = 0,20; recall 0,867, p = 0,13; especificidade 0,778). Colunas: ST_Slope_Up, ST_Slope_Flat,
ChestPainType_ASY, ExerciseAngina_Y, MaxHR, ChestPainType_ATA. O rfe 3 é excluído pelo recall; o
rfe 4, o kbest 4 e o kbest 3 já são piores no roc_auc.
- Nota: o kbest com 6 colunas é calculado DENTRO de cada fold (as colunas escolhidas podem variar
  entre folds); as 6 acima são as do modelo final, treinado no treino inteiro.
- **Consequências:** baseline do R5 com `selection: {name: kbest, k: 6}`. No R6, a ablação deve
  testar a volta de todas as colunas (`selection: none`) e, pela ressalva do R2, a do FastingBS.

### R5 - Recall

**Decisão de projeto (antes de rodar): especificidade mínima = 0,70, priorizando a redução dos
falsos negativos.**
- O falso negativo é o erro mais caro: um doente classificado como saudável deixa de ser
  investigado e tratado. O falso positivo tem custo menor: leva a mais exames (teste ergométrico,
  imagem de estresse), não a dano direto.
- O limite de 0,70 evita o extremo oposto: especificidade baixa demais tornaria a triagem inútil
  (todos encaminhados para investigação).
- Referência de comparação (não de limite): o teste ergométrico (ECG de esforço), exame não
  invasivo padrão, tem sensibilidade média de 68% e especificidade média de 77% contra a
  angiografia (meta-análise de 147 estudos: Gianrossi, Detrano, Froelicher et al., Circulation,
  1989, 80(1):87-98; Detrano é um dos responsáveis pelas bases que compõem esta). Reportar no
  artigo o recall e a especificidade do modelo ao lado desses valores, para a troca ficar
  explícita. Um limite de 0,75 (ancorado nos 77%) foi considerado e descartado: deixaria só ~3
  pontos de folga (o modelo parte de 0,778) e contradiria a prioridade dada ao falso negativo.
- Ponto de partida (vencedor do R4, kbest 6, SVM linear C = 0,01): recall 0,867, especificidade
  0,778.

**Atenção ao class_weight: balanced (verificado antes de rodar):** nesta base a classe
MINORITÁRIA é a dos saudáveis (45%). O "balanced" dá peso 1,12 à classe 0 e 0,90 à classe 1:
desloca a decisão a favor da ESPECIFICIDADE, o contrário do objetivo do R5. Para favorecer o
recall, o peso maior precisa ir para a classe 1 (ex.: class_weight: {0: 1, 1: 2}).

**Execuções:** 3 pesos (balanced, {0: 1, 1: 1,5}, {0: 1, 1: 2}) e 3 critérios de limiar
(TunedThresholdClassifierCV com f1, f2, balanced_accuracy), contra o baseline (vencedor do R4).
Experimentos de peso escritos um a um (o nome da pasta com o dicionário teria ":", inválido no
Windows).

| Config | recall | especificidade | roc_auc | limiar | Cumpre >= 0,70? |
|---|---|---|---|---|---|
| limiar f2 | 0,965 | **0,530** | 0,899 | -0,70 | não |
| **peso {0: 1, 1: 2}** | **0,912** | 0,721 | 0,899 | - | sim |
| peso {0: 1, 1: 1,5} | 0,892 | 0,753 | 0,899 | - | sim |
| limiar f1 | 0,891 | 0,736 | 0,899 | -0,18 | sim |
| baseline (R4, kbest 6) | 0,867 | 0,778 | 0,899 | 0 | sim |
| limiar balanced_accuracy | 0,847 | 0,800 | 0,899 | +0,14 | sim |
| peso balanced | 0,844 | 0,800 | 0,898 | - | sim |

(limiar: na escala da decision_function do SVM; o padrão é 0.)

- **roc_auc não muda** (0,898-0,899, p > 0,23): peso e limiar só deslocam o ponto de operação, a
  ordem dos pacientes é a mesma (no limiar, idêntica por construção).
- **O aviso sobre o balanced se confirmou:** recall cai para 0,844 e a especificidade sobe para
  0,800, o contrário do objetivo (peso maior para os saudáveis, a classe minoritária).
- **f2 passa do ponto:** recall 0,965, mas especificidade 0,530 (abaixo do mínimo). O limiar
  escolhido (-0,70) classifica como doente quase todo paciente com escore levemente negativo.
- **Comparação pelo critério** (teste t corrigido, referência = maior recall entre os que
  cumprem o mínimo, peso {0: 1, 1: 2}): limiar f1 empata em recall (0,891, p = 0,41); peso
  {0: 1, 1: 1,5} é significativamente menor (0,892, p = 0,021); o baseline também (0,867,
  p = 0,006).

**Decisão: class_weight {0: 1, 1: 2}** (`20261117_161029_156add_R5_peso__1x2`): maior recall
entre os que cumprem a especificidade mínima; empata com o limiar f1, e pela ordem registrada
(sem peso nem limiar -> class_weight -> limiar ajustado) o peso é o mais simples.
- Ganho em relação ao R4: recall 0,867 -> 0,912 (+4,5 pontos, p = 0,006); custo: especificidade
  0,778 -> 0,721 (-5,7 pontos, p = 0,002). roc_auc igual.
- Comparado ao teste ergométrico (referência): recall 0,912 contra 68%; especificidade 0,721
  contra 77%.
- **Ressalva:** a especificidade média (0,721) fica pouco acima do mínimo, e 8 dos 25 folds
  ficam abaixo de 0,70 (mínimo 0,638). O critério registrado é sobre a média e foi cumprido, mas
  a folga é pequena: no teste final, a especificidade pode cair abaixo de 0,70 por variação
  amostral. Reportar com o intervalo dos folds.
- **Consequência:** baseline do R6 com `model: {name: svm, kernel: linear, C: 0.01,
  class_weight: {0: 1, 1: 2}}`, sem threshold.

### R6 - Combinação e ablação

**Referência:** a combinação dos vencedores R1-R5 (= vencedor do R5,
`20261117_161029_156add_R5_peso__1x2`): to_nan, mediana, sem flag, sem Cholesterol e FastingBS,
standard, kbest 6, SVM linear C = 0,01, class_weight {0: 1, 1: 2}. Os exemplos originais do YAML
(rbf, limiar, tratamento dos zeros) foram trocados por ablações adequadas aos vencedores reais:
sem limiar não se aplica (o R5 escolheu o peso); sem tratamento dos zeros não mede nada (com o
Cholesterol fora, o to_nan só afeta 1 paciente com RestingBP = 0); o reajuste virou só de C
(kernel linear). A regra de leitura abaixo foi definida antes de rodar, na conversa de
planejamento, e registrada aqui junto com os resultados. Em toda leitura vale a restrição do R5
(especificidade média >= 0,70).

| Experimento | Muda | roc_auc (p) | recall (p) | especificidade (p) | Regra | Leitura |
|---|---|---|---|---|---|---|
| referência | - | 0,899 | 0,912 | 0,721 | - | - |
| sem peso (= vencedor do R4) | R5 | 0,899 (0,96) | 0,867 (**0,006**) | 0,778 (0,002) | recall; se empatar, o peso sai | perde recall: **peso fica** |
| sem_selecao (16 colunas) | R4 | 0,903 (0,48) | 0,921 (0,30) | 0,716 (0,78) | roc_auc e recall; se empatar, fica a seleção | empata: **seleção fica** |
| com_fastingbs | R2 | 0,899 (1,0) | 0,912 (1,0) | 0,721 (1,0) | se empatar, fica fora | idêntico: **fica fora** |
| kernel_poly2 (C = 0,1) | R3 | 0,897 (0,57) | 0,948 (0,007) | **0,607** | roc_auc; se empatar, linear | viola o mínimo e empata: **linear fica** |
| reajuste C = 0,003 | R3 | 0,899 (0,94) | 0,995 | **0,128** | - | degenerado: excluído |
| reajuste C = 0,03 | R3 | 0,897 (0,38) | 0,892 (0,11) | 0,750 (0,025) | roc_auc, depois recall; se empatar, C = 0,01 | empata: **C = 0,01 fica** |

- **Todas as ablações confirmam a combinação:** o único vencedor cuja retirada piora o critério
  é o peso do R5 (recall -4,5 pontos, p = 0,006). Seleção, kernel e C empatam e ficam pela regra
  da mais simples; o FastingBS não muda nada.
- **FastingBS (ressalva do R2):** a execução com o FastingBS de volta tem fold_scores IDÊNTICOS
  aos da referência: o kbest com k = 6 nunca escolhe o FastingBS, em nenhum dos 25 folds. Com a
  seleção no pipeline, a questão do FastingBS não afeta o modelo final. A ressalva do R2
  continua valendo só para modelos sem seleção (não testado aqui).
- **poly2 com peso:** o peso dobrado leva o poly a um recall de 0,948, mas a especificidade cai
  para 0,607 (abaixo do mínimo). Mesmo sem a restrição, o roc_auc empata com o linear.
- **C = 0,03:** recall um pouco menor (0,892, sem diferença significativa) e especificidade
  maior (0,750, p = 0,025). Pela regra registrada, fica C = 0,01 (empate no critério). Ponto para
  a discussão: C = 0,03 seria uma alternativa com mais folga de especificidade.
- **C = 0,003:** com regularização mais forte e o peso dobrado, o modelo degenera (classifica
  quase todos como doentes; especificidade 0,128).

**Decisão: a combinação fica como está.** Configuração final (para R7 em diante):
`20261117_161029_156add_R5_peso__1x2` (roc_auc 0,899 ± 0,025; recall 0,912 ± 0,036;
especificidade 0,721 ± 0,049; acurácia 0,826).
- Para a discussão: o roc_auc final (0,899) fica abaixo da regressão logística do R0 (0,916),
  que usava todas as colunas, inclusive Cholesterol e FastingBS. Parte da diferença vem das
  decisões do R2 (tirar as variáveis ligadas ao hospital de origem) e do R4 (6 colunas); o modelo
  final troca ~0,017 de roc_auc por simplicidade (6 colunas) e independência do atalho.

### R7 - Calibração

**Decisão de projeto (antes de rodar): a calibração serve só para as PROBABILIDADES; a decisão
(classe) continua sendo a do modelo final do R6, sem calibração.**
- Motivo, verificado numa cópia com a mesma CV 5x5 (não gravado em results/): a calibração desfaz
  o efeito do peso do R5. O class_weight {0: 1, 1: 2} desloca a fronteira de propósito; a
  calibração aprende a converter o escore em probabilidade real, e o modelo calibrado classifica
  como doente quem tem probabilidade > 0,5, o que devolve o ponto de decisão ao lugar "neutro":

| Config | roc_auc | recall | especificidade | brier |
|---|---|---|---|---|
| final do R6 (sem calibração) | 0,899 | 0,912 | 0,721 | - |
| + sigmoid | 0,899 | 0,857 | 0,791 | 0,125 |
| + isotonic | 0,897 | 0,850 | 0,785 | 0,126 |
| + sigmoid + limiar tuned (f2) | 0,899 | 0,976 | 0,532 | 0,125 |

- O recall do modelo calibrado volta ao nível do R4 sem peso (0,867): usá-lo como classificador
  anularia a decisão do R5.
- Alternativa considerada e descartada: calibrar e reajustar o limiar sobre a probabilidade
  calibrada. Os critérios de limiar disponíveis não respeitam a especificidade mínima (f2 vai a
  0,532), e um critério novo ("maior recall com especificidade >= 0,70") exigiria código e
  validação adicionais.
- **Como fica:** o R7 escolhe o método (sigmoid x isotonic) pelo brier (menor é melhor;
  roc_auc deve ficar praticamente igual). O teste final avalia o classificador do R6
  (`20261117_161029_156add_R5_peso__1x2`). A versão calibrada é reportada como complemento: o
  risco em % para cada paciente, com o brier e a confiabilidade das probabilidades. No artigo: "o
  modelo decide com o ponto de operação do R5; a versão calibrada fornece o risco estimado".
- Para o teste final: se quiser as métricas da versão calibrada no teste, passá-la em
  `--baselines` (ela será avaliada no limiar de 0,5, com o recall menor esperado).

**Resultados** (teste t corrigido; brier: menor é melhor):

| Config | brier | roc_auc | recall | p (brier) | p (roc_auc, contra o R6) |
|---|---|---|---|---|---|
| sigmoid (`20261117_163434_74c6bd_R7_baseline`) | 0,1249 ± 0,018 | 0,899 | 0,857 | referência | 1,0 (idêntico) |
| isotonic | 0,1264 ± 0,018 | 0,897 | 0,850 | 0,43 | 0,40 |

- Os dois empatam no brier (p = 0,43). O sigmoid deixa o roc_auc idêntico ao do modelo sem
  calibração (é monotônico: não muda a ordem dos pacientes); o isotonic muda um pouco (degraus
  com empates), sem significância.
- O recall do modelo calibrado no limiar de 0,5 (0,857) confirma a verificação feita antes de
  rodar: a calibração desfaz o peso do R5.
- Referência: logreg e gboost do R0 tinham brier 0,115 (com todas as colunas, inclusive as
  ligadas ao hospital de origem).

**Decisão: sigmoid (método de Platt)** para as probabilidades: empata com o isotonic e é o mais
simples (2 parâmetros, contra uma função em degraus que pode sobreajustar com ~640 pacientes).
Pela decisão de projeto acima, o classificador do teste final continua sendo o do R6
(`20261117_161029_156add_R5_peso__1x2`); o calibrado (`20261117_163434_74c6bd_R7_baseline`)
fornece o risco em %.

### R8 - Interpretação (teste de sanidade)

Modelo final do R6 (`20261117_161029_156add_R5_peso__1x2`): SVM linear, C = 0,01, peso
{0: 1, 1: 2}, kbest 6. Esperado (literatura e cardiologia): ST_Slope, ChestPainType e Oldpeak
no topo.

**Importância por permutação** (queda do roc_auc ao embaralhar a coluna ORIGINAL, nos 25 folds
de validação):

| Coluna | Queda média do roc_auc | Folds com queda > 0 | Faixa |
|---|---|---|---|
| ST_Slope | 0,175 ± 0,020 | 25/25 | 0,144 a 0,212 |
| ChestPainType | 0,052 ± 0,012 | 25/25 | 0,029 a 0,073 |
| ExerciseAngina | 0,013 ± 0,008 | 23/25 | -0,004 a 0,032 |
| MaxHR | 0,011 ± 0,006 | 24/25 | -0,000 a 0,024 |
| Oldpeak | -0,0002 | 0/25 | escolhido pelo kbest em só 1 fold |
| Age, Sex, RestingBP, RestingECG | 0 | - | não escolhidas pelo kbest |
| Cholesterol, FastingBS | 0 | - | retiradas no R2 |

**Coeficientes do SVM linear** (colunas padronizadas/dummies, modelo treinado no treino inteiro):
ST_Slope_Up -0,76; ChestPainType_ASY +0,51; ST_Slope_Flat +0,50; ExerciseAngina_Y +0,36;
ChestPainType_ATA -0,23; MaxHR -0,16. Vetores de suporte: 366 (57%), 243 saudáveis e 123
doentes (o peso 2 na classe 1 faz menos doentes ficarem dentro da margem).

**Leitura: o teste de sanidade passa.**
- ST_Slope e ChestPainType no topo, com folga e em todos os folds, como esperado. Todos os
  sinais coerentes com a clínica: segmento ST ascendente, angina atípica (ATA) e frequência
  máxima maior -> menos doença; ST plano, dor assintomática (ASY) e angina no esforço -> mais
  doença.
- O Oldpeak não aparece por REDUNDÂNCIA com o ST_Slope (eta = 0,52 na análise de correlação; os
  dois medem o segmento ST), não por irrelevância: no modelo com as 16 colunas (R3), ele tinha
  coeficiente +0,18. O kbest o escolheu em 1 dos 25 folds.
- Nenhum sinal de vazamento ou de erro de codificação: nenhuma coluna inesperada no topo, e as
  variáveis ligadas ao hospital de origem (Cholesterol, FastingBS) não estão no modelo.
- Cuidado na leitura: importância 0 nas colunas não escolhidas é consequência da seleção (o
  modelo não as usa), não uma medida de irrelevância clínica.

**Achado importante para a discussão: o modelo depende do teste ergométrico.** Das 4 variáveis
usadas, 3 vêm do próprio teste de esforço (ST_Slope, ExerciseAngina e MaxHR); a quarta é o tipo
de dor no peito (anamnese). Consequências:
- O modelo não substitui o teste ergométrico: ele INTERPRETA o resultado do teste junto com o
  tipo de dor. Só pode ser aplicado a pacientes que fizeram o teste de esforço.
- A comparação com a sensibilidade e especificidade do teste ergométrico (68% / 77%,
  Gianrossi et al., 1989) deve ser lida assim: o modelo combina o resultado do teste com a
  anamnese e chega a recall 0,912 / especificidade 0,721 na CV. Não é "um exame contra outro".
- A dependência forte de uma única variável (sem o ST_Slope, o roc_auc cai ~0,17) é uma
  fragilidade: um erro ou falta nessa medida compromete a previsão.

### R9 - Robustez

Modelo avaliado: o final do R6 (`20261117_161029_156add_R5_peso__1x2`). Para comparação, a
regressão logística do R0 (`20261116_162259_aba988_R0_logreg`: as 11 colunas, inclusive
Cholesterol e FastingBS; zeros -> NaN, mediana, SEM flag, sem peso). Só treino (642 pacientes);
o teste não foi tocado. Saídas em `results/robustez/<run>/` (`sementes.csv`, `subgrupos.csv`).

**1. Outras sementes da CV** (5x5 refeita com as sementes 1, 7, 13, 99 e 2024, além da 42):

| Modelo | roc_auc | recall | especificidade | amplitude roc_auc / recall |
|---|---|---|---|---|
| SVM final | 0,897 a 0,900 | 0,909 a 0,912 | 0,721 a 0,725 | 0,003 / 0,003 |
| logreg R0 | 0,916 a 0,918 | 0,853 a 0,862 | 0,810 a 0,815 | 0,003 / 0,009 |

- **Estável:** a divisão dos folds quase não muda a nota. A amplitude entre sementes (0,003) é
  menor que as diferenças que decidiram as rodadas (ex.: ~0,005 da saída do FastingBS no R2),
  então as escolhas não foram efeito do sorteio dos folds.
- Ressalva: isso mede só a variação devida à partição dos folds. A incerteza por ter só 642
  pacientes aparece no desvio entre folds (~0,025 no roc_auc) e será medida no teste final.

**2. Subgrupos** (previsões fora da amostra da CV principal: cada paciente é previsto pelos
modelos que não o viram; média das 5 repetições). Formato: roc_auc / recall / especificidade.

| Subgrupo | Pacientes (doentes / saudáveis) | SVM final | logreg R0 |
|---|---|---|---|
| Sex F | 137 (40 / 97) | 0,860 / 0,870 / 0,695 | 0,876 / **0,685** / 0,876 |
| Sex M | 505 (315 / 190) | 0,907 / 0,917 / 0,734 | 0,909 / 0,874 / 0,777 |
| Colesterol ausente | 113 (101 / 12) | 0,859 / 0,885 / 0,417 | 0,882 / 0,885 / 0,467 |
| Colesterol registrado | 529 (254 / 275) | 0,902 / 0,923 / 0,734 | 0,911 / 0,840 / 0,825 |
| Idade < 55 | 315 (133 / 182) | 0,931 / 0,887 / 0,813 | 0,945 / 0,815 / 0,898 |
| Idade >= 55 | 327 (222 / 105) | 0,842 / 0,927 / **0,560** | 0,863 / 0,876 / 0,659 |

Cuidado na leitura: o "desvio" gravado em `subgrupos.csv` (<= 0,016) é entre as repetições, que
usam os MESMOS pacientes; não mede a incerteza do subgrupo. Para isso vale o tamanho do grupo
(erro padrão aproximado de uma proporção: ~0,05 com ~100 pacientes, ~0,07 com 40, ~0,14 com 12).
Também é esperado que o roc_auc dentro de cada subgrupo fique abaixo do global: parte do acerto
global vem de ordenar bem pacientes de grupos diferentes.

- **Idade >= 55 (achado principal):** especificidade 0,560 contra 0,813 abaixo de 55 anos
  (~105 saudáveis; diferença bem maior que o erro esperado), e roc_auc 0,842 contra 0,931. O
  modelo classifica como doentes muitos idosos saudáveis. A logreg tem a MESMA diferença entre
  as faixas (0,659 contra 0,898; 0,24 contra 0,25 no SVM): a dificuldade vem da base/problema,
  não das escolhas das rodadas. A especificidade do SVM é menor em todos os grupos por causa do
  peso do R5 (troca consciente de especificidade por recall). Hipótese (não testada): 3 das 4
  variáveis usadas vêm do teste ergométrico (ST_Slope, ExerciseAngina, MaxHR), e alterações no
  teste e a queda da frequência máxima são mais comuns com a idade, mesmo sem doença.
- **Mulheres: o SVM final é mais equilibrado que a logreg.** A logreg perde muito mais doentes
  entre as mulheres (recall 0,685, contra 0,874 nos homens); o SVM fica em 0,870 contra 0,917.
  Diferença entre os modelos: ~7 das 40 mulheres doentes. Explicação confirmada pelos
  coeficientes: a logreg usa o Sex (`Sex_M` = +1,19, o 4º maior coeficiente) e, como a
  prevalência é menor nas mulheres (29% contra 62%), baixa o risco de toda mulher. O SVM final
  não usa o Sex (o kbest não o escolhe). Com 40 doentes, é evidência, não prova. Dentro do SVM,
  a diferença entre os sexos (recall 0,870 x 0,917; especificidade 0,695 x 0,734) está dentro do
  erro esperado.
- **Colesterol ausente: inconclusivo quanto ao atalho.** Só 12 saudáveis nesse grupo (cada um
  vale ~0,08 de especificidade). A logreg do R0 não tem a flag (o ausente recebe a mediana), então
  também não "vê" a ausência e não serve para medir o atalho: o recall sai igual (0,885 nos dois).
  Sinal favorável: no SVM final, o recall no grupo ausente (0,885) NÃO é maior que no registrado
  (0,923), ou seja, o modelo não trata a ausência como sinal de doença (coerente com o R2). O
  roc_auc menor no grupo ausente (0,859 contra 0,902), que vem quase todo da base suíça, é um
  indício de desempenho pior num hospital diferente (ver Limitações).
- Pendência opcional: rodar o `robustez.py` num modelo COM a flag
  (`20261116_163728_4252a8_R2_ausentes__imputer=median_missing_flags=flags`) para ver a assinatura
  do atalho no grupo ausente (recall perto de 1, especificidade baixa).

**Conclusão do R9:** o resultado não depende da semente da CV. O modelo é pior em pacientes com
55 anos ou mais (especificidade) e possivelmente na base suíça (grupo sem colesterol); é mais
equilibrado entre os sexos que a regressão logística com todas as colunas.

### Teste final (70/30)

Avaliado UMA vez (`python final_test.py`, 17/11/2026), no conjunto de teste do 70/30: 276
pacientes (153 doentes, 123 saudáveis). Classificador: o final do R6
(`20261117_161029_156add_R5_peso__1x2`). Comparação: os baselines do R0 (dummy, logreg, gboost,
com todas as colunas) e a versão calibrada do R7 (`20261117_163434_74c6bd_R7_baseline`, limiar
0,5). Saídas em `results/final/20261117_161029_156add_R5_peso__1x2/` (métricas, matriz de
confusão, curva ROC e `baselines.csv`).

**SVM final: CV x teste**

| Métrica | CV 70/30 (média ± desvio) | Teste |
|---|---|---|
| roc_auc | 0,899 ± 0,025 | **0,910** |
| recall | 0,912 ± 0,036 | **0,908** (139 de 153; 14 perdidos) |
| especificidade | 0,721 ± 0,049 | **0,756** (93 de 123; 30 falsos positivos) |
| acurácia | 0,826 | 0,841 |
| precisão | - | 0,822 |
| f1 | - | 0,863 |

Matriz de confusão (linhas = real, colunas = previsto):

| | sem doença | com doença |
|---|---|---|
| sem doença | 93 | 30 |
| com doença | 14 | 139 |

- **O teste confirma a CV:** todas as métricas dentro do desvio da CV, e roc_auc e
  especificidade um pouco acima. O otimismo da seleção (Limitações) não se manifestou.
- **O ponto de operação do R5 se manteve fora da amostra:** recall 0,908 com especificidade
  0,756 (acima do mínimo de 0,70 definido no R5).

**Comparação com os baselines no teste**

| Modelo | roc_auc | recall | especificidade | acurácia | brier | FN / FP |
|---|---|---|---|---|---|---|
| SVM final (R6) | 0,910 | **0,908** | 0,756 | 0,841 | - | **14** / 30 |
| SVM calibrado (R7, limiar 0,5) | 0,910 | 0,830 | 0,829 | 0,830 | 0,117 | 26 / 21 |
| logreg R0 | 0,930 | 0,889 | 0,862 | 0,877 | 0,097 | 17 / 17 |
| gboost R0 | **0,934** | 0,882 | **0,894** | **0,888** | **0,093** | 18 / 13 |
| dummy R0 | 0,500 | 1,000 | 0,000 | 0,554 | 0,446 | 0 / 123 |

- **O SVM final NÃO supera os baselines no roc_auc:** logreg e gboost ficam ~0,02 acima (0,930 e
  0,934 contra 0,910), como já acontecia na CV (logreg 0,916 contra 0,899). Com 276 pacientes, a
  incerteza do roc_auc no teste é de ~±0,02: a diferença está no limite. Não foi feito teste
  pareado (ex.: DeLong) no teste; sem ele, não dá para afirmar nem que o SVM é pior nem
  equivalente.
- **Troca do SVM:** identifica 3 a 4 doentes a mais (14 perdidos, contra 17 da logreg e 18 do
  gboost), ao custo de 13 a 17 falsos positivos a mais. É a consequência do peso {0: 1, 1: 2}
  do R5, escolhido de propósito (o falso negativo é o erro mais caro).
- **Parte da vantagem dos baselines pode vir do atalho:** eles usam Cholesterol e FastingBS,
  ligados ao hospital de origem (R2). O teste vem dos MESMOS hospitais, então não penaliza o
  atalho; num hospital novo a diferença pode ser menor (não medido).
- **O que justifica o SVM final não é a nota, é o perfil:** recall mais alto, só 4 variáveis
  efetivamente usadas (ST_Slope, ChestPainType, ExerciseAngina, MaxHR; R8), independência das
  variáveis ligadas ao hospital de origem (R2) e mais equilíbrio entre os sexos que a logreg
  (R9). Confirma a troca prevista no R6 (~0,02 de roc_auc por simplicidade e independência do
  atalho).
- **Calibrado (R7):** roc_auc idêntico ao do SVM (0,910; o método de Platt não muda a ordem dos
  pacientes). No limiar 0,5, o recall cai para 0,830 (26 doentes perdidos), como previsto no R7:
  a calibração desfaz o peso do R5. Serve para o risco em %, não para a decisão. O brier (0,117)
  fica acima do da logreg (0,097) e do gboost (0,093), que usam todas as colunas.

### R10 - Split 80/20

Análise de sensibilidade (opção B do grupo: o 80/20 não escolhe nada). A configuração final do
70/30 foi refeita no 80/20 (`20261117_235218_1776a5_R10_svm`; 734 pacientes no treino, 184 no
teste: 102 doentes, 82 saudáveis), junto com os baselines espelhando o R0 (todas as colunas, sem
seleção): `20261117_235307_794042_R10_dummy`, `20261117_235335_b36241_R10_logreg`,
`20261117_235408_bdd18f_R10_gboost`. Configs conferidas: idênticas às do 70/30, só muda o split.
Saídas: `results/comparacoes/R10_splits.csv` e `results/final/20261117_235218_1776a5_R10_svm/`.

**SVM final: 70/30 x 80/20** (`split_check.py`; sem teste t, porque treinos e folds são
diferentes: a leitura é se a diferença cabe no desvio da CV)

| Métrica | CV 70/30 | CV 80/20 | dif CV (80-70) | dentro do desvio? | teste 70/30 | teste 80/20 |
|---|---|---|---|---|---|---|
| roc_auc | 0,899 ± 0,025 | 0,904 ± 0,032 | +0,006 | sim | 0,910 | 0,885 |
| recall | 0,912 ± 0,036 | 0,910 ± 0,032 | -0,002 | sim | 0,908 | 0,892 |
| especificidade | 0,721 ± 0,049 | 0,746 ± 0,063 | +0,025 | sim | 0,756 | 0,732 |
| acurácia | 0,826 ± 0,028 | 0,837 ± 0,032 | +0,010 | sim | 0,841 | 0,821 |
| f1 | 0,853 ± 0,024 | 0,860 ± 0,026 | +0,007 | sim | 0,863 | 0,847 |

- **O SVM não depende do split:** todas as diferenças da CV ficam dentro do desvio. No teste, o
  roc_auc do 80/20 (0,885) fica 0,025 abaixo do 70/30, também dentro do desvio da CV e da
  incerteza de um teste com 184 pacientes (~±0,025). Matriz de confusão do 80/20: 60 VN, 22 FP,
  11 FN, 91 VP.

**Baselines no 80/20**

| Modelo | CV roc_auc | CV recall | teste roc_auc | teste recall | teste especificidade | FN / FP |
|---|---|---|---|---|---|---|
| SVM final | 0,904 | 0,910 | 0,885 | 0,892 | 0,732 | 11 / 22 |
| logreg | 0,919 | 0,870 | **0,932** | **0,912** | 0,854 | **9** / 12 |
| gboost | **0,920** | 0,876 | 0,915 | 0,873 | **0,866** | 13 / 11 |
| dummy | 0,500 | 1,000 | 0,500 | 1,000 | 0,000 | 0 / 82 |

- **A vantagem dos baselines no roc_auc se repete nos dois splits:** 0,015 a 0,02 na CV e no
  teste do 70/30; 0,03 a 0,047 no teste do 80/20. É o resultado consistente da comparação.
- **A vantagem do SVM em recall NÃO se repete:** no teste do 70/30, o SVM achava 3 a 4 doentes a
  mais que os baselines; no do 80/20, a logreg acha 2 a mais que o SVM (93 contra 91 de 102). As
  duas diferenças são de poucos pacientes (dentro do ruído), mas, juntas, mostram que a
  vantagem de recall do SVM é pequena e frágil. Na CV, o SVM continua com recall maior (0,910
  contra 0,870 e 0,876) nos dois splits.
- **Ressalva:** os dois testes vêm da mesma base, e ~70% dos pacientes do teste 80/20 estavam no
  treino do 70/30, usado para escolher a configuração. O teste 80/20 não é totalmente "novo"
  para as escolhas; se isso o deixa otimista, reforça que o SVM não supera os baselines.

**Conclusão do R10:** o desempenho do SVM final é estável entre os splits. A comparação com os
baselines também é estável no roc_auc (os baselines ficam acima); a vantagem do SVM em recall
aparece na CV, mas não de forma consistente nos testes. O argumento a favor do SVM final é o
perfil do modelo (4 variáveis, sem as variáveis ligadas ao hospital de origem, mais equilíbrio
entre os sexos que a logreg), não um desempenho superior.

## Limitações

- **Base combinada de 5 hospitais, sem a coluna de origem:** diferenças entre hospitais (como
  o colesterol não registrado na Suíça) podem virar atalhos. O desempenho em pacientes de um
  hospital novo não é estimado.
- **O atalho do colesterol foi medido (R2) e evitado:** o VALOR do colesterol praticamente não
  ajuda (importância por permutação -0,002 no baseline); o que ajuda é o exame estar AUSENTE
  (89% de doentes entre os 113 sem colesterol no treino, contra 48% entre os 529 registrados).
  A ausência vem quase toda da base suíça: a flag indica o hospital de origem, não fisiologia.
  O FastingBS parece carregar parte do mesmo efeito: é a variável mais associada à ausência do
  colesterol (V de Cramér 0,36), e sua importância cai pela metade (0,0084 -> 0,0046) quando a
  flag entra no modelo. As duas colunas foram retiradas (regra da mais simples, sem perda
  significativa). Consequência: o modelo final não usa colesterol nem glicemia de jejum, dois
  fatores de risco cardíaco conhecidos, por limitação da base e não por irrelevância clínica.
- **Possível perda pequena com a saída do FastingBS:** tirá-lo deu duas vezes uma queda de
  ~0,005 no roc_auc, no limite da significância (p = 0,085 sozinho; p = 0,078 junto com o
  colesterol). Um ganho real pequeno pode existir e não ser detectável com 642 pacientes no
  treino. Reavaliado na ablação do R6.
- **Otimismo da seleção:** o R3 testa 45 configurações (4 kernels) e escolhe a melhor; a nota de CV do
  vencedor tende a ser um pouco otimista (ele também venceu por acaso entre muitas). A
  estimativa honesta é a do teste final, avaliado uma única vez. Resultado: no teste, o
  roc_auc (0,910) ficou até um pouco acima da CV (0,899); o otimismo não se manifestou.
- **Teste pequeno para alguns casos:** só 1 Oldpeak negativo caiu em cada teste; o efeito
  dessa decisão só é medido na CV.
- **Especificidade baixa em pacientes com 55 anos ou mais (R9):** 0,560 contra 0,813 abaixo de
  55 anos (CV). A mesma diferença aparece na regressão logística do R0, então vem da base e não
  das escolhas do modelo. Na prática, o modelo gera mais falsos positivos (exames ou
  encaminhamentos desnecessários) em idosos.
- **Desempenho menor no grupo sem colesterol (R9):** roc_auc 0,859 contra 0,902. Esse grupo vem
  quase todo da base suíça: indício de que o modelo funciona pior em pacientes de outro
  hospital. Só 12 saudáveis no grupo: a especificidade dele (0,417) não é confiável.
- **Subgrupos pequenos:** 40 mulheres doentes e 12 pacientes saudáveis sem colesterol no treino.
  As diferenças nesses grupos são indícios, não conclusões.
- **O SVM final não supera os baselines (teste final e R10):** a regressão logística e o
  gradient boosting do R0, com todas as colunas, têm roc_auc maior nos dois splits, e a
  vantagem do SVM em recall não se repetiu no teste do 80/20. Parte da vantagem dos baselines
  pode vir das variáveis ligadas ao hospital de origem, que os testes (dos mesmos hospitais)
  não penalizam.
