# Anotações para o artigo

Decisões, achados e limitações, registrados ao longo das rodadas. Critérios de leitura que
dependem de um valor escolhido pelo grupo (ex.: especificidade mínima do R6) são registrados
ANTES de rodar.

## Decisões de projeto (antes das rodadas)

- **Alvo:** `HeartDisease` (doença cardíaca), não infarto especificamente.
- **Splits:** 70/30 estratificado, seed 42, como nos trabalhos anteriores (DecisionTree e
  RandomForest). O 80/20 é usado só no R11, como análise de sensibilidade (opção B do grupo):
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
- **Desbalanceamento:** pequeno (~55/45); não há rodada de reamostragem. O R6 trata o
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
  conferir no R10 o desempenho no subgrupo "colesterol registrado".
- R6: especificidade mínima aceitável = ______ (sugestão: 0,70).

## Rodadas

### R0 - Baselines

### R1 - Escalonamento

### R2 - Zeros e Oldpeak

### R3 - Kernel

### R4 - Hiperparâmetros

### R5 - Seleção de atributos

### R6 - Recall

### R7 - Combinação e ablação

### R8 - Calibração

### R9 - Interpretação (teste de sanidade)

### R10 - Robustez

### Teste final (70/30)

### R11 - Split 80/20

## Limitações

- **Base combinada de 5 hospitais, sem a coluna de origem:** diferenças entre hospitais (como
  o colesterol não registrado na Suíça) podem virar atalhos. O desempenho em pacientes de um
  hospital novo não é estimado.
- **Otimismo da seleção:** o R4 testa 40 configurações e escolhe a melhor; a nota de CV do
  vencedor tende a ser um pouco otimista (ele também venceu por acaso entre muitas). A
  estimativa honesta é a do teste final, avaliado uma única vez.
- **Teste pequeno para alguns casos:** só 1 Oldpeak negativo caiu em cada teste; o efeito
  dessa decisão só é medido na CV.
