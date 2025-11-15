# Trabalho de Redes Neurais

Este repositório contém o projeto **"Classificação de Doenças Cardíacas"**, desenvolvido como parte das atividades acadêmicas da disciplina de Fundamentos de Inteligência Artificial no Instituto de Computação da Universidade Federal do Amazonas (IComp/UFAM).

## 👥 Equipe

| Nome | E-mail |
|------|---------|
| Anna Luisa Antony Afonso | anna.antony@icomp.ufam.edu.br |
| Beatriz Quaresma Athaide | beatriz.quaresma@icomp.ufam.edu.br |
| Elaine de Castro Freire | elaine.freire@icomp.ufam.edu.br |
| Manuela Figueira Batista | manuela.batista@icomp.ufam.edu.br |
| Raíssa Clara Teixeira Brasil | raissa.brasil@icomp.ufam.edu.br |
| Ruthelene Rodrigues Farias | ruthelene.farias@icomp.ufam.edu.br |

# 🫀 Classificação de Doenças Cardíacas com Redes Neurais


Este projeto implementa e avalia um modelo de Rede Neural Sequencial (utilizando Keras) para a classificação binária de doença cardíaca com base em dados clínicos. O objetivo é configurar um ambiente robusto, limpar e pré-processar o dataset Cleveland, treinar um modelo de Deep Learning e otimizá-lo com técnicas de regularização para garantir a capacidade de generalização. 

# 1. 🛠️ Inicialização e Carregamento de Dados

Este bloco de código marca o início de qualquer projeto robusto de Machine Learning (ML), estabelecendo as bases do ambiente de software e preparando as ferramentas necessárias para as fases de pré-processamento, análise e modelagem.

***

### Verificação e Gerenciamento de Dependências

O primeiro objetivo do código é importar todas as bibliotecas Python que servirão como a espinha dorsal do projeto. Imediatamente após a importação, o código imprime as versões de cada uma das principais ferramentas. Essa prática de **versionamento** é crucial para garantir a **reprodutibilidade** do ambiente. Caso o código precise ser executado em outra máquina ou em uma data futura, ter as versões exatas documentadas permite diagnosticar e prevenir erros de compatibilidade.

As bibliotecas importadas e verificadas incluem:

* **`sys`:** Essencialmente para obter informações sobre o ambiente de execução, especialmente a versão do **Python** em uso.
* **`pandas`:** A ferramenta primária para lidar com **dados tabulares**, convertendo dados brutos em estruturas DataFrames fáceis de manipular, limpar e analisar.
* **`numpy`:** Fornece a base para **operações numéricas** de alto desempenho, sendo o formato de array fundamental que a maioria dos algoritmos de ML consome.
* **`sklearn` (Scikit-learn):** A biblioteca mais popular para tarefas de ML clássico, oferecendo uma vasta gama de algoritmos de **classificação**, **regressão** e **agrupamento**, além de utilitários para pré-processamento.
* **`matplotlib`:** A biblioteca de referência para a **criação de gráficos** e visualizações de dados, permitindo a construção de histogramas, gráficos de linha e gráficos de barras.
* **`keras`:** Uma interface de alto nível usada para construir e treinar **Redes Neurais** de maneira eficiente e amigável.

***

### Preparação para Visualização e Acesso a Dados

A seção seguinte, conforme descrito na sua explicação, foca em configurar o acesso aos dados e armar o ambiente com as ferramentas de Análise Exploratória de Dados (EDA).

* **Montagem do Google Drive:** Em plataformas de *notebook* baseadas em nuvem, este comando é fundamental para **conectar o ambiente de código** aos arquivos de dados do projeto, permitindo que os dados brutos sejam carregados.
* **Importações Gráficas Avançadas:**
    * As importações de **`matplotlib.pyplot`** e **`seaborn`** são feitas para equipar o projeto com capacidades de visualização de dados. O `seaborn` é construído sobre o `matplotlib` e permite criar **gráficos estatísticos complexos** com menos código, sendo ideal para a EDA.
    * A importação de **`pandas.plotting.scatter_matrix`** é um atalho prático para gerar uma **matriz de gráficos de dispersão**, o que permite inspecionar rapidamente as relações de correlação entre todos os pares de variáveis numéricas do *dataset*.

Em resumo, o bloco de "Inicialização e Carregamento de Dados" está configurando o *pipeline* do projeto. Ele verifica a integridade do ambiente, garante que o acesso à fonte de dados esteja estabelecido e importa as bibliotecas especializadas que serão usadas nas próximas etapas de Análise Exploratória e Modelagem.

# 2. 📊 Importação e Exploração do Dataset
O foco desta etapa é a limpeza, transformação e análise exploratória dos dados brutos.


| Etapa                     | Descrição                                                                                                                                                                                                                                                                          | Importância                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Carregamento e Limpeza    | O dataset é carregado (ex: 303 amostras, 14 colunas). Valores ausentes, codificados como '?', são substituídos por NaN e, em seguida, as linhas com valores nulos são removidas (resultando em ≈297 linhas).                                                                        | Garante a integridade dos dados e remove inconsistências.                                        |
| Transformação de Tipo     | Todas as colunas são convertidas do tipo object para tipos numéricos (int64 ou float64), passo fundamental para permitir o cálculo e a modelagem.                                                                                                                                   | Essencial para o uso em algoritmos de Machine Learning.                                         |
| Estatísticas Descritivas  | data.describe() gera um resumo estatístico (mean, std, min, max), crucial para identificar a escala das variáveis e planejar o scaling.                                                                                                                                             | Auxilia na detecção de outliers e no pré-processamento.                                         |
| Análise de Distribuição   | data.hist() plota histogramas para a análise visual da distribuição de frequência de cada variável.                                                                                                                                                                                  | Essencial para verificar o balanceamento da classe alvo (target).                               |
| Análise de Correlação     | A matriz de correlação de Pearson é calculada e visualizada em um Heatmap.                                                                                                                                                                                                          | Identifica os preditores mais fortes (alta correlação com target) e a multicolinearidade.       |
| Análise Específica        | pd.crosstab e gráficos de barras/pontos exploram relações cruciais (ex: casos positivos/negativos por idade, e a tendência de thalach (frequência cardíaca máxima) em relação à idade).                                                                                              | Oferece insights diretos e valida a coerência fisiológica dos dados.                             |

---
### Detalhamento das Etapas de Pré-processamento

#### 1. Limpeza e Integridade dos Dados

O processo de carregamento utiliza o `pandas` para ler o arquivo `heart.csv`. A inspeção inicial revela que o *dataset* contém valores ausentes codificados de forma não padrão (o caractere **'?'**). O código soluciona este problema **substituindo** e, em seguida, **removendo** as linhas que continham esses valores inválidos (`dropna`). Adicionalmente, a remoção de **linhas duplicadas** é realizada para garantir que o modelo não seja treinado com informações redundantes, o que poderia enviesar a avaliação de desempenho.

#### 2. Normalização de Tipos e Estatística Descritiva

A conversão de tipos de dados (`data.apply(pd.to_numeric)`) é um passo não negociável no pré-processamento, pois garante que todas as colunas estejam em formato **numérico** (`int64` ou `float64`), requisito básico para o cálculo de distância e otimização em algoritmos de Machine Learning. Uma vez limpo e tipado, o DataFrame é resumido pelo **`data.describe()`**. Este resumo estatístico é inspecionado para:
* Confirmar a **escala** de cada variável.
* Avaliar a **dispersão** (`std`).
* Identificar **valores extremos** (*outliers*) nos valores mínimos e máximos, que podem exigir tratamento especializado (como *winsorizing* ou logaritmização) posteriormente.

---

### Detalhamento da Análise Exploratória (EDA)

A EDA é a fase visual e estatística que fundamenta as decisões de modelagem:

#### Distribuições e Frequências
A geração de **histogramas** (`data.hist()`) é utilizada para visualizar a distribuição de frequência de cada variável. Esta análise é essencial para entender se os dados seguem uma **distribuição normal** e, sobretudo, para inspecionar o **balanceamento da variável alvo (`target`)**. Um *target* desbalanceado (onde uma classe é muito mais frequente que a outra) pode levar o modelo a ser enviesado, necessitando de técnicas de reamostragem como SMOTE.

#### Análise de Relações e Coerência
* **Correlação (Heatmap):** A matriz de correlação de Pearson, exibida como um **Heatmap**, permite identificar de forma rápida quais variáveis têm a **maior relação linear** com o `target`. Além disso, é o principal método para detectar a **multicolinearidade** (correlação alta entre dois preditores independentes), que pode inflacionar a variância e prejudicar a interpretabilidade de modelos como Regressão Logística.
* **Idade vs. Doença (`pd.crosstab`):** O cruzamento da variável categórica `age` com o `target` é plotado para identificar as **faixas etárias de maior risco**. Este *insight* direto valida a importância preditiva dessa *feature*.
* **Validação Fisiológica (`thalach` vs. Idade):** A plotagem da frequência cardíaca máxima (`thalach`) em função da idade é uma etapa de **validação de qualidade**. A expectativa é que essa variável **diminua** com o avanço da idade, um comportamento fisiológico conhecido. A confirmação dessa tendência no *dataset* atesta a coerência e a integridade dos dados coletados.

# 3. 🧠 Criação dos Dados de Treinamento

Este bloco de código é o ponto de transição crucial entre a Análise Exploratória de Dados (EDA) e a fase de Modelagem. Ele cobre as etapas essenciais de **estruturação, divisão e escalonamento** das variáveis, garantindo que o *dataset* esteja no formato ideal para o treinamento e a avaliação dos algoritmos de Machine Learning.

### 1. Separação de Variáveis e Conversão para NumPy

A primeira ação do código é estabelecer o problema de classificação através da separação formal das variáveis. A coluna **`target`** é isolada para formar a variável **y** (o rótulo, ou o que deve ser previsto), enquanto todas as demais colunas do DataFrame, que representam as características clínicas, são agrupadas na matriz **X** (os preditores). O código, em seguida, realiza a conversão imediata de **X** e **y** para **NumPy Arrays**. Essa conversão é um requisito técnico fundamental, pois o NumPy é o formato de matriz de alto desempenho exigido por praticamente todas as bibliotecas de Machine Learning, como Scikit-learn e Keras, otimizando o consumo de memória e a velocidade dos cálculos. A inspeção inicial de `X[0]` confirma que o processo de separação ocorreu com sucesso e que os dados numéricos estão estruturados corretamente.

### 2. Divisão Estratificada dos Dados (*Train-Test Split*)

A próxima etapa é a divisão dos dados em conjuntos de **treinamento (`X_train`, `y_train`)** e **teste (`X_test`, `y_test`)**. O padrão utilizado é de **80%** dos dados para treinamento e **20%** para teste (`test_size=0.2`). Esta separação é vital para que o modelo seja treinado em uma porção dos dados e avaliado em uma porção **inédita**, fornecendo uma estimativa imparcial de seu desempenho em novos dados. O uso do parâmetro **`stratify=y` é absolutamente crítico** neste contexto de classificação. Ele assegura que a **proporção da classe alvo** (pacientes com e sem a doença cardíaca) seja **mantida de forma idêntica** nos subconjuntos de treino e teste.  Sem a estratificação, o conjunto de teste poderia, por acaso, conter uma proporção desequilibrada das classes, resultando em uma avaliação de desempenho irrealista ou tendenciosa.

### 3. Padronização de Características (*Standard Scaling*) e Prevenção de *Data Leakage*

A etapa final e mais sofisticada de pré-processamento é a **padronização** das características (`StandardScaler`). Este método transforma os dados de forma que cada característica tenha uma **média próxima de zero e um desvio padrão próximo de um**. Isso é essencial para algoritmos que calculam distâncias entre pontos (como k-Nearest Neighbors, KNN) ou para modelos baseados em otimização por gradiente (como Redes Neurais), pois impede que *features* com escalas naturalmente maiores (como a idade) dominem o processo de aprendizado.

O processo de *scaling* é aplicado em duas fases obrigatórias para **prevenir o vazamento de dados (*data leakage*)**:

1.  **Ajuste e Transformação no Treino:** O *scaler* é **ajustado** e **transformado** (`fit_transform`) **somente** no conjunto de treinamento (`X_train`). Isso significa que a média e o desvio padrão usados para a padronização são derivados **exclusivamente** dos dados de treino.
2.  **Transformação no Teste:** Os **mesmos parâmetros** (média e desvio padrão) aprendidos no conjunto de treino são, então, usados para **transformar** (`transform`) o conjunto de teste (`X_test`).

Essa separação garante que o modelo de avaliação (`X_test`) permaneça totalmente desconhecido em todas as etapas, simulando com precisão o cenário real onde o modelo encontrará dados novos. A verificação final do `X[0]` no código serve para confirmar que a matriz original **X** não foi modificada pelo `StandardScaler`, mantendo a integridade do *array* principal.      
## 📄 4 - Treinamento da Rede Neural

Esta documentação abrange o desenvolvimento completo do modelo de **Deep Learning**, desde a garantia de um ambiente **reprodutível** até a aplicação de **técnicas avançadas de regularização** e a **análise visual** do desempenho. O objetivo é criar um modelo robusto, com alta capacidade de generalização e evitar o *overfitting*.

---

### 4.1. Garantia de Reprodutibilidade (*Seeding*) e Preparação do Alvo

O bloco inicial é essencialmente técnico, focando na **garantia de reprodutibilidade** do experimento. A definição de uma **semente fixa (`SEED = 42`)** e sua aplicação sistemática nos geradores de números pseudo-aleatórios do **Python (`random`)**, **NumPy (`np.random`)** e **TensorFlow (`tf.random`)** asseguram que todas as execuções subsequentes do código, incluindo a inicialização dos pesos da rede e a sequência de *dropout*, resultarão nos mesmos valores. Isso permite a comparação consistente dos resultados e a validação do processo de otimização.

O bloco seguinte prepara a variável alvo (`y`) para o formato **binário estrito**. Cópias dos rótulos de treino (`y_train`) e teste (`y_test`) são criadas e, em seguida, todos os valores **maiores que zero** são explicitamente convertidos para **1** (`Doente`), enquanto a ausência de doença permanece **0** (`Saudável`). Esta conversão é **fundamental** para a utilização da função de perda **`binary_crossentropy`**, que exige rótulos de classe estritamente 0 ou 1. A impressão dos primeiros 20 elementos de `Y_train_binary` confirma a **integridade** desta conversão.

---

### 4.2. Definição da Arquitetura Otimizada (Regularização Reforçada)

O código define uma função construtora para a **Rede Neural Sequencial Otimizada** (`create_binary_model_tuned`), com um foco explícito no combate ao *overfitting* através de regularização avançada. A arquitetura é construída com três camadas ocultas, introduzindo maior profundidade e complexidade para a rede:

* **Camadas Ocultas:** São três camadas (`Dense`) com **16, 8 e 4 neurônios** respectivamente, todas utilizando a função de ativação **ReLU** (Rectified Linear Unit), ideal para camadas intermediárias. O `input_shape=(13,)` confirma o número de características de entrada após o pré-processamento.
* **Regularização L2 ($\ell_2$):** Um termo de penalidade $\ell_2(0.001)$ é aplicado aos *kernels* (pesos) de cada camada densa (`kernel_regularizer`). Esta técnica força os pesos a serem menores e mais esparsos, **mitigando o *overfitting*** ao simplificar o modelo.
* **Dropout Reforçado:** Uma alta taxa de **`Dropout(0.25)`** é aplicada entre *cada* camada densa. Este valor elevado aumenta a desativação aleatória de 25% dos neurônios durante o treinamento, o que impede a co-adaptação e reforça a generalização do modelo.
* **Compilação e Otimizador:** O modelo é compilado com a perda **`binary_crossentropy`** e o otimizador **Adam** configurado com um **`learning_rate` baixo (0.0005)**. Esta taxa de aprendizado reduzida permite um ajuste mais fino e estável dos pesos, otimizando a convergência.

O **`binary_model.summary()`** impresso no final do bloco fornece uma verificação da arquitetura, incluindo a contagem de parâmetros treináveis. 

---

### 4.3. Treinamento Controlado e *Early Stopping*

O treinamento do modelo é realizado sob um rigoroso controle através do *callback* **`EarlyStopping`**. Esta técnica é fundamental para **evitar o *overfitting***, parando o treinamento no ponto de máxima generalização.

* **Métrica Monitorada:** O *Early Stopping* monitora a **perda de validação (`val_loss`)**, que é a métrica mais sensível para detectar o início do *overfitting*.
* **Paciência (`patience=10`):** O treinamento será interrompido se a perda de validação não apresentar melhora após **10 épocas consecutivas**.
* **Restauração de Pesos:** O parâmetro **`restore_best_weights=True`** garante que, mesmo após a interrupção, o modelo retorne e utilize os pesos da época que resultou no **melhor desempenho de validação**, e não os pesos da última época, que já podem ter sofrido *overfitting* leve.

O modelo (`binary_model`) é treinado por no máximo **50 épocas** com um **`batch_size` de 32**. O uso do `validation_data` (conjunto de teste) junto ao `EarlyStopping` cria um processo de treinamento eficiente e robusto.

---

### 4.4. Análise Visual de Convergência e Generalização

O bloco final gera uma figura com dois gráficos de linha lado a lado para visualizar o desempenho do modelo, utilizando o histórico (`history`) registrado durante o treinamento:

1.  **Acurácia do Modelo:** Compara a acurácia no **conjunto de treinamento** com a acurácia no **conjunto de validação**.
2.  **Perda (*Loss*) do Modelo:** Compara a perda no **conjunto de treinamento** com a perda no **conjunto de validação**.

A **análise visual** destas curvas é o resultado final e a validação das otimizações. Uma **distância pequena (gap)** entre as curvas de treino e validação é a confirmação de que as técnicas de regularização (L2 e Dropout) foram eficazes. A interrupção precoce das curvas, se o *Early Stopping* foi acionado, confirma que o modelo parou antes de entrar em *overfitting* severo, resultando em um modelo **estável e com alta capacidade de generalização**. 

### 4.5. Avaliação Final da Robustez

Os blocos finais de código são dedicados à plotagem das curvas de desempenho do modelo otimizado (com L2 e *Dropout*):

1.  **Curva de Acurácia Otimizada:** A análise visual desta nova curva é essencial para determinar se as técnicas de regularização foram eficazes em suavizar as oscilações e **reduzir a lacuna (*gap*)** entre a acurácia de treino e a acurácia de validação. Uma lacuna menor indica um modelo que generaliza melhor para dados não vistos.
2.  **Curva de Perda Otimizada:** O objetivo final é observar se a perda de validação **não aumenta drasticamente** após um certo número de épocas, mantendo-se mais próxima da perda de treinamento. Se esta curva for notavelmente mais estável do que a do modelo base, confirma-se que a regularização **mitigou o *overfitting***, resultando em um modelo mais robusto e com maior poder preditivo em cenários reais. 

# 5. ✅ Avaliação Final do Modelo

Este bloco de código representa a fase final do ciclo de Machine Learning, onde o desempenho do modelo binário otimizado é **avaliado de forma abrangente e imparcial** utilizando o conjunto de teste (`X_test`), que permaneceu inédito. O foco está na análise de métricas que vão além da acurácia simples, sendo o **Recall** e a **Matriz de Confusão** os elementos centrais para a tomada de decisão em um contexto médico.

### Resumo das Ações de Avaliação e Métricas Chave

A avaliação final é realizada no conjunto de teste para determinar a eficácia do modelo otimizado, conforme detalhado abaixo:

| Métrica / Ação | Descrição Técnica | Relevância no Contexto Médico |
| :--- | :--- | :--- |
| **Geração de Previsões** | A saída Sigmoid (probabilidades contínuas) é convertida em classes binárias definitivas (**0** ou **1**) através da função de arredondamento (`np.round`). | Transforma a probabilidade do modelo na **classe final de previsão**, necessária para o cálculo de todas as métricas discretas. |
| **Acurácia Geral** | Mede a proporção total de previsões corretas (VP + VN) em relação ao total de amostras. | Fornece uma visão inicial da performance geral do modelo. |
| **Relatório de Classificação** | Fornece métricas detalhadas (Precisão, Recall e F1-Score) por classe. | Permite uma **análise granular** da performance, essencial para validar a Precisão e o Recall em desequilíbrio de classes. |
| **Recall (Sensibilidade)** | Proporção de casos **Positivos Reais** que foram corretamente identificados (VP / (VP + FN)). | **Métrica mais crucial:** Um **Recall alto** minimiza os **Falsos Negativos (FN)** — paciente doente diagnosticado como saudável — o que representa o **erro mais crítico** e de maior consequência em diagnóstico médico. |
| **Matriz de Confusão** | Visualizada como um Heatmap, compara as previsões do modelo com os valores verdadeiros (VP, VN, FP, FN). | Ferramenta fundamental para **entender a natureza e a distribuição dos erros** do modelo, servindo como base visual para a interpretação do Recall e da Precisão. |

---

### Detalhamento da Avaliação e Análise da Matriz de Confusão

#### Geração de Previsões e Acurácia Geral

O código inicia com a **geração das previsões** (`binary_pred`) aplicando uma função **`np.round`** na saída da camada Sigmoid. Esta etapa de arredondamento converte as probabilidades contínuas em classes binárias discretas (0 ou 1), permitindo o uso das métricas de classificação. Em seguida, a **Acurácia Geral** é impressa. Embora seja uma métrica inicial útil, ela é insuficiente e pode ser enganosa em *datasets* onde as classes não estão perfeitamente balanceadas, justificando o uso do Relatório de Classificação.

#### Relatório de Classificação e o Papel do Recall

O **Relatório de Classificação** (`classification_report`) é a principal fonte de métricas detalhadas. Ele apresenta o **F1-Score** (que equilibra Precisão e Recall), a **Precisão** (que mede a confiabilidade das previsões positivas) e o **Recall** para cada classe.

O **Recall** é a métrica mais crucial para este trabalho, também conhecido como Sensibilidade ou Taxa de Verdadeiros Positivos (TPR). Ele responde à pergunta: "De todos os pacientes que estavam realmente Doentes, quantos o modelo conseguiu detectar?". Um **Recall alto é vital** porque minimiza o **Falso Negativo (FN)**, que é o **Erro Tipo II** — o modelo prevê 'Saudável' quando o paciente está 'Doente'. Perder um diagnóstico positivo pode ter consequências graves, o que torna a minimização do FN a prioridade máxima do modelo. O Recall, por si só, não se preocupa com os Falsos Positivos, por isso ele é analisado em conjunto com a Precisão.

#### Matriz de Confusão e a Natureza dos Erros

A **Matriz de Confusão** (`confusion_matrix`) é calculada e visualizada como um **Heatmap** . Esta visualização tabular é fundamental para entender a natureza dos erros do modelo, comparando as previsões com a verdade real:

| Componente | Descrição Técnica | Classificação (Verdadeiro vs. Previsto) | Natureza do Erro |
| :--- | :--- | :--- | :--- |
| **Verdadeiros Positivos (VP)** | O modelo previu 'Doente' corretamente. | Real: Doente, Previsto: Doente | Acerto |
| **Verdadeiros Negativos (VN)** | O modelo previu 'Saudável' corretamente. | Real: Saudável, Previsto: Saudável | Acerto |
| **Falsos Positivos (FP - Erro Tipo I)** | O modelo previu 'Doente', mas o real era 'Saudável'. | Real: Saudável, Previsto: Doente | Erro Tipo I / Alarme Falso |
| **Falsos Negativos (FN - Erro Tipo II)** | O modelo previu 'Saudável', mas o real era 'Doente'. | Real: Doente, Previsto: Saudável | **Erro Crítico** |

A inspeção visual do *heatmap* permite quantificar diretamente os **FN** e **FP**, validando se as técnicas de regularização L2 e Dropout foram eficazes em manter o FN em níveis aceitáveis, garantindo que o modelo seja robusto e seguro.

# 6. 📝 Conclusão sobre a Eficácia e a Importância da Normalização

A eficácia final do modelo de classificação binária é determinada pela análise conjunta das **métricas de teste** (Acurácia, Precisão, Recall e F1-Score) e pela interpretação da **Matriz de Confusão**. Um modelo de alto desempenho é validado por dois critérios essenciais que foram perseguidos durante o treinamento e otimização:

1.  **Alta Sensibilidade (Recall para a Classe 'Doente'):** Em problemas de diagnóstico médico, a métrica mais crucial é o **Recall** para a classe positiva (Doente). A eficácia só é confirmada se o modelo apresentar um Recall elevado, garantindo que o número de **Falsos Negativos (FN)** seja minimizado. Isso significa que a rede neural está detectando corretamente a grande maioria dos casos de doença, priorizando a segurança e a intervenção precoce do paciente.
2.  **Robustez e Generalização:** A comparação das curvas de perda e acurácia entre o modelo base e o **modelo regularizado** (com L2 e Dropout) é o fator decisivo para a robustez. O modelo otimizado só é considerado eficaz se apresentar uma **menor diferença (gap)** entre o desempenho no conjunto de treino e no conjunto de teste. Uma lacuna reduzida indica que as técnicas de regularização foram bem-sucedidas em mitigar o *overfitting*, garantindo que o modelo não memorizou o ruído dos dados de treinamento e possui uma **alta capacidade de generalização** para classificar corretamente novos pacientes não vistos.

### Importância Crítica da Normalização (Padronização) dos Dados

A **normalização (ou padronização)** dos dados, realizada através do **`StandardScaler`**, é de importância crítica e fundamental para o sucesso das Redes Neurais e de muitos outros algoritmos baseados em distância. Sua relevância técnica é dupla e direta:

1.  **Contribuição Equitativa das Características:** A padronização transforma os dados, colocando todas as variáveis em uma escala comparável, onde a média é aproximadamente $0$ ($\mu \approx 0$) e o desvio padrão é aproximadamente $1$ ($\sigma \approx 1$). Isso garante que todas as características contribuam **equitativamente** para o cálculo do **loss (perda)**.
2.  **Estabilidade e Velocidade de Convergência:** Sem a normalização, características com valores de magnitude muito grande (por exemplo, colesterol, que pode ser $\approx 300$) dominariam as atualizações de peso durante o processo de treinamento via **Gradient Descent**. Essas *features* com grande escala levariam a gradientes acentuados e, consequentemente, a grandes saltos nos pesos. A **consequência direta** disso é um processo de aprendizado:
    * **Lento:** Otimizador gasta tempo navegando em um espaço de busca alongado.
    * **Instável:** As atualizações de peso oscilam violentamente.
    * **Subótimo:** O modelo frequentemente converge para um mínimo local inferior ou falha em generalizar bem.

Em suma, a padronização elimina a dependência da escala original dos dados, **acelera significativamente a convergência** do otimizador e **melhora a robustez** da Rede Neural, sendo uma etapa não negociável para alcançar a eficácia máxima do modelo. 

## 📄 Licença

Este projeto é de uso acadêmico e foi desenvolvido exclusivamente para fins educacionais no contexto da disciplina.

## 🏛️ Universidade

**Universidade Federal do Amazonas (UFAM)**  
**Instituto de Computação (IComp)**

✳️ *Manaus, 2025*
