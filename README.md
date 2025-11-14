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


Esta seção estabelece o ambiente de execução, garantindo a reprodutibilidade do projeto.

Verificação de Versões: O código importa e exibe as versões das principais bibliotecas utilizadas, como sys, pandas, numpy, sklearn, matplotlib, e keras. Esta é uma prática recomendada para evitar problemas de compatibilidade e documentar o ambiente.

Setup: O ambiente é configurado com a montagem do Google Drive (para acesso aos dados) e a importação das bibliotecas de visualização matplotlib.pyplot e seaborn para a criação de gráficos estatísticos, e pandas.plotting.scatter_matrix para análise de dispersão.

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


# 3. 🧠 Criação dos Dados de Treinamento


Nesta seção, os dados são estruturados para o treinamento do modelo.
1. Separação de Preditores e Alvo: A coluna target é separada para formar a variável alvo y, e o restante das colunas forma a matriz de características X. Ambas são convertidas em arrays NumPy.
2. Divisão Estratificada: O train_test_split divide o dataset em conjuntos de treino ($\approx 80\%$) e teste ($\approx 20\%$). O parâmetro stratify=y é crucial para garantir que a proporção da classe alvo seja mantida consistentemente nos subconjuntos.
3. Padronização (Scaling): O StandardScaler é usado para padronizar as características (média $\approx 0$, desvio padrão $\approx 1$). É aplicado ajustando-o apenas no conjunto de treino (fit_transform) e depois aplicado (transform) ao conjunto de teste (X_test), evitando vazamento de dados (data leakage).
4. Importância da Padronização: Este passo é vital para o desempenho de Redes Neurais, pois garante que todas as características contribuam igualmente para o cálculo do loss, facilitando a convergência do gradient descent (ver Seção 6).

# 4. 📈 Treinamento e Otimização da Rede Neural


Esta etapa foca na definição, treinamento e otimização do modelo de Deep Learning para a classificação binária.                                                                                             


**Arquitetura Base (Modelo 1)**
1. Estrutura: Modelo Sequencial com duas camadas ocultas ([16] -> [8] neurônios) e ativação ReLU. Uma camada de Dropout(0.2) é adicionada para prevenir overfitting inicial.
2. Camada de Saída: 1 neurônio com ativação sigmoid (ideal para classificação binária).
3. Compilação: Função de perda binary_crossentropy, otimizador adam, e métrica accuracy.


**Treinamento (Modelo 2)**
1. Nova Estrutura: Uma terceira camada oculta ([16] -> [8] -> [4] neurônios) é adicionada na tentativa de capturar relações mais complexas.
2. Alvo Binário: O alvo é redefinido para garantir que todos os casos de doença sejam rotulados estritamente como 1 (ausência como 0), essencial para a função binary_crossentropy.
3. Treinamento: O modelo é treinado (model.fit) por 50 épocas com um batch_size de 10. O validation_data (X_test, Y_test) é usado para monitorar o desempenho.
4. Análise de Curvas: As curvas de Acurácia e Loss (train vs. test) são plotadas. Uma divergência crescente entre as curvas (train subindo/loss caindo e test estabilizando/loss subindo) é um indicativo de overfitting.


**Otimização e Regularização (Modelo Final)**
1. Técnicas de Regularização: O modelo é otimizado com a introdução de regularização L2 (regularizers.l2(0.001)) nas camadas densas e um aumento no Dropout (de 0.2 para 0.25). O learning rate do otimizador Adam é ajustado para 0.001.
2. Objetivo: Penalizar pesos grandes, forçando o modelo a ser mais simples e reduzindo o overfitting para melhorar a generalização.
3. Treinamento Otimizado: O modelo final com regularização é treinado novamente e suas curvas de Acurácia e Loss são plotadas para confirmar se as técnicas de otimização reduziram a lacuna entre o desempenho de treino e teste.

# 5. ✅ Avaliação Final do Modelo


A avaliação final é realizada no conjunto de teste para determinar a eficácia do modelo otimizado.


| Métrica                  | Descrição                                                                                                                           | Relevância no Contexto Médico                                                                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Previsões                | A saída sigmoid (probabilidades) é convertida em classes binárias (0 ou 1) através do arredondamento.                               | Classe final de previsão.                                                                                                                                                        |
| Relatório de Classificação | Fornece Precisão, Recall e F1-Score por classe.                                                                                    | Permite uma análise granular da performance.                                                                                                                                     |
| Recall (Sensibilidade)   | Proporção de casos Positivos Reais que foram corretamente identificados.                                                             | **Crucial para problemas médicos.** Um Recall alto minimiza Falsos Negativos (FN) — paciente doente diagnosticado como saudável — que podem ter consequências graves.           |
| Matriz de Confusão       | Visualizada como um Heatmap, compara as previsões do modelo com os valores verdadeiros (VP, VN, FP, FN).                            | Ferramenta fundamental para entender a natureza dos erros do modelo e calcular as métricas de desempenho.                                                                       |


**Componentes da Matriz de Confusão:**


1. Verdadeiros Positivos (VP): Doente, previsto como Doente (Acerto).
2. Verdadeiros Negativos (VN): Saudável, previsto como Saudável (Acerto).
3. Falsos Positivos (FP - Erro Tipo I): Saudável, previsto como Doente (Erro).
4. Falsos Negativos (FN - Erro Tipo II): Doente, previsto como Saudável (Erro Crítico).

# 6. 📝 Conclusão sobre a Eficácia e a Importância da Normalização
**Eficácia do Modelo**


A eficácia do modelo é validada se o Recall para a classe 'Doente' for alto e se o modelo otimizado (com L2 e Dropout) apresentar uma menor diferença entre o desempenho de treino e teste em comparação com o modelo base. Isso indica que:
1. O modelo está detectando corretamente a maioria dos casos de doença (alto Recall).
2. O modelo tem uma alta capacidade de generalização, ou seja, funciona bem com novos pacientes.


**Importância da Normalização/Padronização dos Dados**


A padronização com o StandardScaler é de importância crítica para o sucesso das Redes Neurais:
1. Contribuição Equitativa: Garante que todas as características (variáveis clínicas) contribuam igualmente para o cálculo da perda (loss).
2. Estabilidade do Treinamento: Sem a normalização, características com escalas muito diferentes (ex: idade vs. colesterol) dominariam as atualizações de peso durante o gradient descent.
3. Consequência: Isso levaria a um processo de aprendizado lento, instável e, frequentemente, a uma convergência para resultados subótimos ou a um modelo que não generaliza bem. A padronização coloca todas as variáveis em uma escala comparável (média $\approx 0$, desvio padrão $\approx 1$), acelerando a convergência e melhorando a robustez.

## 📄 Licença

Este projeto é de uso acadêmico e foi desenvolvido exclusivamente para fins educacionais no contexto da disciplina.

## 🏛️ Universidade

**Universidade Federal do Amazonas (UFAM)**  
**Instituto de Computação (IComp)**

✳️ *Manaus, 2025*
