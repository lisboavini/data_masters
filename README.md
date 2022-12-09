# Data Master Case - Machine Learning Engineer
## Treinamento e Deploy de Modelos de Deep Learning utilizando Databricks + Spark + TensorFlow + MLFlow em ambiente Cross Cloud – Microsoft Azure e Google Cloud Plataform


Por: Marcos Vinícius Lisboa Melo

Este repositório visa condensar as informações pertinentes a arquitetura, desenvolvimento de código e ambiente para desenho e execução do case para obtenção da certificação Data Master na carreira de Engenheiro Machine Learning.

As informações estarão dispostas de acordos com os tópicos a seguir:

1. INTRODUÇÃO
2. MOTIVAÇÃO
3. ARQUITETURA DE SOLUÇÃO (BluePrints)
4. ARQUITETURA TÉCNICA/DADOS\
  4.1 _spark-tensorflow-distributor_\
  4.2 _MirroredStrategyRunner_\
  4.3 MLFLOW API - PYFUNC\
  4.4 INFRAESTRUTURA AZURE\
  4.5 INFRAESTRUTURA GCP\
5. DATASET E ARQUITETURA CNN
6. TREINAMENTO DATABRICKS AZURE\
  6.1 TREINAMENTO SINGLE NODE\
  6.2 TREINAMENTO DISTRIBUIDO\
    6.2.1 TREINAMENTO DISTRIBUIDO: LOCAL MODE\
    6.2.2 TREINAMENTO DISTRIBUIDO: DISTRIBUTED MODE\
    6.2.3 TREINAMENTO DISTRIBUIDO: CUSTOM MODE
7. MODEL LOGGING AND REGISTRY VIA MLFLOW
8. INTEGRAÇÃO VIA MLFLOW API\
  8.1 CRIAÇÃO DE SCRETS/SCOPE DATABRICKS\
  8.2 UTILIZAÇÃO DE CLIENT PARA CONEXÃO COM REPOSITÓRIOS REMOTOS
9. DEPLOY DATABRICKS GCP
10. PROPOSTA DE EVOLUÇÃO TÉCNICA
11. CONCLUSÕES
12. REFERÊNCIAS
