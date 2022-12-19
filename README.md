# Data Master Case - Machine Learning Engineer
## Treinamento e Deploy de Modelos de Deep Learning utilizando Databricks + Spark + TensorFlow + MLFlow em ambiente Cross Cloud – Microsoft Azure e Google Cloud Plataform


Por: Marcos Vinícius Lisboa Melo

Este repositório visa condensar as informações pertinentes a arquitetura, desenvolvimento de código e ambiente para desenho e execução do case para obtenção da certificação Data Master na carreira de Engenheiro Machine Learning. Para uma melhor organização e facilitação da replicação de toda a estrutura desenvolvida cada branch possui os códigos e READMEs específicos para build e utilização de cada ambiente.
As branchs estão dispostas da seguinte maneira:
* **_master:_ Contém a documentação e Descrição de ambientes** 
* **_train_environment:_ Notebook de treinamento Databricks Azure**
* **_deploy_environment:_ Notebook de deploy Databricks GCP**
* **_request_environment:_ Notebook de Teste Local - Jupyter**

As informações estarão dispostas de acordos com os tópicos a seguir:

[1. Introdução](#1-introdução)\
[2. Motivação](#2-motivação)\
[3. Arquitetura Técnica e de Dados _(BluePrints)_](#3-arquitetura-de-solução)\
[4. ARQUITETURA TÉCNICA E DE DADOS](#4-arquitetura-técnica-e-de-dados)\
&nbsp;&nbsp;  [4.1 _spark-tensorflow-distributor_](#41-spark-tensorflow-distributor)\
&nbsp;&nbsp;  [4.2 _MirroredStrategyRunner_](#42-mirroredstrategyrunner)\
&nbsp;&nbsp;  [4.3 MLFlow API - PyFunc](#43-mlflow-api---pyfunc)\
&nbsp;&nbsp;  [4.4 Infraestrutura Azure](#44-infraestrutura-azure)\
&nbsp;&nbsp;  [4.5 Infraestrutura GCP](#45-infraestrutura-gcp)\
[5. Dataset e Arquitetura CNN](#5-dataset-e-arquitetura-cnn)\
[6. Treinamento Databricks Azure](#6-treinamento-databricks-azure)\
&nbsp;&nbsp;  [6.1 Treinamento Single Node](#61-treinamento-single-node)\
&nbsp;&nbsp;  [6.2 Treinamento Distribuido](#62-treinamento-distribuido)\
&nbsp;&nbsp;&ensp;    [6.2.1 Treinamento Distribuido: Local Mode](#621-treinamento-distribuido-local-mode)\
&nbsp;&nbsp;&ensp;    [6.2.2 Treinamento Distribuido: Distributed Mode](#622-treinamento-distribuido-distributed-mode)\
&nbsp;&nbsp;&ensp;    [6.2.3 Treinamento Distribuido: Custom Mode](#623-treinamento-distribuido-custom-mode)\
[7. Model Logging e Registry via MLFlow](#7-model-logging-e-registry-via-mlflow)\
[8. Integração via MLFlow API](#8-integração-via-mlflow-api)\
&nbsp;&nbsp;  [8.1 Criação de Scope e Secrets Databricks](#81-criação-de-scope-e-secrets-databricks)\
&nbsp;&nbsp;  [8.2 Utilização de Client para Conexão com Repositórios Remotos](#82-utilização-de-client-para-conexão-com-repositórios-remotos)\
[9. Deploy DAtabricks GCP](#9-deploy-databricks-gcp)\
[10. Requisições de Teste na API Deployada](#10-requisições-de-teste-na-api-deployada)\
[11. Proposta de Evolução Técnica](#11-proposta-de-evolução-técnica)\
[12. Conclusões](#12-conclusões)\
[13. Referências](#13-referências)

# 1. Introdução

Esta documentação tem como objetivo contemplar técnica e funcionalmente a construção e implementação do Case para a Certificação interna **Data Masters** com referência a carreira de **_Machine Learning Engineer_**. Nos capítulos a seguir estarão descritos, em detalhes, desde a motivação de uso das tecnologias mencionadas no título como também a construção técnica necessária para atingir o resultado pretendido de treinar modelos utilizando **TensorFlow** com a estratégia distribuída do **Spark** e realizar o deploy destes modelos em outro **Workspace Databricks** de um provedor de Cloud diferente, utilizando dos recursos de MLOps disponíveis no **MLFlow**. Também estarão listadas na antepenúltima seção propostas de melhoria para que efetivamente possa se incorporar o desenvolvido ao pipeline de Modelos do Santander.

# 2. Motivação

O case segue, conforme a instrução de execução, os seguintes cenários de implementação: 
* **A5-)** Servir um modelo escolhido do Kaggle de Imagem; 
* **B2-)** Demonstre como seria o pipeline de implantação de um modelo implementado;
* **C1-)** Demostre por que uma ferramenta de orquestração de modelo pode ajudar nas implantações de modelos;
* **D2-)** Demonstre a importância dos pipelines de DEVOPs para a implantação de modelos.

A motivação deste case, se dá em dois pontos principais: demonstrar e instrumentar o treinamento de modelos de Deep Learning, especificamente modelos de Convolutional Neural Network (CNN) para Visão Computacional, utilizando o TensorFlow (spark-tensorflow-distributor) alinhado com o recurso de distribuição de tarefas do Spark (MirroredStrategyRunner) e em sequência o deploy de modelos de ML utilizando o fluxo do MLFlow, cross-workspaces e cross-cloud – reforçando a proposta de estarmos sempre agnósticos a um único provedor, simulando o real fluxo de deploy do banco de um workspace de Sandbox para um workspace de esteira produtiva. Além disso, validar a possibilidade de fazer deploy dos mesmos modelos multi-cloud, possibilitando manter os serviços críticos de forma reduntante, neste case foram utilizados os seguintes provedores de Cloud: Microsoft Azure ® e Google Cloud Plataform ®.

A escolha destes provedores foi motivada pela disponibilidade de recursos, nestes foi possível, por meio de parceria, criar workspaces com alguns recursos (créditos) para execução dos clusters sem dificuldades. A ideia core deste case, é que, o desenvolvido aqui, seja facilmente acoplável aos pipelines existentes do banco, podendo ser um potencializador dos nossos negócios e ferramentas, principalmente retirando a necessidade de realizar uma implantação de MLFlow paralela, podendo aproveitar os recursos já disponíveis nativamente na plataforma. Importante destacar que este case possibilita validar uma possível utilização da Databricks no fluxo de vida completo de um modelo de Machine Learning.

# 3. Arquitetura de Solução

Como proposta de arquitetura de solução, tem-se uma arquitetura híbrida Azure-GCP. Em ambos os provedores é instanciando uma workspace Databricks com disponibilidade para criação de clusters conforme a necessidade. É proposto para solução a utilização da workspace Azure como Sandbox para treinamento e registro do modelo de CNN, utilizando clusters com recurso de GPU e a biblioteca do TensorFlow como auxiliadora para o uso de imagens/matrizes. 

Já para a GCP, a workspace utilizada trata-se de um cluster sem GPU, pois não é necessário o recurso gráfico para otimização do deploy desta aplicação, considerando um cenário no qual a API Rest tem uma quantidade de requisições limitadas. Poderia ser criado um cluster com GPU caso fosse necessário que a aplicação respondesse com maior performance ou para um número maior de chamadas simultâneas. A Figura 2 mostra o desenho completo da solução end-to-end.

![arquitetura_solucao](https://user-images.githubusercontent.com/37118856/206886651-56002d26-4f51-442c-9514-40f9e269cb8c.jpg)

# 4. Arquitetura Técnica e de Dados

Tecnicamente é possível destacar alguns pontos principais e fundamentais para que a solução proposta funcione de maneira correta e eficiente. Neste aspecto é importante destacar o modo de funcionamento específico dos três componentes-chaves propostos: **_spark-tensorflow-distributor, MirroredStrategyRunner_** e **_MLFlow API_**. Como também, as definições técnicas de Infra para cada ambiente cloud.

  ## 4.1 _spark-tensorflow-distributor_
  
O _spark-tensorflow-distributor_ é um pacote native de TensorFlow de código aberto que auxilia os desenvolvedores de modelos de IA (cientistas de dados e engenheiros de ML) a distribuir o treinamento dos modelos TF em clusters Spark. Abaixo tem-se um desenho de arquitetura técnica que explica em mais detalhes o funcionamento do pacote.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/37118856/208334328-2a68cdb5-aa4d-4d3c-b23d-1dffe5826695.png">

  
  ## 4.2 _MirroredStrategyRunner_
  ## 4.3 MLFlow API - PyFunc
  ## 4.4 Infraestrutura Azure
  
Foi definida para implementação em uma worksapce Databricks instanciada na Azure uma estrutura com dois clusters, um cluster de apoio para evitar o custo excessivo em tarefas triviais e um cluster de treinamento de modelos, munido de GPU Tesla. A tabela abaixo contém a descrição detalhada dos clusters e os correspondentes recursos criados.

<img width="231" alt="image" src="https://user-images.githubusercontent.com/37118856/208329993-53da5029-9ce7-4186-8f16-e8c0861244ff.png">

Para armazenamento do token de conexão entre workspaces foi criado escopo **data_masters** juntamente com a secret/key **data_master_sandbox**.
  
  ## 4.5 Infraestrutura GCP
  
Para implementação na workspace Databricks GCP, apenas é necessário um único cluster para teste e deploy do modelo via MLFlow na forma de API REST. Este cluster não necessita, obrigatoriamente, de GPU, o modelo pode ser deployado em um cluster apenas com CPU, e a depender da necessidade por velocidade e processamento da aplicação a ser utilizada, pode ser utilizado o recurso de placas gráficas para acelerar o desempenho.
  
  <img width="241" alt="image" src="https://user-images.githubusercontent.com/37118856/208330032-69a19d4f-b7d2-4d26-b8ec-2840dfa70b65.png">
 
Importante salientar também que foi utilizado neste workspace o recurso de serving via API de modelos, que realiza o deploy utilizando o modelo registrado no MLFlow invocando as APIs padrão, neste caso em específico do TFX para deploy de um modelo Tensorflow.

Para armazenamento do token de conexão entre workspaces foi criado escopo **data_masters_gcp** juntamente com a secret/key **data_master_deploy**.


# 5. Dataset e Arquitetura CNN

Foi utilizado o conhecido dataset chamado MNIST - _Modified National Institute of Standards and Technology_, composto por cerca de 70.000 imagens de algarismos numéricos escritos a mão e pode ser utilizado de forma aberta para o desenvolvimento de modelos de reconhecimento de números utilizando técnicas de visão computacional.

![image](https://user-images.githubusercontent.com/37118856/206886708-a049ffd8-9af0-4ac7-a5b4-d5ebc5e3d19b.png)


# 6. Treinamento Databricks Azure

Foi construído um cenário de Sandbox dentro de uma workspace Databricks Azure, para a construção de modelos de Deep Learning. A escolha do provedor Azure neste cenário específico foi em decorrência dos recursos disponíveis neste ambiente, e a liberdade para o consumo de créditos, acordado previamente.

O cenário aqui implementado consistiu no consumo do MNIST, mencionado na sessão anterior, e utilizando-se uma Rede Convolucional (CNN) com poucas camadas (diminuindo o tempo e recursos para treinamento), para validar o cenário de implantação com um modulo funcional.

Para treinamento utilizando os recursos de GPU e distribuição de tarefas entre os workers existem 4 modos de treinamento conforme detalhados nos tópicos a seguir.


  ## 6.1 Treinamento Single Node
  
O treinamento Single Node consiste na modalidade mais simples de treino, na qual o recurso é completamente alocado no driver, se esse possuir uma GPU a mesma pode ser utilizada para acelerar o treinamento, entretanto a paralelização de tarefas ocorrerá somente dentro das threads da GPU.

  ## 6.2 Treinamento Distribuido
  
O treinamento distríbuido permite a utilização mais de um nó para paralelização da execução das tarefas de treinamento, podendo multiplexar o treinamento de vários modelos simultaneamente, ou até mesmo de um único modelo cross-workers.

  ### 6.2.1 Treinamento Distribuido: Local Mode
  
  
  ### 6.2.2 Treinamento Distribuido: Distributed Mode
  
  
  ### 6.2.3 Treinamento Distribuido: Custom Mode
  
  
# 7. Model Logging e Registry via MLFlow


# 8. Integração via MLFlow API


  ## 8.1 Criação de Scope e Secrets Databricks
  
  
  ## 8.2 Utilização de Client para Conexão com Repositórios Remotos
  
  
# 9. Deploy Databricks GCP


# 10. Requisições de Teste na API Deployada

Para testar o ambiente desenvolvido foi necessário localmente desenvolver um outro notebook responsável por montar a requisição a partir de uma imagem e via POST receber o response do modelo deployado. Seguindo a arquitetura da figura abaixo:

<img width="600" alt="image" align="center" src="https://user-images.githubusercontent.com/37118856/206933989-349fed2f-b7c2-44a2-8ae3-0f394c177ad3.png">

Este notebook é responsável por montar a requisição no formato pré-definido na documentação do TFX, seguindo o padrão:

E após a chamada tratar o retorno (numpy array) para encontrar o maior valor que corresponde a predição do modelo nas posições de 0 a 9.

# 11. Proposta de Evolução Técnica

Existem oportunidades de melhoria para o desenho de solução implementado, como por exemplo, a utilização de terraform para criação dos clusters de deploy dimensionados de forma mais inteligente para a demanda que o serviço terá. É possível também explorar em mais detalhes a estratégia custom de treinamento possibilitando orquestrar de maneira mais eficiente os recursos necessários para um treinamento end-to-end de um modelo de Deep Learning.

Para avaliação de performance comparativa entre o uso de GPU ou apenas de CPU para treinamento seria necessário evoluir o modelo, adicionando uma maior quantidade de camadas, e épocas em seu treinamento, ganhando assim uma robusteza maior em relação a tempo de treino e acurácia. Haja vista que o foco deste case não era a evolução dentro do treinamento do modelo, e o tempo para realização era limitado, a construção do pipeline foi priorizada.

# 12. Conclusões

No desenvolvimento deste case foi possível constatar a importância de possuir as ferramentas de consumo, preparação, modelagem e deploy da forma mais agnóstica possível. Sendo possível facilmente alternar entre provedores de cloud, quando necessário, e utilizando um fluxo de MLOps/DevOps para de forma eficiente gerir todo o ciclo de vida de desenvolvimento de modelos de ML.

Ademais, para futuras abordagens em problemas de Deep Learning a estratégia de processamento distribuído do Spark e Tensorflow é altamente recomendada, podendo ser um grande catalizador da modelagem DL.

Importante constatar também que, o MLFlow mais uma vez se mostra um importantíssimo e poderosíssimo aliado do Engenheiro de Machine Learning e do Cientista de Dados, e a integração nativa do MLFlow com a plataforma Databricks permite um grande aumento de velocidade para os times de Dados realizarem o deploy dos seus pipelines.


# 13. Referências

Site: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/train-model/distributed-training/spark-tf-distributor

Site: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/multiple-workspaces

Site: https://www.databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html

Site: https://www.mlflow.org/docs/latest/rest-api.html

