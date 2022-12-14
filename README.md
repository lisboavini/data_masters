# Data Master Case - Machine Learning Engineer
## Treinamento e Deploy de Modelos de Deep Learning utilizando Databricks + Spark + TensorFlow + MLFlow em ambiente Cross Cloud – Microsoft Azure e Google Cloud Plataform


Por: Marcos Vinícius Lisboa Melo

Este repositório visa condensar as informações pertinentes a arquitetura, desenvolvimento de código e ambiente para desenho e execução do case para obtenção da certificação Data Master na carreira de Engenheiro Machine Learning. Para uma melhor organização e facilitação da replicação de toda a estrutura desenvolvida cada branch possui os códigos e READMEs específicos para build e utilização de cada ambiente.
As branchs estão dispostas da seguinte maneira:
* **_master:_ Contém a documentação, apresentação e descrição de ambientes** 
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
&nbsp;&nbsp;  [6.2 Treinamento Distribuído](#62-treinamento-distribuído)\
&nbsp;&nbsp;&ensp;    [6.2.1 Treinamento Distribuído: Local Mode](#621-treinamento-distribuído-local-mode)\
&nbsp;&nbsp;&ensp;    [6.2.2 Treinamento Distribuído: Distributed Mode](#622-treinamento-distribuído-distributed-mode)\
&nbsp;&nbsp;&ensp;    [6.2.3 Treinamento Distribuído: Custom Mode](#623-treinamento-distribuído-custom-mode)\
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
* **A5-)** Servir um modelo escolhido do Kaggle (ou outro repositório) de Imagem; 
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
  
O `_spark-tensorflow-distributor_` é um pacote native de TensorFlow de código aberto que auxilia os desenvolvedores de modelos de IA (cientistas de dados e engenheiros de ML) a distribuir o treinamento dos modelos TF em clusters Spark. Abaixo tem-se um desenho de arquitetura técnica que explica em mais detalhes o funcionamento do pacote.

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/37118856/208334328-2a68cdb5-aa4d-4d3c-b23d-1dffe5826695.png">

  
  ## 4.2 _MirroredStrategyRunner_
  
Dentro do pacote mencionado no tópico anterior é possível encontrar um método chamado `MirroredStrategyRunner()`, que dentro de si encapsula as funções necessárias para definir o número de slots, tipo de treino e quantidade de GPUs utilizadas. Ele tem a função embarcada de realizar as conexões e divisão de paralelismo entre os nodes, e é o principal componente do pacote de distribuição.
  
  ## 4.3 MLFlow API - PyFunc
  
O MLflow tem como intuito oferecer suporte ao ciclo de vida de projetos de ML, mas também disponibiliza uma API para diminuir alguns desafios comuns, como: compartilhamento de artefatos, reprodutibilidade do modelo, etc. MLflow é uma ferramenta de ciclo de vida de aprendizado de máquina de código aberto, que facilita o gerenciamento do fluxo de trabalho para treinamento, rastreamento e produção de modelos de machine learning. 

O MLFlow foi organizado para funcionar com as bibliotecas e estruturas de aprendizado de máquina mais recentes e utilizadas no mercado atualmente. Na imagem abaixo pode-se observar o funcionamento arquitetural como um todo do fluxo de MLOps.
  
![image](https://user-images.githubusercontent.com/37118856/210183675-52e00553-6184-4049-9d45-c7f065d73ffb.png)
  
O _flavour_ python_function serve como uma interface de modelo padrão para modelos Python do MLflow. Espera-se que qualquer modelo MLflow Python seja carregável como um modelo `python_function`.

Além disso, o módulo `mlflow.pyfunc` define um formato de sistema de arquivos genérico para modelos Python e fornece utilitários para salvar e carregar desse formato. O formato é independente no sentido de que inclui todas as informações necessárias para que qualquer pessoa possa carregá-lo e usá-lo. As dependências são armazenadas diretamente com o modelo ou referenciadas por meio de um ambiente Conda.

O módulo `mlflow.pyfunc` também define utilitários para criar modelos pyfunc personalizados usando estruturas e lógica de inferência que podem não estar incluídas nativamente no MLflow.
  
  ## 4.4 Infraestrutura Azure
  
Caso seja necessário realizar a criação de um Workspace Databricks em um recurso Azure, vide referência com passo-a-passo: https://learn.microsoft.com/en-us/azure/databricks/scenarios/quickstart-create-databricks-workspace-vnet-injection.
  
Foi definida para implementação em uma worksapce Databricks instanciada na Azure uma estrutura com dois clusters, um cluster de apoio para evitar o custo excessivo em tarefas triviais e um cluster de treinamento de modelos, munido de GPU Tesla. A tabela abaixo contém a descrição detalhada dos clusters e os correspondentes recursos criados.

<img width="231" alt="image" src="https://user-images.githubusercontent.com/37118856/208329993-53da5029-9ce7-4186-8f16-e8c0861244ff.png">

Para armazenamento do token de conexão entre workspaces foi criado o escopo `**data_masters**` juntamente com a secret/key `**data_master_sandbox**`.
  
  ## 4.5 Infraestrutura GCP
  
Caso seja necessário realizar a criação de um Workspace Databricks de um recurso Google, vide referência com passo-a-passo: https://docs.gcp.databricks.com/administration-guide/account-settings-gcp/workspaces.html.
  
Para implementação na workspace Databricks GCP, apenas é necessário um único cluster para teste e deploy do modelo via MLFlow na forma de API REST. Este cluster não necessita, obrigatoriamente, de GPU, o modelo pode ser deployado em um cluster apenas com CPU, e a depender da necessidade por velocidade e processamento da aplicação a ser utilizada, pode ser utilizado o recurso de placas gráficas para acelerar o desempenho.
  
  <img width="241" alt="image" src="https://user-images.githubusercontent.com/37118856/208330032-69a19d4f-b7d2-4d26-b8ec-2840dfa70b65.png">
 
Importante salientar também que foi utilizado neste workspace o recurso de serving via API de modelos, que realiza o deploy utilizando o modelo registrado no MLFlow invocando as APIs padrão, neste caso em específico do TFX para deploy de um modelo Tensorflow.

Para armazenamento do token de conexão entre workspaces foi criado o escopo `**data_masters_gcp**` juntamente com a secret/key `**data_master_deploy**`.


# 5. Dataset e Arquitetura CNN

Foi utilizado o conhecido dataset chamado MNIST - _Modified National Institute of Standards and Technology_, composto por cerca de 70.000 imagens de algarismos numéricos escritos a mão e pode ser utilizado de forma aberta para o desenvolvimento de modelos de reconhecimento de números utilizando técnicas de visão computacional.

![image](https://user-images.githubusercontent.com/37118856/206886708-a049ffd8-9af0-4ac7-a5b4-d5ebc5e3d19b.png)

O objetivo deste case em específico não consiste no treinamento de um modelo com a melhor acurácia possível, falhas são toleráveis se tratando da predição, logo, a arquitetura selecionada para a rede não foi de grande complexidade, uma vez que, apenas era necessário validar o treinamento de uma CNN, com n camadas ocultas. 

Para tanto, foi utilizada uma rede com 3 camadas `Conv2D`, com ativação `relu`, e nas camadas de saída duas camadas densas, uma com ativação `relu` e a camada de classificação `softmax` para 10 classes.

O `learning_rate` setado foi de `0.001`, utilizando um treino com 50 épocas e 100 steps por época. A métrica logada e avaliada foi apenas acurácia, uma vez que o enfoque não era a qualidade do modelo apresentado.


# 6. Treinamento Databricks Azure

Foi construído um cenário de Sandbox dentro de uma workspace Databricks Azure, para a construção de modelos de Deep Learning. A escolha do provedor Azure neste cenário específico foi em decorrência dos recursos disponíveis neste ambiente, e a liberdade para o consumo de créditos, acordado previamente.

O cenário aqui implementado consistiu no consumo do MNIST, mencionado na sessão anterior, e utilizando-se uma Rede Convolucional (CNN) com poucas camadas (diminuindo o tempo e recursos para treinamento), para validar o cenário de implantação com um modulo funcional, a função de treino pode ser vista abaixo.

```python
def train():
  import tensorflow as tf
  import uuid
 
  BUFFER_SIZE = 10000
  BATCH_SIZE = 64
 
  def make_datasets():
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')
 
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
        tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset
 
  def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'],
    )
    return model
  
  train_datasets = make_datasets()
  multi_worker_model = build_and_compile_cnn_model()
 
  # Specify the data auto-shard policy: DATA
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = \
      tf.data.experimental.AutoShardPolicy.DATA
  train_datasets = train_datasets.with_options(options)
  
  multi_worker_model.fit(x=train_datasets, epochs=50, steps_per_epoch=100)
```

Para treinamento utilizando os recursos de GPU e distribuição de tarefas entre os workers existem 4 modos de treinamento conforme detalhados nos tópicos a seguir.


  ## 6.1 Treinamento Single Node
  
O treinamento Single Node consiste na modalidade mais simples de treino, na qual o recurso é completamente alocado no driver, se esse possuir uma GPU a mesma pode ser utilizada para acelerar o treinamento, entretanto a paralelização de tarefas ocorrerá somente dentro das threads da GPU.

A partir de uma função de treinamento pré-construída, detalhada anteriormente, pode-se invocar o treinamento single-node conforme abaixo:

```python
train()
```

  ## 6.2 Treinamento Distribuído
  
O treinamento distríbuido permite a utilização mais de um nó para paralelização da execução das tarefas de treinamento, podendo multiplexar o treinamento de vários modelos simultaneamente, ou até mesmo de um único modelo cross-workers.

  ### 6.2.1 Treinamento Distribuído: Local Mode

O modo de treinamento local permite apenas que o código seja executado dentro do driver-node. E para tanto é necessário utilizar o código abaixo. Entretanto, vale salientar que esse modo não utiliza o poder de paralelismo disponível em um Cluster. Ou seja, é o mesmo que realizar o descrito no método [6.1](#61-treinamento-single-node).

```python
from spark_tensorflow_distributor import MirroredStrategyRunner
 
runner = MirroredStrategyRunner(num_slots=1, local_mode=True, use_gpu=USE_GPU)
runner.run(train)
```
  
  ### 6.2.2 Treinamento Distribuído: Distributed Mode

Neste modo em específico os workers dentro do cluster são utilizados para divisão das tarefas, ao invés de serem executadas no driver. O número de workers pode ser definido assim como se serão utilizadas GPUs ou não para o processamento paralelizado.

```python
NUM_SLOTS = TOTAL_NUM_GPUS if USE_GPU else 4  # For CPU training, choose a reasonable NUM_SLOTS value
runner = MirroredStrategyRunner(num_slots=NUM_SLOTS, use_gpu=USE_GPU)
runner.run(train)
```

  ### 6.2.3 Treinamento Distribuído: Custom Mode
  
Para utilização da estratégia customizada de treinamento, podendo utilizar simultaneamente vários modelos com diferentes parâmetros é necessário definir uma função customizada de treino, adicionando o código abaixo dentro do escopo da função `train()`já citada:

```python
def train_custom_strategy():
  import tensorflow as tf
  import mlflow.tensorflow
  
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
      tf.distribute.experimental.CollectiveCommunication.NCCL)
  
  with strategy.scope():
    import uuid
 
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
```

Após a criação de um escopo personalizado, a função precisa ser invocada de forma similar a já mostrada anteriormente.
```python
runner = MirroredStrategyRunner(num_slots=2, use_custom_strategy=True, use_gpu=USE_GPU)
runner.run(train_custom_strategy)
```
  
# 7. Model Logging e Registry via MLFlow

Nativamente embarcado a solução da Databricks existe um serviço de MLFlow gerenciado, este serviço contempla todas as funções já conhecidas do MLFlow e disponibiliza uma camada gráfica de UI na plataforma. Além disso, esse serviço nativamente é configurado para armazenar as informações pertinentes ao modelo dentro de um diretório no DBFS da workspace.

Podem ser encontradas informações detalhadas de como realizar cada uma das etapas de modelos dentro do ambiente Databricks no link: [MLFlow Log & Registry](https://docs.databricks.com/mlflow/models.html).

É possível utilizar todas as funções disponíveis do MLFlow fazendo o import da biblioteca e utilizando os recursos de log, tracking, models e registry da ferramenta, conforme exemplificados na imagem abaixo.

![image](https://user-images.githubusercontent.com/37118856/210175030-f20b84d9-02c2-4df4-b3cf-004f60ad3a32.png)

A abordagem aqui descrita fez uso do componente de autolog do MLFlow, já que o foco não era detalhar ao máximo os registros dentro do experimento, mas sim, validar todo o pipeline criado. E após isso realizar o registry do modelo, podendo ser diretamente pela UI ou através de comando em código. Detalhes são facilmente observados dentro dos notebooks de treino e deploy, os quais, respectivamente, logam e registram e após carregam e servem.

Para importar a biblioteca o código abaixo pode ser utilizado:
```python
import mlflow
import mlflow.tensorflow
import mlflow.spark
```
Para a utilização do método de autolog, basta invoca-lo para Spark e TensorFlow:
```python
mlflow.autolog()
mlflow.spark.autolog()
mlflow.tensorflow.autolog()
```
Para o registry de modelos é necessário definir o `registry_uri` que é a composição criada pelo scope e key e depois disso efetuar o registry em um repositório remoto, conforme abaixo:
```python
scope = 'data_masters_gcp'
key = 'data_masters_deploy'
registry_uri = 'databricks://' + scope + ':' + key if scope and key else None

mlflow.set_registry_uri(registry_uri)
mlflow.register_model(model_uri=<path_to_uri_model>, name=<name_of_model_to_save>)
```

# 8. Integração via MLFlow API

Uma das vantagens do serviço já ser nativo a plataforma é a facilidade de não necessitar da instanciação de um serviço paralelo, via container, do MLFlow para se utilizar todas as vantagens desta ferramenta. 

Aproveitando desse fato e também da possibilidade de registrar modelos em um repositório remoto (feature disponibilizada nativamente pelo MLFlow), é possível orquestrar um pipeline para se realizar o treinamento e logging de um modelo em um ambiente de experimentação, sandbox, e registrá-lo remotamente em um ambiente produtivo, de deploy.


  ## 8.1 Criação de Scope e Secrets Databricks
  
Para utilização da API do MLFlow para acesso de modelos em um repositório remoto é necessário criar um escopo dentro do que seria a “key vault” do workspace local Databricks e adicionar três secrets referentes ao workspace remoto (host, token e workspace-id), isso pode ser feito de duas maneiras distintas, via Databricks CLI ou via Databricks API, para esta implantação foi utilizada o Databricks CLI. É necessário seguir o procedimento detalhado descrito na documentação que pode ser encontrada no link: [Secret scopes - Azure Databricks | Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes).

Após a correta realização do procedimento é possível acessar um repositório remoto utilizando o Client da API do MLFlow, utilizando os métodos descritos no passo-a-passo. Uma vez conectado ao repositório remoto tem-se o acesso completo as funções do MLFlow.

Na seção [4.4](#44-infraestrutura-azure) e [4.5](#45-infraestrutura-gcp) foram descritas as secrets criadas em ambos os ambientes GCP e Azure para utilização dos recursos cross-workspace. Vale salientar que faz-se necessário a prévia criação do token de acesso nos workspaces que serão utilizados.

  
  ## 8.2 Utilização de Client para Conexão com Repositórios Remotos
  
Um dos recursos nativos mais interessantes disponíveis para o MLFlow é a possibilidade de através de um pipeline de CI/CD realizar a integração entre diferentes workspaces Databricks em diferentes provedores de Cloud. A imagem a seguir retrata o funcionamento da integração (pipeline de CI/CD) utilizado para realizar a integração entre ambientes.
 
<img width="1000" alt="image" src="https://user-images.githubusercontent.com/37118856/208350713-01446417-5076-42de-b8a0-935604d371f1.png">

No cenário desse experimento não foi implementado um workspace de staging, apesar de ser possível, haja vista que o intuito era validar a implantação. Entretanto, um workspace de staging pode ser de grande serventia em um cenário produtivo usual, sendo um espaço transitório onde os times de Cientistas de Dados persistem os registros de seus modelos, e a partir deste workspace a esteira para produção é empurrada.

# 9. Deploy Databricks GCP

O ambiente Databricks criado na Google Cloud Plataform (GCP) tem o objetivo de replicar o que seria um ambiente de "produção", neste caso, um workspace isolado em uma outra instância de Cloud Pública. 

A partir de um modelo já registrado nesse repositório remoto é possível validar este modelo, realizar um "deploy" com predict direto em uma célula de notebook e efetivamente servir como REST API ou como job agendado.

Dentro deste ambiente de produção é possível realizar algumas validações, por exemplo, adicionar uma descrição de alto nível, uma descrição por versão, promover ou regredir de stagging para produção e vice-versa, e por fim realizar um predict de teste e o deploy como API do modelo criado.

Antes de realizar quaisquer alterações sejam de stage ou descrição é necessário realizar o load do modelo, conforme descrito:
```python
model_name = <yours_model_name>
model_version = <specified_version>
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
```

Para adicionar uma descrição de alto nível utiliza-se o código:
```python
client.update_registered_model(
  name=model_name,
  description=<yours_description>
)
```

Para adicionar uma descrição para uma versão do modelo, utiliza-se:
```python
model_description = client.update_model_version(
  name=model_name,
  version=model_version,
  description=<yours_description>
)
```

Já uma mudança de estágio de modelo pode ser realizada através dos comandos:
```python
model_transiction = client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='Staging', # or Production
)
```

Após realizadas todas as mudanças de estágio e descrição previstas é possível fazer um teste de predição do modelo utilizando:
```python
predictions = model.predict(sample)
```

De posse de todas as informações e validações necessárias, é possível realizar o deploy deste modelo, procedendo de duas maneiras distintas: inferência batch ou REST API, sendo a adotada para este experimento a segunda opção de inferência via REST API. 

<img width="1439" alt="image" src="https://user-images.githubusercontent.com/37118856/210193344-feebe65c-89fa-47d6-9120-763c8e52d293.png">

Para tanto, pode e é recomendável a utilização do componente de serving de modelos nativo da plataforma. Que permite a seleção de cluster e parametrização, assim como a geração de logs de API e templates de consumo conforme imagens e documentação disponível em [Serving de Modelos utilizando REST API](https://docs.databricks.com/mlflow/model-serving.html). 

![image](https://docs.databricks.com/_images/enable-serving.gif)

A animação acima retrata o passo-a-passo para realização do serving do modelo como REST API. Este .gif está disponível dentro da documentação oficial da Databricks. É digno de nota que para cada cenário em específico um tipo de cluster é recomendado com o intuito de maximizar a performance e custo da aplicação.


# 10. Requisições de Teste na API Deployada

Para testar o ambiente desenvolvido foi necessário localmente desenvolver um outro notebook responsável por montar a requisição a partir de uma imagem e via POST receber o response do modelo deployado. Seguindo a arquitetura da figura abaixo:

<img width="600" alt="image" align="center" src="https://user-images.githubusercontent.com/37118856/206933989-349fed2f-b7c2-44a2-8ae3-0f394c177ad3.png">

Este notebook é responsável por montar a requisição no formato pré-definido na documentação do TFX, seguindo o padrão:

E, após a chamada, tratar o retorno no formato `np.array()` para encontrar o maior valor que corresponde a predição do modelo, isso é realizado utilizando o método `argmax()` que retorna a posição de 0 a 9 do maior valor encontrado no array, conforme detalhado no código abaixo.
```python
def score_model(image, url=url, headers=headers):
    
    data_json = json.dumps(create_formated_image(image))
    
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
    prediction = response.json()['predictions']
    prediction = np.argmax(prediction)
    
    return prediction
```
A chamada precisa conter o token para conexão com o workspace Databricks em seu header, visando sempre as boas práticas de programação e segurança o token foi criado utilizando uma váriavel de ambiente, para que o mesmo não ficasse exposto em código, a variável utilizada foi:

`DATABRICKS_TOKEN_GCP`.

Já o payload deve seguir o formato:

`'{"instances" : [image_np_array]}'`.

Caso mais de uma imagem seja enviada na chamada é necessário adicionar mais instances de acordo com a quantidade desejada. Detalhes podem ser consultados diretamente no notebook: `test_environment_notebook.ipynb`, no qual está descrita toda implementação e consulta na API REST. O tamanho definido para imagem de entrada é de 28x28 pixels, logo, dentro da função foi implementado o redimensionamento das imagens fornecidas pelo usuário.


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

Site: https://docs.databricks.com/mlflow/model-serving.html

Site: https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes

Site: https://docs.databricks.com/mlflow/models.html

