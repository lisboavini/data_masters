# Databricks notebook source
# MAGIC %md
# MAGIC # Data Masters Case - Machine Learning Engineer: 
# MAGIC ## Distributing GPU Training and Deploy (Cross Workspace) using TensorFlow 2 + Spark + MLFlow
# MAGIC ### Marcos Vinícius Lisboa Melo - BigData & Analytics - vinicius.lisboa@f1rst.com.br
# MAGIC 
# MAGIC The main approach here is use Spark combined with TensorFlow 2 to habilitate fastest train of Deep Learning Models, using multiple GPU resources. We'll compare the performance using a single-node mode and using a Mirrored Strategy Runner with three modes: local, distributed and custom. After trained we'll be able to tracking, registry and deploy this model using MLFlow. After that, we'll able to evaluate the best distribuction-train approach for this application. The dataset choiced for this application is MNIST, this dataset is very popular on Keaggle and are a default sample Dataset on Keras/Tensorflow and it contains rotulated images of numbers, and can be easily used to train a simple reference classification model using a Convolutional Neural Network. 
# MAGIC 
# MAGIC References to this approachs can be found on: \
# MAGIC https://www.databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html \
# MAGIC https://docs.databricks.com/machine-learning/train-model/dl-best-practices.html.

# COMMAND ----------

import tensorflow as tf
 
NUM_WORKERS = 2
# Assume the driver node and worker nodes have the same instance type.
TOTAL_NUM_GPUS = len(tf.config.list_logical_devices('GPU')) * NUM_WORKERS
USE_GPU = TOTAL_NUM_GPUS > 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-Node Train
# MAGIC Using just one node to train a classification model based on MNIST Dataset. More details about MNIST dataset is available on the documentation.

# COMMAND ----------

# single-node train
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

# COMMAND ----------

# MAGIC %md
# MAGIC So, after we define our train function, it will be called to generate and log (using MLFlow auto log) a model, and this model will be added a experiment artifact.

# COMMAND ----------

train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Training with `MirroredStrategyRunner`: Local-Mode, Distributed-Mode and Custom-Mode
# MAGIC The Distribituted Training approach has three primary modes to be used. At next cells each mode will be discussed and executed, showing their specifications.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local-Mode
# MAGIC Local mode only's allow the `train()` function to run on the driver node with all GPUs. 

# COMMAND ----------

from spark_tensorflow_distributor import MirroredStrategyRunner
 
runner = MirroredStrategyRunner(num_slots=1, local_mode=True, use_gpu=USE_GPU)
runner.run(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed-Mode
# MAGIC At this mode tasks will be runned on ther workers node, and the driver node can run other workloads, just's necessary call the `train()` function once more.

# COMMAND ----------

NUM_SLOTS = TOTAL_NUM_GPUS if USE_GPU else 4  # For CPU training, choose a reasonable NUM_SLOTS value
runner = MirroredStrategyRunner(num_slots=NUM_SLOTS, use_gpu=USE_GPU)
runner.run(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom-Mode
# MAGIC In this mode it's necessary define a custom strategy to run the distributed expirement, in the library documentation we'll able to found further details about the available strategys. So, bellow the `train_custom_strategy()` is defined using the `tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)` strategy.

# COMMAND ----------

def train_custom_strategy():
  import tensorflow as tf
  import mlflow.tensorflow
  
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
      tf.distribute.experimental.CollectiveCommunication.NCCL)
  
  with strategy.scope():
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
    
    mlflow.tensorflow.autolog()
    multi_worker_model.fit(x=train_datasets, epochs=50, steps_per_epoch=100)

# COMMAND ----------

# MAGIC %md
# MAGIC After the custom train function is definied, is possible to runner using the pre-definied strategy.

# COMMAND ----------

# Use the local mode to verify `CollectiveCommunication.NCCL` is printed in the logs
runner = MirroredStrategyRunner(num_slots=2, use_custom_strategy=True, use_gpu=USE_GPU)
runner.run(train_custom_strategy)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting MLFlow Registry in Local Workspace and at the Remote GCP Repository
# MAGIC F1rst (understanding the joke?! hehe) it's necessary to set MLFlow to autolog the experiments. After that we'll use the API to registrate a model in a remote workspace, this workspace is acessible using the token, associated to this workspace by secure Scope, created with Databricks CLI.

# COMMAND ----------

import mlflow.tensorflow
import mlflow.spark
mlflow.autolog()
mlflow.spark.autolog()
mlflow.tensorflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Registrating the trained model in a remote workspace using the setted scope and URI.

# COMMAND ----------

scope = 'data_masters_gcp'
key = 'data_masters_deploy'
registry_uri = 'databricks://' + scope + ':' + key if scope and key else None

# COMMAND ----------

# MAGIC %md
# MAGIC Setting Databricks CLI on environment, necessary to register models in remote artifact repository.

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put('file:///root/.databrickscfg','[DEFAULT]\nhost=https://adb-8555374716844985.5.azuredatabricks.net\ntoken = '+token,overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the local path of trained model, the `register_model` function will save this model in remote uri defined before.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
mlflow.set_registry_uri(registry_uri)
mlflow.register_model(model_uri='dbfs:/databricks/mlflow-tracking/182935783505295/7207b3b534794d5db7ddc9c741683c0f/artifacts/model', name='mnist_cnn_data_masters')
