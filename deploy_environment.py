# Databricks notebook source
# MAGIC %md
# MAGIC # Data Masters Case - Machine Learning Engineer: 
# MAGIC ## Deploying Model in Cross-Workspace Scenario (Cross Cloud)
# MAGIC ### Marcos Vinícius Lisboa Melo - BigData & Analytics - vinicius.lisboa@f1rst.com.br
# MAGIC 
# MAGIC At this notebok, we'll be able to load and deploy the model made at the other workspace. We can use the integrated MLFlow API features to deploy and evaluate the model performance. The TensorFlow model will be deployed by API, and be called by REST POST endpoint. It's important to be remember that TTensorFlow FX it's used by MLFlow Deploy to set the data format in the request. An instance of TFX is created by MLFlow in the deploy pipeline.

# COMMAND ----------

import mlflow
import cv2
import mlflow.pyfunc
import numpy as np
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verifiyng model uploaded in Azure
# MAGIC F1rst of all we can verify if the registred model by Azure Workspace, called `mnist_cnn_data_masters`, is correct seeing model informations, and if necessary, add a description and move forward the Model Status to: Stagging or Production as preferred. Bellow we'll load the model, using name and version.

# COMMAND ----------

model_name = 'mnist_cnn_data_masters'
model_version = 1
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Creating local client:

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC So, we can correct load the model, now we can verify the stage and description:

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_name,
  version=model_version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))
print("The current model description is: '{description}'".format(description=model_version_details.description))

# COMMAND ----------

# MAGIC %md
# MAGIC Adding a high-level description to model:

# COMMAND ----------

client.update_registered_model(
  name=model_name,
  description="Modelo de CNN desenvolvido para o case Data Masters Machine Learning Engineer."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Add a model version description with information about the model architecture:

# COMMAND ----------

model_description = client.update_model_version(
  name=model_name,
  version=model_version,
  description="This model version was built using TensorFlow Keras. It is a convolutional neural network with three hidden layers."
)
print("The current model description is: '{description}'".format(description=model_description.description))

# COMMAND ----------

# MAGIC %md
# MAGIC Change model stage:

# COMMAND ----------

model_transiction = client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='Staging',
)
print("The current model stage is: '{stage}'".format(stage=model_transiction.current_stage))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a sample request to validate predict
# MAGIC After we verified all the model infos, we would like to perform a prediction in the model before deploy in a Cluster REST API. For that is necessary use a sample image test, at this case we need an image named `gray_image_test.jpg`.

# COMMAND ----------

from dbruntime.patches import cv2_imshow

expected_width = 28
expected_length = 28

image = cv2.imread('/dbfs/FileStore/shared_uploads/marcos.melo@cear.ufpb.br/gray_image_test.jpg')

width = expected_width/image.shape[0]
length = expected_length/image.shape[1]

image = cv2.resize(image0, (0, 0), fx = width, fy = length)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

sample = [gray]
sample = np.array(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC At least we invoke the predict model passing the array that contains sample test image.

# COMMAND ----------

predictions = model.predict(sample)
classes = np.argmax(predictions, axis = 1)
print('Predict number is:' + str(classes))

# COMMAND ----------

# MAGIC %md 
# MAGIC **Now we can shift to "Model" window on Databricks and serve this model by REST API. If you preffer the deploy can be realized creating an API using a framework like FastAPI or using the MLServer by command line.**
