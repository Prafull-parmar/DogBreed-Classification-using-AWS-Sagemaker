# DogBreed-Classification-using-AWS-Sagemaker

This is an Image classification project that uses and fine-tunes a pretrained ResNet50 model with AWS Sagemaker to classify Dog breeds from dog images.

# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. <br>
Open the jupter notebook "train_and_deployipynb" and start by installing all the dependencies. <br>
For ease of use you may want to use "Python 3 ( PyTorch 1.6 Python 3.6 CPU Optimized)" Kernel so that you do not need to install most of the pytorch libraries <br>

## Dataset
We will be using the Dog breed dataset that is provided by Udacity.<br>
The dataset contains images of dogs belonging to a total of 133 different breeds from around the world. <br>
We will be using these dog images to train our image classification model to classify between the  different dog breeds.<br>
We will be uploading our dataset to S3 bucket so we could use it for training our model
![S3 Upload Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/S3_upload_snapshot.PNG)

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Overview of Project Steps
The jupyter notebook "train_and_deploy.ipynb" walks through implementation of  Image Classification Machine Learning Model to classify between 133 kinds of dog breeds using dog breed dataset provided by Udacity (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

* We will be using a pretrained Resnet50  model from pytorch vision library (https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html)
* We will be adding in two Fully connected Neural network Layers on top of the above Resnet50 model.
* Note: We will be using concepts of Transfer learning and so we will be freezing all the exisiting Convolutional layers in the pretrained resnet50 model and only changing gradients for the tow fully connected layers that we have added.
* Then we will perform Hyperparameter tuning, to help figure out the best hyperparameters to be used for our model.
* Next we will be using the best hyperparameters and fine-tuning our Resent50 model.
* We will also be adding in configuration for Profiling and Debugging our training mode by adding in relevant hooks in the Training and Testing( Evaluation) phases.
* Next we will be deploying our model. While deploying we will create our custom inference script. The custom inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.
* Finally we will be testing out our model with some test images of dogs, to verfiy if the model is working as per our expectations.


## Files used throughout the project

* **hpo.py** - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
* **train_deploy.py** - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning
* **endpoint_inference.py** - This script contains code that is used by the deployed endpoint to perform some preprocessing( transformations) , serialization- deserialization and predictions/inferences  and post-processing using the saved model from the training job.
* **train_and_deploy.ipynb** -- This jupyter notebook contains all the code and steps that we performed in this project and their outputs.

## Hyperparameter Tuning

* The ResNet50 model with a two Fully connected Linear NN layer's is used for this image classification problem. ResNet-50 is 50 layers deep and is trained on a million images of 1000 categories from the ImageNet database. Furthermore the model has a lot of trainable parameters, which indicates a deep architecture that makes it better for image recognition
* The optimizer that we will be using for this model is AdamW ( For more info refer : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )
* Hence, the hyperparameters selected for tuning were:
  * **Learning rate** - default(x)  is 0.001 , so we have selected 0.01x to 100x range for the learing rate
  * **eps** - defaut is 1e-08 , which is acceptable in most cases so we have selected a range of 1e-09 to 1e-08
  * **Weight decay**  - default(x)  is 0.01 , so we have selected 0.1x to 10x range for the weight decay
  * **Batch size** -- selected only two values [ 64, 128 ]   
* Best Hyperparamters post Hyperparameter fine tuning are : <br>
 { 'batch_size': 128, 'eps': '1.5009475698763981e-09', 'lr': '0.0029088382171354715', 'weight_decay': '0.08373215706456894' }
### Hyperparameter Tuning Sagemaker snapshot
![HPO Tuning Job Sagemaker](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/hpo_tuning_sagemaker_snapshot.PNG)
### HyperParameter tuning Job
![HPO Tuning Job Status](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/hpo_job_success_snapshot.PNG)
### Multiple training jobs triggered by the HyperParameter Tuning Job
![HyperParameter Training Job Execution Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/Hyperparameter_tuning_job_executions.PNG)
### All hyperParameter training jobs Summary
![HyperParameter Training Job Summary Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/hpo_training_job_summary_snapshot.PNG)
### Best hyperparameter Training Job
![Best Hyperparameters Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/best_hyperparameters_snapshot.PNG)
### Best hyperparameter Training Job Logs
![Best Hyperparameters Training Job Log Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/best_hpo_training_job_logs.PNG)

## Debugging and Profiling

We had set the Debugger hook to record and keep trach of the Loss Criterion metrics of the process in training and validation/testing phases. The Plot of the Cross entropy loss is shown below:
![Cross Entropy Loss Tensor Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/debugging_tensor_plot.PNG)

## Endpoint Metrics
![Endpoint Metrics Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/Endpoint_cpu_metrics.PNG)
### Results
Results look pretty good, as we had utilized the GPU while hyperparameter tuning and training of the fine-tuned ResNet50 model. We used the ml.g4dn.xlarge instance type for the runing the traiing purposes.
However while deploying the model to an endpoint we used the "ml.t2.medium" instance type to save cost and resources.




## Model Deployment
* Model was deployed to a "ml.t2.medium" instance type and we used the "endpoint_inference.py" script to setup and deploy our working endpoint.
* For testing purposes , we will be using some test images that we have stored in the "testImages" folder. 
* We will be reading in some test images from the folder and try to send those images as input and invoke our deployed endpoint
* We will be doing this via two approaches
  * Firstly using the Prdictor class object
  * Secondly using the boto3 client
### Deployed Active Endpoint Snapshot
![Deployed Endpoint Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/Deployed_Endpoint.PNG)
### Deployed Endpoint Logs Snapshot, showing that the request was recieved and processed successfullly by the endpoint
![Deployed Endpoint Logs Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/Deployed_endpoint_query_logs.PNG)
### Test cell used for testing the Endpoint Snapshot
![Testing cell for Endpoint Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/test_image_sample_cell.PNG)
### Sample output returned from endpoint Snapshot
![Test Image sample output Snapshot](https://github.com/Prafull-parmar/DogBreed-Classification-using-AWS-Sagemaker/blob/dogbreed_proj/snapshots/sample_endpoint_test_output.PNG)

