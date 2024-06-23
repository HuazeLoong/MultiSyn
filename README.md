# Mytest
This project contains six files for training and evaluating drug combination prediction models.   
The project is mainly divided into sections such as data processing, model definition, training, and evaluation.  

train.py: Main program for training the prediction model.    
data_drug.py: process drug data,construct molecular graphs,and create datasets.  
dataset_drug.py: process drug-target data and convert it into PyTorch Geometric format.   
model_drug.py: Defines the model for drug molecular feature.     
model.py: predict drug combination.    
utils.py: loss functions, evaluation metrics, etc.    

Step:
Ensure that the dataset is prepared and preprocessed as needed.  
 
Pytorch XXX  
Python XXX  

When running for the first time, please run the data_drug.py file to create data.  
Use train.py to train the model.
