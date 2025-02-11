The zip file contains a few files within it and their descriptions are:

- ILPD.csv: The original dataset from UCI Machine Learning Repository
- trainingData.csv: The training dataset
- testingData.csv: The testing dataset
- trainingCode.m: The Matlab code used for training the models
- testingCode.m: The Matlab code used for testing the trained models
- nbModel.mat: The trained Na•ve Bayes model which is trained on all the features
- rfModel.mat: The trained Random Forest model which is trained on all the features
- nbModelFS.mat: The trained Na•ve Bayes model which is trained on only the selected features
- rfModelFS.mat: The trained Random Forest model which is trained on only the selected features
- selectedFeaturesIdx.mat: This file contains the indices of the selected features
- graphs: A folder where all the generated graphs are stored
- dataForPythonPreprocessing: A folder which contains two csv files used to make graphs in Python

Instructions to test the trained models:

In order to test the trained models on the testing set, all that needs to be done is running the testingCode.m file. This file contains 4 labelled sections. Run section 1, section 2 and section 4 to test the models trained using only the selected features, and get the corresponding evaluation metrics and figures. Run section 1, section 3 and section 4 to test the models trained using all the features, and get their corresponding evaluation metrics and figures (these are the ones mentioned in the poster).

Software Specification: 

I am using Matlab version R2024b Update 2. This has a function called 'auc' which calculates the area under the curve using the rocmetrics object. I was initially using this function but since the lab machines at university have Matlab version R2024a, I have used an alternate approach instead which works on the earlier versions. 