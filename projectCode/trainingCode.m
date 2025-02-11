clearvars, close all                          % Clears all the variables and closes all the figures every time the code is run

% Loading data

data = readtable('ILPD.csv');                 % Loads the dataset into a table

head(data)                                    % Displays the first few rows with the column names
columnNames = data.Properties.VariableNames;  % Stores the column names

% Replacing 2 with 0 (no liver disease)

data.Selector(data.Selector == 2) = 0;        % Replaces all the Selector values of 2 to 0 for denoting no liver disease
numRowsWith0 = sum(data.Selector == 0);       % Gets the count of all the patients who do not have liver disease
numRowsWith1 = sum(data.Selector == 1);       % Gets the count of all the patients who have liver disease

disp(['Number of rows with 0: ' num2str(numRowsWith0)]);  % Displays the number of patients who do not have liver disease (0)
disp(['Number of rows with 1: ' num2str(numRowsWith1)]);  % Displays the number of patients who have liver disease (0)

% Converting Gender to numerical binary

data.Gender = double(strcmp(data.Gender, 'Male'));     % Converts Gender column to numerical denoting 1 for 'Male' and 0 for 'Female'
head(data)

% Identify missing values

missingValues = ismissing(data);                 % Checks for missing values in the the data and returns a logical array indicating 
missingSummary = sum(missingValues);             % true for missing values and false otherwise. Counts the number of missing values in
disp(missingSummary)                             % each column and displays them

% Fill missing values with mean

missingValues = ismissing(data(:, 10));          % Only the 10th column has missing values so it gets the indices of the rows with missing values
columnMean = mean(data{~missingValues, 10});     % Calculates the mean of the 10th column ignoring the missing values
data{missingValues, 10} = columnMean;            % Replaces the missing values with the column mean

numRowsWith0 = sum(data.Selector == 0); 
numRowsWith1 = sum(data.Selector == 1);

disp(['Number of rows with 0: ' num2str(numRowsWith0)]);    % Displays the number of patients who do not have liver disease (0)
disp(['Number of rows with 1: ' num2str(numRowsWith1)]);    % Displays the number of patients who have liver disease (0)

% Identify and remove duplicate rows
[~, uniqueIdx, duplicateIdx] = unique(data, 'rows', 'stable');      % This section checks for duplicates rows in the dataset. If it has duplicate
duplicateRows = setdiff(1:size(data, 1), uniqueIdx);                % rows then the number of duplicate rows are displayed along with the actual
                                                                    % duplicate rows. Then the duplicate rows are removed and only the unique rows                                                              
numDuplicates = numel(duplicateRows);                               % are retained
disp(['Number of duplicate rows: ' num2str(numDuplicates)]);

if numDuplicates > 0
    disp('Duplicate rows:');
    disp(data(duplicateRows, :));
else
    disp('No duplicate rows found.');
end

data = data(uniqueIdx, :);

numRowsWith0 = sum(data.Selector == 0);
numRowsWith1 = sum(data.Selector == 1);
disp(['Number of rows with 0: ' num2str(numRowsWith0)]);       % Displays the number of patients who do not have liver disease (0)
disp(['Number of rows with 1: ' num2str(numRowsWith1)]);       % Displays the number of patients who have liver disease (0)

%%
writetable(data,'dataForPythonPreprocessing/classImbalance.csv');  % Exports data as a csv file to make a pie chart using Python (show in the poster)

%%
% Splitting into training and testing data

rng(10);     % for reproducibility 

trainRatio = 0.7;
cv = cvpartition(size(data, 1), 'HoldOut', 1 - trainRatio);   % Splits the data into 70% for training and 30% for testing

trainingData = data(training(cv), :);           % Stores all the rows assigned to the training set
testingData = data(test(cv), :);                % Stores all the rows assigned to the test set

writetable(trainingData, 'trainingData.csv');       % Saves the training set as a .csv file to be used later
writetable(testingData, 'testingData.csv');         % Saves the testing set as a .csv file to be used later
%%
% Handling outliers (removing them)

% In this section, the code iterates over all the columns in the table except 
% for the Age and Gender columns. For each column it iterates over, it calculates 
% the mean and standard deviation of that column and using these values it 
% calculates the z score of each entry in that column. Values that have a
% z score greater than 3 are treated as outliers and the entire row is
% removed. This makes sure that outliers of each features are removed from
% the table.

rng(10);       % For reproducibility                            
dataCleanedNoOutliers = trainingData;                
limit = size(dataCleanedNoOutliers, 2)-1;            
zThreshold = 3;                                      
                                                                                                      
for i = 3:limit                        
    feature = dataCleanedNoOutliers{:, i}; 
    featureMean = mean(feature);          
    featureStd = std(feature);           
    zScores = (feature - featureMean) / featureStd;
    dataCleanedNoOutliers = dataCleanedNoOutliers(abs(zScores) <= zThreshold, :);
end
                                           
trainingData = dataCleanedNoOutliers;
numRowsWith0 = sum(trainingData.Selector == 0);
numRowsWith1 = sum(trainingData.Selector == 1);
disp(['Number of rows with 0: ' num2str(numRowsWith0)]);   % Displays the number of patients who do not have liver disease (0) after outlier removal
disp(['Number of rows with 1: ' num2str(numRowsWith1)]);   % Displays the number of patients who have liver disease (0) after outlier removal
%%
writetable(trainingData,'dataForPythonPreprocessing/heatmap.csv');  % Exports data as a csv file to make a correlation matrix and bar plots using Python (show in the poster)

%%
% Correlation Matrix

columnNames = trainingData.Properties.VariableNames;      % In this section a correlation matrix is made and displayed as a heatmap.
numericData = trainingData{:, :};                         % It shows the pairwise correlations between all the variables in the table
correlationMatrix = corr(numericData);
figure;
h = heatmap(columnNames, columnNames, correlationMatrix);
h.Title = 'Correlation Matrix';

%%

% Feature Selection
% Run only for training with feature selection

% In this section Feature selection is done on the training data. The code 
% iterates over all the features and for each feature it calculates the
% correlation with the class column. Features that have correlation score
% with the class column greater than the threshold (0.15) are kept whereas 
% features that have a correlation lesser than or equal to 0.15 are
% removed. For all the retained features, their pairwise correlation score
% is calculated. The pairs that have a correlation score greater than 0.8
% are further analysed and out of the pair, the feature that has a higher
% correlation score with the class column is retained whereas the other one
% is discarded. The features that are left at the end of this process are
% kept for training and their indices are saved as a .mat file to be used
% later on

rng(10);        % For reproducibility                                           
                                                         
features = trainingData(:, 1:end-1);               
target = trainingData{:, end};                     

numFeatures = width(features);                    
correlationScores = zeros(1, numFeatures);         

for i = 1:numFeatures
    featureData = table2array(features(:, i));
    correlationScores(i) = corr(featureData, target, 'Type', 'Pearson');
end

classCorrThreshold = 0.15;
initalSelectedFeaturesIdx = find(abs(correlationScores) > classCorrThreshold);

pairCorrThreshold = 0.8;
initialSelectedFeatures = features(:, initalSelectedFeaturesIdx);
pairwiseCorrMatrix = abs(corr(table2array(initialSelectedFeatures)));
featuresToKeep = true(length(initalSelectedFeaturesIdx), 1);

for i = 1:length(initalSelectedFeaturesIdx)
    for j = i+1:length(initalSelectedFeaturesIdx)
        if pairwiseCorrMatrix(i, j) > pairCorrThreshold && featuresToKeep(j)
            % Keep the feature with higher correlation to the target
            if abs(correlationScores(initalSelectedFeaturesIdx(i))) >= abs(correlationScores(initalSelectedFeaturesIdx(j)))
                featuresToKeep(j) = false;
            else
                featuresToKeep(i) = false;
            end
        end
    end
end

selectedFeaturesIdx = initalSelectedFeaturesIdx(featuresToKeep);
selectedFeatures = features(:, selectedFeaturesIdx);
selectedFeatureNames = selectedFeatures.Properties.VariableNames;

save('selectedFeaturesIdx.mat', 'selectedFeaturesIdx');

disp('Selected Features based on PCC-FS:');
disp(selectedFeatureNames');
disp('Correlation Scores of Selected Features:');
disp(correlationScores(selectedFeaturesIdx));

%%
% Defining Input (X) and Class (Y) variables
rng(10);

XTrain = trainingData{:, 1:end-1};         % Extracts all the predictor variables from the training data and store them in an array
YTrain = trainingData{:, 'Selector'};      % Extracts the class column from the training data and store it in an array

numRowsWith0 = sum(YTrain == 0);
numRowsWith1 = sum(YTrain == 1);

disp(['Number of training rows with 0: ' num2str(numRowsWith0)]);
disp(['Number of training rows with 1: ' num2str(numRowsWith1)]);
%%
% Run only for training with selected features

XTrain = XTrain(:, selectedFeaturesIdx);   % Extracts only the selected features from the training data
head(XTrain)
%%

% Bayesian Optimisation for Naive Bayes

% Implements automatic hyperparameter fine-tuning for Naive Bayes using
% Bayesian Optimisation. It does this for Distribution, Width and whether
% to Standardize or not. While doing so, it implements 10 fold cross
% validation and repartitions the data into folds everytime. It displays
% the information in the command window for every function evaluation

rng(10);
hyperparams = {'DistributionNames', 'Width', 'Standardize'};
nbModel = fitcnb(XTrain, YTrain, ...
    'OptimizeHyperparameters', hyperparams, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'KFold', 10, ...
        'Repartition', true, ...
        'Verbose', 1));

%%

% Grid search to fine tune hyperparameters using K fold cross validation (Naive Bayes)

% In this section a grid search is implemented to fine tune the
% hyperparameters for Naive Bayes. First the hyperparamter values are
% initialised and then they are iterated over in the grid search. For each
% unique set of hyperparameters, the model is trained and validated using
% 10 fold cross validation. The predictions made by each fold on their
% validation set are combined to get the predictions of the entire training
% set. These are compared with the true labels to calculate evaluation
% metrics such as validation accuracy, precision, recall, F1 score, ROC and
% AUC. Then the cross validation error of the model is calculated and if it
% less than the lowest one yet, the value is updated. The evaluation
% metrics and validation error for each set of hyperparameters is stored and
% the set with the lowest validation error is selected as the best set of
% hyperparameters and displayed. Then a plot of cross validation error
% against kernel width is plotted (displayed in the poster)

rng(10);

priorRange = {'empirical', 'uniform'};
% priorRange = {'uniform'};
distributionRange = {'normal', 'kernel'};
% distributionRange = {'normal'};
% widthRange = [0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 5, 10];
widthRange = 0.1:0.1:2;
% widthRange = 0.05:0.01:1;

results = [];
bestROC = [];
lowestError = inf;

for prior = priorRange
    for dist = distributionRange
        for width = widthRange
            
            if strcmp(dist, 'kernel')
                nbModel = fitcnb(XTrain, YTrain, ... 
                    'DistributionNames', dist{1}, ... 
                    'Prior', prior{1}, ... 
                    'Width', width, ...,
                    'CrossVal', 'on', ... 
                    'KFold', 10);
            else
                nbModel = fitcnb(XTrain, YTrain, ... 
                    'DistributionNames', dist{1}, ... 
                    'Prior', prior{1}, ...
                    'CrossVal', 'on', ... 
                    'KFold', 10);
            end
    
            [predictions, scores] = kfoldPredict(nbModel);
            trueLabels = YTrain;
            TP = sum((predictions == 1) & (trueLabels == 1));
            FP = sum((predictions == 1) & (trueLabels == 0));
            FN = sum((predictions == 0) & (trueLabels == 1));
            TN = sum((predictions == 0) & (trueLabels == 0));

            meanAccuracy = mean(predictions == YTrain);
            meanPrecision = TP / (TP + FP);
            meanRecall = TP / (TP + FN);
            meanF1 = 2 * (meanPrecision * meanRecall) / (meanPrecision + meanRecall);
            allScores = scores(:, 2);
            allLabels = trueLabels;
            rocObj = rocmetrics(allLabels, allScores, 1);
            fpr = rocObj.Metrics{:, 'FalsePositiveRate'};
            tpr = rocObj.Metrics{:, 'TruePositiveRate'};
            a = trapz(fpr, tpr);
             
            meanError = kfoldLoss(nbModel);

            if meanError < lowestError
                lowestError = meanError;
                bestROC = rocObj;
            end

            if strcmp(dist{1}, 'normal')
                widthToStore = '-';
            else
                widthToStore = num2str(width);
            end

            results = [results; {prior{1}, dist{1}, widthToStore, meanError, meanAccuracy*100, meanPrecision*100, meanRecall*100, meanF1, a}]; %#ok<AGROW>
            if strcmp(dist{1}, 'normal')
                break;
            end
      
        end
    end
end


% Displaying the best set of hyperparameters
resultsTable = cell2table(results, 'VariableNames', {'Prior', 'Distribution', 'Width', 'MeanError', 'MeanAccuracy', 'MeanPrecision', 'MeanRecall', 'MeanF1', 'AUC'});

[~, bestIdx] = min(resultsTable{:, 'MeanError'}); 

bestPrior = resultsTable.Prior{bestIdx};
bestDistribution = resultsTable.Distribution{bestIdx};
bestWidth = resultsTable.Width{bestIdx};
lowestMeanError = resultsTable.MeanError(bestIdx);

fprintf('Lowest Mean Validation Loss: %.2f%%\n', lowestMeanError);
fprintf('Best Configuration:\nPrior: %s, Distribution: %s, Width: %s\n', bestPrior, bestDistribution, bestWidth);

figure;
plot(bestROC);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Best Model (Naive Bayes)');
grid on;

% Validation loss vs. Width for kernel distribution plot
kernelErrors = resultsTable(strcmp(resultsTable.Distribution, 'kernel') & strcmp(resultsTable.Prior, 'empirical'), :);
kernelErrors.Width = cellfun(@str2double, kernelErrors.Width);

figure;
plot(kernelErrors.Width, kernelErrors.MeanError, '-o', 'LineWidth', 2);
xlabel('Kernel Width', 'FontSize', 14);
ylabel('Cross Validation Loss', 'FontSize', 14);
title('Cross Validation Loss vs. Kernel Width', 'FontSize', 16);
grid on;
set(gca, 'FontSize', 14);
saveas(gcf, 'graphs/width.png', 'png');
%%

% Bayesian Optimisation for Random Forest


% Implements automatic hyperparameter fine-tuning for Random Forest using
% Bayesian Optimisation. It does this for NumLearningCycles, MaxNumSplits
% and NumVariablesToSample. While doing so, it implements 10 fold cross
% validation and repartitions the data into folds everytime. It displays
% the information in the command window for every function evaluation

rng(10);

hyperparams = {'NumLearningCycles', 'MaxNumSplits', 'NumVariablesToSample'};

rfModel = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ... 
    'OptimizeHyperparameters', hyperparams, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Repartition', true, ...
        'KFold', 10, ...
        'Verbose', 1));

%%
% Grid search to fine tune hyperparameters using K fold cross validation (Random Forest)

% In this section a grid search is implemented to fine tune the
% hyperparameters for Random Forest. First the hyperparameter values are
% initialised and then they are iterated over in the grid search. For each
% unique set of hyperparameters, the model is trained and validated using
% 10 fold cross validation. The predictions made by each fold on their
% validation set are combined to get the predictions of the entire training
% set. These are compared with the true labels to calculate evaluation
% metrics such as validation accuracy, precision, recall, F1 score, ROC and
% AUC. Then the cross validation error of the model is calculated and if it
% less than the lowest one yet, the value is updated. The evaluation
% metrics and validation error for each set of hyperparameters is stored and
% the set with the lowest validation error is selected as the best set of
% hyperparameters and displayed. Then a plot of cross validation error
% against number of learning cycles is plotted (displayed in the poster)

rng(10);

numLearningCyclesRange = [40, 50, 57, 60, 70, 80, 90, 100];
% numLearningCyclesRange = [50, 60, 100, 250, 400];
maxNumSplitsRange = [85, 105, 125, 140, 160];
% maxNumSplitsRange = [1, 50, 100, 167, 200];
numVariablesToSampleRange = [1, 2, 3, 4];
% numVariablesToSampleRange = [2, 5, 10];


results = [];
lowestError = inf;
bestROC = [];


for numLearningCycles = numLearningCyclesRange
    for maxNumSplits = maxNumSplitsRange
        for numVariablesToSample = numVariablesToSampleRange
            
            t = templateTree('MaxNumSplits', maxNumSplits, ...               % There are the learners (decision trees) that will be used in
                     'NumVariablesToSample', numVariablesToSample, ...       % the ensemble
                     'Reproducible', true);

            rfModel = fitcensemble(XTrain, YTrain, ...
                  'Method', 'Bag', ...
                  'NumLearningCycles', numLearningCycles, ...
                  'Learners', t, ...
                  'CrossVal', 'on', ... 
                  'KFold', 10);

            [predictions, scores] = kfoldPredict(rfModel);
            trueLabels = YTrain;
            TP = sum((predictions == 1) & (trueLabels == 1));
            FP = sum((predictions == 1) & (trueLabels == 0));
            FN = sum((predictions == 0) & (trueLabels == 1));
            TN = sum((predictions == 0) & (trueLabels == 0));

            meanAccuracy = mean(predictions == YTrain);
            meanPrecision = TP / (TP + FP);
            meanRecall = TP / (TP + FN);
            meanF1 = 2 * (meanPrecision * meanRecall) / (meanPrecision + meanRecall);
            allScores = scores(:, 2);
            allLabels = trueLabels;
            rocObj = rocmetrics(allLabels, allScores, 1);
            fpr = rocObj.Metrics{:, 'FalsePositiveRate'};
            tpr = rocObj.Metrics{:, 'TruePositiveRate'};
            a = trapz(fpr, tpr);
            
            meanError = kfoldLoss(rfModel);

            if meanError < lowestError
                lowestError = meanError;
                bestROC = rocObj;
            end
            results = [results; {numLearningCycles, maxNumSplits, numVariablesToSample, meanError, meanAccuracy*100, meanPrecision*100, meanRecall*100, meanF1, a}]; %#ok<AGROW>
        end
    end
end


% Displaying the best set of hyperparameters
resultsTable = cell2table(results, 'VariableNames', {'numLearningCycles', 'maxNumSplits', 'numVariablesToSample', 'MeanError', 'MeanAccuracy', 'MeanPrecision', 'MeanRecall', 'MeanF1', 'AUC'});

[~, bestIdx] = min(resultsTable{:, 'MeanError'}); 

bestNumLearningCycles = resultsTable.numLearningCycles(bestIdx);
bestMaxNumSplits = resultsTable.maxNumSplits(bestIdx);
bestNumVariablesToSample = resultsTable.numVariablesToSample(bestIdx);
lowestMeanError = resultsTable.MeanError(bestIdx);


fprintf('Lowest Mean Validation Loss: %.2f%%\n', lowestMeanError);
fprintf('Best Configuration:\nNumber of Learning Cycles: %d, Max Number of Splits: %d, Number of Variables To Sample: %d\n', bestNumLearningCycles, bestMaxNumSplits, bestNumVariablesToSample);

figure;
plot(bestROC);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Best Model');
grid on;

% Plot cross validation error against numLearningCycles
filteredErrors = resultsTable(resultsTable.numVariablesToSample == bestNumVariablesToSample & ...
                            resultsTable.maxNumSplits == bestMaxNumSplits, :);

figure;
plot(filteredErrors.numLearningCycles, filteredErrors.MeanError, '-o', 'LineWidth', 2);
xlabel('Number of Learning Cycles (Trees)');
ylabel('Cross Validation Loss');
title('Cross Validation Loss vs Number of Trees');
grid on;
set(gca, 'FontSize', 14);
saveas(gcf, 'graphs/numTrees.png', 'png');

%%
% Training using best set of hyperparameters for Naive Bayes

% In this section the Naive Bayes model is retrained using the best set of
% hyperparameters. This trained model is saved as a .mat file to be used later
% one for testing. The training time is calculated and displayed

rng(10);
tic;

if strcmp(bestDistribution, 'kernel')
    nbModel = fitcnb(XTrain, YTrain, ... 
                    'DistributionNames', bestDistribution, ... 
                    'Prior', bestPrior, ... 
                    'Width', str2double(bestWidth));
else
    nbModel = fitcnb(XTrain, YTrain, ... 
                     'DistributionNames', bestDistribution, ... 
                     'Prior', bestPrior);
end

trainingTime = toc;
disp(['Training Time: ', num2str(trainingTime), ' seconds']);
save('nbModel.mat', 'nbModel');
%%
% Training using best set of hyperparameters for Random Forest

% In this section the Random Forest model is retrained using the best set of
% hyperparameters. This trained model is saved as a .mat file to be used later
% one for testing. The training time is calculated and displayed

rng(10);

tic;

t = templateTree('MaxNumSplits', bestMaxNumSplits, ...
                 'NumVariablesToSample', bestNumVariablesToSample, ...,
                 'Reproducible', true);

rfModel = fitcensemble(XTrain, YTrain, ...
                  'Method', 'Bag', ...
                  'NumLearningCycles', bestNumLearningCycles, ...
                  'Learners', t);

trainingTime = toc;
disp(['Training Time: ', num2str(trainingTime), ' seconds']);
save('rfModel.mat', 'rfModel');