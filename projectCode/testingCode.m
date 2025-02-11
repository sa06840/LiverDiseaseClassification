%% Section 1
clearvars, close all

% Loading testing data and trained models

load('nbModel.mat', 'nbModel');                % Loads the trained Naive Bayes model
load('rfModel.mat', 'rfModel');                % Loads the trained Random Forest model
load('nbModelFS.mat', 'nbModelFS');            % Loads the Naive Bayes model that has been trained on only selected features
load('rfModelFS.mat', 'rfModelFS');            % Loads the Random Forest model that has been trained on only selected features
testingData = readtable('testingData.csv');    % Loads and saves the testing data
disp('Test data loaded successfully:');

%% Section 2
% Run to get predictions using the models that have been trained on only selected features

% Defining Input (X) and Class (Y) variables

XTest = testingData{:, 1:end-1};             % Extracts and stores all predictor features
YTest = testingData{:, 'Selector'};          %#ok<NASGU> % Extracts and stores the class column

load('selectedFeaturesIdx.mat', 'selectedFeaturesIdx');    % Loads the indices of the selected features from feature selection
XTest = XTest(:, selectedFeaturesIdx);                     % Extracts and stores only the selected features

% Getting predictions for Naive Bayes
[predictionsNB, scoresNB] = predict(nbModelFS, XTest);         %#ok<ASGLU> % Stores the predictions and scores made by the Naive Bayes model
                                                                 % on the test set
                                                             
% Getting predictions for Random Forest
[predictionsRF, scoresRF] = predict(rfModelFS, XTest);         %#ok<ASGLU> % Stores the predictions and scores made by the Random Forest model
                                                                  % on the test set

%% Section 3
% Run to get predictions using the models that have been trained on all the features

% Defining Input (X) and Class (Y) variables

XTest = testingData{:, 1:end-1};             % Extracts and stores the all predictor features
YTest = testingData{:, 'Selector'};          % Extracts and stores the class column

% Getting predictions for Naive Bayes
[predictionsNB, scoresNB] = predict(nbModel, XTest);         % Stores the predictions and scores made by the Naive Bayes model
                                                             % on the test set
                                                             
% Getting predictions for Random Forest
[predictionsRF, scoresRF] = predict(rfModel, XTest);         % Stores the predictions and scores made by the Random Forest model
                                                             % on the test set

%% Section 4
% Get Figures and Evaluation Metrics

% Confusion Matrix (Naive Bayes) 
% Uses the true labels to create a confusion matrix for the predictions 
% made by Naive Bayes. Also displays the true positive rate and the true
% negative rate
cm = confusionmat(YTest, predictionsNB);
figure;
confChart = confusionchart(cm, {'0', '1'});
confChart.Title = 'Confusion Matrix (Naive Bayes)';
confChart.RowSummary = 'row-normalized';
confChart.FontSize = 18;
saveas(gcf, 'graphs/cfnb.png', 'png');


% Confusion Matrix (Random Forest)
% Uses the true labels to create a confusion matrix for the predictions 
% made by Random Forest. Also displays the true positive rate and the true
% negative rate
cm = confusionmat(YTest, predictionsRF);
figure;
confChart = confusionchart(cm, {'0', '1'});
confChart.Title = 'Confusion Matrix (Random Forest)';
confChart.RowSummary = 'row-normalized';
confChart.FontSize = 18;
saveas(gcf, 'graphs/cfrf.png', 'png');



% Evaluation Metrics (Naive Bayes)
% Calculates the testing evaluation metrics for Naive Bayes using its
% predictions and the true labels. Evaluation metrics such as test accuracy,
% precision, recall, F1 Score, ROC and AUC are calculated and displayed
accuracy = sum(predictionsNB == YTest) / length(YTest);

TP = sum((predictionsNB== 1) & (YTest == 1));
FP = sum((predictionsNB == 1) & (YTest == 0));
FN = sum((predictionsNB == 0) & (YTest == 1));
TN = sum((predictionsNB == 0) & (YTest == 0));
precision = TP / (TP + FP );
recall = TP / (TP + FN);
f1Score = 2 * (precision * recall) / (precision + recall);

disp('Testing Evaluation Metrics for Naive Bayes:')
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f%%\n', f1Score * 100);
rocObjNB = rocmetrics(YTest, scoresNB(:, 2), 1);
fpr = rocObjNB.Metrics{:, 'FalsePositiveRate'};
tpr = rocObjNB.Metrics{:, 'TruePositiveRate'};
a = trapz(fpr, tpr);
disp(['AUC: ', num2str(a)]);


% Evaluation Metrics (Random Forest)
% Calculates the testing evaluation metrics for Random Forest using its
% predictions and the true labels. Evaluation metrics such as test accuracy,
% precision, recall, F1 Score, ROC and AUC are calculated and displayed
accuracy = sum(predictionsRF == YTest) / length(YTest);

TP = sum((predictionsRF== 1) & (YTest == 1));
FP = sum((predictionsRF == 1) & (YTest == 0));
FN = sum((predictionsRF == 0) & (YTest == 1));
TN = sum((predictionsRF == 0) & (YTest == 0));
precision = TP / (TP + FP );
recall = TP / (TP + FN);
f1Score = 2 * (precision * recall) / (precision + recall);

disp('Testing Evaluation Metrics for Random Forest:')
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f%%\n', f1Score * 100);
rocObjRF = rocmetrics(YTest, scoresRF(:, 2), 1);
fpr = rocObjRF.Metrics{:, 'FalsePositiveRate'};
tpr = rocObjRF.Metrics{:, 'TruePositiveRate'};
a = trapz(fpr, tpr);
disp(['AUC: ', num2str(a)]);


% Comparing ROC curves of Naive Bayes and Random Forest
% Plots the ROC curves for both Naive Bayes and Random Forest on the same
% figure for comparison. Saves the figure to be displayed in the poster

figure
c = cell(2,1);
g = cell(2,1);
[c{1},g{1}] = plot(rocObjNB);
hold on
[c{2},g{2}] = plot(rocObjRF);

modelNames = ["Naive Bayes", "Random Forest"];

for i = 1 : 2
    c{i}.DisplayName = modelNames(i); 
    g{i}(1).DisplayName = join([modelNames(i)," Operating Point"]);
end

hold off
grid on;
xlabel('False Positive Rate', 'FontSize', 18); 
ylabel('True Positive Rate', 'FontSize', 18); 
title('ROC Curves for Naive Bayes and Random Forest', 'FontSize', 18); 

legend('FontSize', 16); 
set(gca, 'FontSize', 16);
saveas(gcf, 'graphs/ROC.png', 'png');