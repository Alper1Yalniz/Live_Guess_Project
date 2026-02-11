%%Alper YALNIZ 230709039 %% Tuygun ERGIN 220709090

clc;
clear all;
close all;

% --- Environment setup ---
% Clear command window, workspace variables and close all open figures
% to ensure the script runs in a clean and controlled environment.

%% This script trains an image classification model
% Note: The dataset folder is assumed to contain one subfolder per class

% Get dataset directory (should be in the same folder as this script)
datasetDir = fullfile(pwd, "dataset");

% Create image datastore from the dataset
imds = imageDatastore(datasetDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display number of images per class
disp("Classes and number of images in the dataset:");
countEachLabel(imds)

%% Split data into training and test sets
% Approximately 80% training and 20% testing
[trainData, testData] = splitEachLabel(imds, 0.8, 'randomized');

%% Use AlexNet as a pretrained base model
% Other pretrained networks could also be tested
baseNet = alexnet;

% Get the input size required by the network
inputDims = baseNet.Layers(1).InputSize;

%% Resize images to match network input size
% Images may have different sizes, so resizing is required
augTrainData = augmentedImageDatastore(inputDims(1:2), trainData);
augTestData  = augmentedImageDatastore(inputDims(1:2), testData);

%% Customize final layers for transfer learning
layers = baseNet.Layers;

% Get number of classes dynamically
numClasses = numel(categories(imds.Labels));

% Replace the fully connected layer based on the number of classes
layers(end-2) = fullyConnectedLayer(numClasses, ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

% Replace the classification layer
layers(end) = classificationLayer;

%% Set training options
% These parameters gave stable results during experiments
trainOpts = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augTestData, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Start training
disp("Training the model... this may take a while.");
trainedNet = trainNetwork(augTrainData, layers, trainOpts);
disp("Training completed successfully.");

%% Evaluate model performance on test data
predictions = classify(trainedNet, augTestData);
trueLabels = testData.Labels;

% Calculate overall accuracy
acc = mean(predictions == trueLabels);
fprintf("Test Accuracy: %.2f%%\n", acc * 100);

%% Display confusion matrix
figure;
confusionchart(trueLabels, predictions);
title("Confusion Matrix");

%% Save the trained network for later use
save trainedNet trainedNet;

% Note: The model can be retrained later with more data if needed
