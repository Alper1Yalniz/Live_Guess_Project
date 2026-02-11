%%Alper YALNIZ 230709039 %% Tuygun ERGIN 220709090

clc;
clear all;
close all;

% --- Environment setup ---
% Clear previous variables and figures before starting live prediction

%% Load the previously trained network
load trainedNet
myNet = trainedNet;  % renamed for better readability

% Get the input dimensions required by the network
inputDims = myNet.Layers(1).InputSize;

% Define confidence threshold (percentage)
% Predictions below this value will be marked as uncertain
confThreshold = 60;

%% Initialize camera connection
% Camera index may vary depending on the system
try
    camObj = webcam(1);
catch
    error('Camera could not be accessed. Please check the camera index.');
end

% Create figure window for live camera feed
figure('Name','Live Camera - Object Recognition');

%% Run live prediction while the window is open
while ishandle(gcf)

    % Capture a frame from the camera
    currentFrame = snapshot(camObj);

    % Resize frame to match network input size
    resizedImg = imresize(currentFrame, inputDims(1:2));

    % Perform classification
    [predictedLabel, predictionScores] = classify(myNet, resizedImg);

    % Get highest confidence score
    confValue = max(predictionScores) * 100;

    % Display the camera frame
    imshow(currentFrame);

    % Display result based on confidence level
    if confValue < confThreshold
        % Low confidence: mark as uncertain
        title(sprintf('Object: UNCERTAIN | Confidence: %.1f%%', confValue), ...
              'FontSize',14,'FontWeight','bold','Color','y');
    else
        % High confidence: show predicted label
        title(sprintf('Object: %s | Confidence: %.1f%%', ...
              string(predictedLabel), confValue), ...
              'FontSize',14,'FontWeight','bold','Color','g');
    end

    drawnow;  % Update display
end

%% Release camera resource
clear camObj;

