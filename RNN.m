clc;
clear;
close all;

% Load the MNIST dataset
[XTrain, YTrain] = digitTrain4DArrayData; % Training dataset
[XTest, YTest] = digitTest4DArrayData;   % Test dataset

% Number of training iterations
N = 10; 

% Vector to store test accuracies
testAccuracies = zeros(N, 1);
isFirstLoop = true;
epochSize = 50;

% Format data to be suitable for RNN
% Each 28x28 image is reshaped into 28 time steps with 28 features.
XTrainSeq = squeeze(permute(XTrain, [2, 1, 3, 4])); % (28, 28, 1, numSamples) -> (28, 28, numSamples)
XTestSeq = squeeze(permute(XTest, [2, 1, 3, 4]));

for i = 1:N
    fprintf('Simulation %d/%d started...\n', i, N);
    
    % RNN model layers
    layers = [
        sequenceInputLayer(28, 'Name', 'input') % 28 features (each row as a time step)
        lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm') % LSTM layer
        fullyConnectedLayer(10, 'Name', 'fc') % 10 classes (digits 0-9)
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    if (isFirstLoop)
        options = trainingOptions('adam', ...
            'MaxEpochs', epochSize, ... % Number of epochs
            'MiniBatchSize', 128, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'Plots', 'training-progress'); % Display training progress graph
    else
        options = trainingOptions('adam', ...
            'MaxEpochs', epochSize, ...
            'MiniBatchSize', 128, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'Plots', 'none'); % Disable graph display
    end

    % Manual data preparation for training instead of augmentedImageDatastore
    XTrainSeqCell = arrayfun(@(idx) XTrainSeq(:, :, idx)', 1:size(XTrainSeq, 3), 'UniformOutput', false);
    XTestSeqCell = arrayfun(@(idx) XTestSeq(:, :, idx)', 1:size(XTestSeq, 3), 'UniformOutput', false);
    
    % Train the model
    net = trainNetwork(XTrainSeqCell, YTrain, layers, options);

    % Measure test accuracy
    YPred = classify(net, XTestSeqCell);
    accuracy = sum(YPred == YTest) / numel(YTest);
    testAccuracies(i) = accuracy;
    
    fprintf('Simulation %d completed: Test Accuracy = %.2f%%\n', i, accuracy * 100);
    isFirstLoop = false;
end

% Calculate the average of test accuracies
averageAccuracy = mean(testAccuracies);

% Display results
fprintf('\nSimulation Results:\n');
fprintf('Accuracy of each simulation:\n');
disp(testAccuracies);
fprintf('Average Accuracy: %.2f%%\n', averageAccuracy * 100);
