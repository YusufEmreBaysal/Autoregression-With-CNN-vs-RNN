clc;
clear;
close all;

% Load the MNIST dataset as a preloaded MATLAB dataset
[XTrain, YTrain] = digitTrain4DArrayData; % Training dataset
[XTest, YTest] = digitTest4DArrayData;   % Test dataset

% Number of training iterations
N = 10; 

% Vector to store test accuracies
testAccuracies = zeros(N, 1);
isFirstLoop = true;
epochSize = 50;

for i = 1:N
    fprintf('Simulation %d/%d started...\n', i, N);
    
    % Data augmentation settings
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-20, 20], ...
        'RandXTranslation', [-3 3], ...
        'RandYTranslation', [-3 3]);
    augimds = augmentedImageDatastore(size(XTrain, 1:3), XTrain, YTrain, 'DataAugmentation', imageAugmenter);

    % CNN model layers
    layers = [
        imageInputLayer([28 28 1], 'Name', 'input')
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'batchnorm1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'batchnorm2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

        fullyConnectedLayer(10, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    if (isFirstLoop)
            options = trainingOptions('adam', ...
        'MaxEpochs', epochSize, ... % Number of epochs
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ... % Training details will be displayed
        'Plots', 'training-progress'); % A graphical window will open
    else
            options = trainingOptions('adam', ...
        'MaxEpochs', epochSize, ... % Number of epochs
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none'); % Disable graphical display for subsequent simulations
    end



    % Train the model
    net = trainNetwork(augimds, layers, options);

    % Test the model
    YPred = classify(net, XTest);
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
