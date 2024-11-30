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
autoencoderSize = 10;
epochSize = 50;

for i = 1:N
    fprintf('Simulation %d/%d started...\n', i, N);
    
    % Autoencoder model layers
    autoencoderLayers = [
        imageInputLayer([28 28 1], 'Name', 'input')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'batchnorm1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') % 28x28 -> 14x14
        
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'batchnorm2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2') % 14x14 -> 7x7
        
        transposedConv2dLayer(3, 8, 'Stride', 2, 'Cropping', 'same', 'Name', 'deconv1')
        reluLayer('Name', 'relu3') % 7x7 -> 14x14
        transposedConv2dLayer(3, 16, 'Stride', 2, 'Cropping', 'same', 'Name', 'deconv2')
        reluLayer('Name', 'relu4') % 14x14 -> 28x28
        convolution2dLayer(3, 1, 'Padding', 'same', 'Name', 'outputConv') % Final layer
        regressionLayer('Name', 'output')
    ];

    % Autoencoder training settings
    if (isFirstLoop)
         autoencoderOptions = trainingOptions('adam', ...
        'MaxEpochs', autoencoderSize, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress'); % Display progress graph during training
    else
         autoencoderOptions = trainingOptions('adam', ...
        'MaxEpochs', autoencoderSize, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none');
    end


    % Train the autoencoder
    autoencoder = trainNetwork(XTrain, XTrain, autoencoderLayers, autoencoderOptions);

    % Data compression with autoencoder
    compressedXTrain = predict(autoencoder, XTrain); % Compressed training data
    compressedXTest = predict(autoencoder, XTest);   % Compressed test data

    % Check autoencoder output size
    outputSize = size(compressedXTrain);
    featureDim = outputSize(2); % Feature dimension (e.g., 14)
    timeSteps = outputSize(1); % Time steps (e.g., 14)

    % Format data for RNN
    compressedXTrainSeq = squeeze(permute(compressedXTrain, [2, 1, 3, 4])); % (14, 14, 1, numSamples) -> (14, 14, numSamples)
    compressedXTestSeq = squeeze(permute(compressedXTest, [2, 1, 3, 4]));

    % RNN model layers
    layers = [
        sequenceInputLayer(featureDim, 'Name', 'input') % Dynamically set feature dimension
        lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm') % LSTM layer
        fullyConnectedLayer(10, 'Name', 'fc') % 10 classes (digits 0-9)
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % RNN training settings
     if (isFirstLoop)
        options = trainingOptions('adam', ...
            'MaxEpochs', epochSize, ...
            'MiniBatchSize', 128, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'Plots', 'training-progress'); % Display progress graph during training
    else
        options = trainingOptions('adam', ...
            'MaxEpochs', epochSize, ...
            'MiniBatchSize', 128, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'Plots', 'none');
    end

    % Convert data format to cell array for training
    XTrainSeqCell = arrayfun(@(idx) compressedXTrainSeq(:, :, idx)', 1:size(compressedXTrainSeq, 3), 'UniformOutput', false);
    XTestSeqCell = arrayfun(@(idx) compressedXTestSeq(:, :, idx)', 1:size(compressedXTestSeq, 3), 'UniformOutput', false);

    % Train the RNN model
    net = trainNetwork(XTrainSeqCell, YTrain, layers, options);

    % Test the model
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
