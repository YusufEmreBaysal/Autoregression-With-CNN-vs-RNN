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
        'Plots', 'training-progress'); % Enable plot display for the first simulation
    else
            autoencoderOptions = trainingOptions('adam', ...
        'MaxEpochs', autoencoderSize, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none'); % Disable plot display for subsequent simulations
    end


    % Train the Autoencoder
    autoencoder = trainNetwork(XTrain, XTrain, autoencoderLayers, autoencoderOptions);

    % Compress data with the autoencoder
    compressedXTrain = predict(autoencoder, XTrain);
    compressedXTest = predict(autoencoder, XTest);

    % Data augmentation settings
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-20, 20], ...
        'RandXTranslation', [-3 3], ...
        'RandYTranslation', [-3 3]);
    augimds = augmentedImageDatastore(size(compressedXTrain, 1:3), compressedXTrain, YTrain, 'DataAugmentation', imageAugmenter);

    % CNN model layers
    layers = [
        imageInputLayer(size(compressedXTrain, 1:3), 'Name', 'input')
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

    % CNN training settings
    if(isFirstLoop)
            options = trainingOptions('adam', ...
        'MaxEpochs', epochSize, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress'); % Enable plot display for the first simulation
    else
            options = trainingOptions('adam', ...
        'MaxEpochs', epochSize, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none'); % Disable plot display for subsequent simulations
    end


    % Train the CNN model
    net = trainNetwork(augimds, layers, options);

    % Test the model
    YPred = classify(net, compressedXTest);
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
