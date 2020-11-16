clear all;
close all;
rng(123);

%% To reproduce the results:
%Note that Matlab imageDatastore objects only stores absolute paths to
%images, thus, unless you reproduce my path to images you will obtain an
%error. My absolute path to images is:
%/media/jvargas/big/Collaborations/Maxime/OneDrive_1_27-04-2019/Maxime2
%Note that you can always retrain the network as indicated below
load('validationSetAlexNet.mat','valDs','valImgs');
load alexnet

% Evaluation
predictedLabels = classify(net,valDs);
acc = nnz(predictedLabels == valImgs.Labels)/numel(predictedLabels);

fprintf('Obtained accuracy is %4.2f \n',acc);
confusionchart(valImgs.Labels,predictedLabels);
[confMat,order] = confusionmat(valImgs.Labels,predictedLabels);
fprintf('Confusion matrix \n');
confMat
fprintf('\n');

%recall & precision per class
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
f_score=2*(recall.*precision)./(precision+recall);

fprintf('Precision, Recall and F_score for the possitive class: %4.2f %4.2f %4.2f \n', ...
precision(1),recall(1),f_score(1));

fprintf('Precision, Recall and F_score for the negative class: %4.2f %4.2f %4.2f \n', ...
precision(2),recall(2),f_score(2));

%average for all classes
Recall = sum(recall)/size(confMat,1);
Precision = sum(precision)/size(confMat,1);
F_score=2*Recall*Precision/(Precision+Recall);

% TO TRAIN THE NET AGAIN FOLLOW THESE STEPS:
%% Load data, slipt the data and load pretrained AlexNet
%Path to the images. Should be two folders in this path. One composed by possitive
%cases and other by negative cases
pathIm = '.';

imds = imageDatastore(pathIm,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,valImgs] = splitEachLabel(imds,0.8,0.2,'randomized');
imageAugmenter = imageDataAugmenter('RandRotation',[-5 5],'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

trainDs = augmentedImageDatastore([227 227],trainImgs,'DataAugmentation',imageAugmenter);
valDs = augmentedImageDatastore([227 227],valImgs,'DataAugmentation',imageAugmenter);

%pretrained AlexNet can be found from https://www.mathworks.com/help/deeplearning/ref/alexnet.html
net = alexnet;
trainingFeatures = activations(net,trainDs,'fc7','OutputAs','rows');

%% Baseline accuracy with clasical machine learning
classifier = fitcecoc(trainingFeatures,trainImgs.Labels);

valFeatures = activations(net,valDs,'fc7','OutputAs','rows');
predictedLabels = predict(classifier,valFeatures);

acc = nnz(predictedLabels == valImgs.Labels)/numel(predictedLabels);
%confusionchart(valImgs.Labels,predictedLabels);

%% Transfer Learning : Replace the fully connected layer of AlexNet and the classification layer and train
ly = net.Layers;

newFC = fullyConnectedLayer(2);
newC = classificationLayer();

ly(23) = newFC;
ly(25) = newC;

options = trainingOptions('sgdm','MaxEpochs',50,'InitialLearnRate',0.00005,"Plots","training-progress",...
    'ValidationData',valDs,'ValidationFrequency',10,'Shuffle','every-epoch',...
    'MiniBatchSize',20);
    
net = trainNetwork(trainDs,ly,options)

%% Evaluation
predictedLabels = classify(net,valDs);
acc = nnz(predictedLabels == valImgs.Labels)/numel(predictedLabels);

fprintf('Obtained accuracy is %4.2f \n',acc);
confusionchart(valImgs.Labels,predictedLabels);
[confMat,order] = confusionmat(valImgs.Labels,predictedLabels);
fprintf('Confusion matrix \n');
confMat
fprintf('\n');

%recall & precision per class
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
f_score=2*(recall.*precision)./(precision+recall);

fprintf('Precision, Recall and F_score for the possitive class: %4.2f %4.2f %4.2f \n', ...
precision(1),recall(1),f_score(1));

fprintf('Precision, Recall and F_score for the negative class: %4.2f %4.2f %4.2f \n', ...
precision(2),recall(2),f_score(2));

%average for all classes
Recall = sum(recall)/size(confMat,1);
Precision = sum(precision)/size(confMat,1);
F_score=2*Recall*Precision/(Precision+Recall);