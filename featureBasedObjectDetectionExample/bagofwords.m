%% Image Category Classification Using Bag of Features
% This example shows how to use a bag of features approach for image
% category classification. This technique is also often referred to as bag
% of words. Visual image categorization is a process of assigning a
% category label to an image under test. Categories may contain images
% representing just about anything, for example, dogs, cats, trains, boats.
%
% Copyright 2014 The MathWorks, Inc. 

%% Download Caltech101 Image Set
% To learn about bag of features image category classification, you will
% first download a suitable image data set. One of the most widely cited
% and used data sets is  <http://www.vision.caltech.edu/Image_Datasets/Caltech101 Caltech 101>, collected by Fei-Fei Li, Marco Andreetto,
% and Marc 'Aurelio Ranzato.

% Location of the compressed data set
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile('Q:', 'caltech101'); % define output folder

%%
% Note that downloading the set from the web can take a very long time depending on
% your Internet connection. It can be 30 minutes or more since the set contains 
% 126MB of data. The commands below will block MATLAB for that period of
% time. Alternatively, you can use your web browser to first download the
% set to your local disk. If you choose that route, re-point the 'url' variable
% above to the file that you downloaded.

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

%% Load Image Sets
% Instead of operating on the entire Caltech 101 set, which can be time
% consuming, use three categories: airplanes, ferry, and laptop.
% Note that for the bag of features approach to be effective, majority of
% each image's area must be occupied by the subject of the category, for
% example, an object or a type of scene.

rootFolder = fullfile(outputFolder, '101_ObjectCategories');

%%
% Construct an array of image sets based on the following categories from
% Caltech 101: 'airplanes', 'ferry', 'laptop'. Use |imageSet| class to help
% you manage the data. Since |imageSet| operates on image file locations,
% and therefore does not load all the images into memory, it is safe to use
% on large image collections.
imgSets = [ imageSet(fullfile(rootFolder, 'airplanes1')), ...
            imageSet(fullfile(rootFolder, 'ferry')), ...
            imageSet(fullfile(rootFolder, 'laptop')) ];

%%
% Each element of the |imgSets| variable now contains images associated
% with the particular category.  You can easily inspect the number of
% images per category as well as category labels as shown below:

{ imgSets.Description } % display all labels on one line
[imgSets.Count]         % show the corresponding count of images

%%
% Note that the labels were derived from directory names used to construct
% the image sets, but can be customized by manually setting the Description
% property of the imageSet object.

%% Prepare Training and Validation Image Sets
% Since |imgSets| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set is balanced.

minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
[imgSets.Count]

%%
% Separate the sets into training and validation data. Pick 30% of images
% from each set for the training data and the remainder, 70%, for the 
% validation data. Randomize the split to avoid biasing the results.

[trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

%%
% The above call returns two arrays of imageSet objects ready for training
% and validation tasks. Below, you can see example images from the three
% categories included in the training data.

airplanes = read(trainingSets(1),1);
ferry     = read(trainingSets(2),1);
laptop    = read(trainingSets(3),1);

figure

subplot(1,3,1);
imshow(airplanes)
subplot(1,3,2);
imshow(ferry)
subplot(1,3,3);
imshow(laptop)

%% Create a Visual Vocabulary and Train an Image Category Classifier
% Bag of words is a technique adapted to computer vision from the
% world of natural language processing. Since images do not actually
% contain discrete words, we first construct a "vocabulary" of 
% <matlab:doc('extractFeatures'); SURF> features representative of each image category.

%%
% This is accomplished with a single call to |bagOfFeatures| function,
% which:
%
% # extracts SURF features from all images in all image categories
% # constructs the visual vocabulary by reducing the number of features
%   through quantization of feature space using K-means clustering
bag = bagOfFeatures(trainingSets);

%%
% Additionally, the bagOfFeatures object provides an |encode| method for
% counting the visual word occurrences in an image. It produced a histogram
% that becomes a new and reduced representation of an image.

img = read(imgSets(1), 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

%%
% This histogram forms a basis for training a classifier and for the actual
% image classification. In essence, it encodes an image into a feature vector. 
%
% Encoded training images from each category are fed into a classifier
% training process invoked by the |trainImageCategoryClassifier| function.
% Note that this function relies on the multiclass linear SVM classifier
% from the Statistics and Machine Learning Toolbox(TM).

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

%%
% The above function utilizes the |encode| method of the input |bag| object
% to formulate feature vectors representing each image category from the 
% |trainingSets| array of imageSet objects.

%% Evaluate Classifier Performance
% Now that we have a trained classifier, |categoryClassifier|, let's
% evaluate it. As a sanity check, let's first test it with the training
% set, which should produce near perfect confusion matrix, i.e. ones on 
% the diagonal.

confMatrix = evaluate(categoryClassifier, trainingSets);

%%
% Next, let's evaluate the classifier on the validationSet, which was not
% used during the training. By default, the |evaluate| function returns the
% confusion matrix, which is a good initial indicator of how well the
% classifier is performing.

confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));

%%
% Additional statistics can be derived using the rest of arguments returned
% by the evaluate function. See help for |imageCategoryClassifier/evaluate|.
% You can tweak the various parameters and continue evaluating the trained
% classifier until you are satisfied with the results.

%% Try the Newly Trained Classifier on Test Images
% You can now apply the newly trained classifier to categorize new images.

%img = imread(fullfile(rootFolder, 'airplanes1', 'image_0001.jpg'));
img = imread('58D5.tmp.jpg');
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)

displayEndOfDemoMessage(mfilename)
