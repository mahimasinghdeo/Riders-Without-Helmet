clear;
   foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 50);

% to run vid1 (no helmet)
videoReader = vision.VideoFileReader('vid1.mp4');

% to run vid2 (helmet)
% videoReader = vision.VideoFileReader('vid2.mp4');

% in case of vid1 the value of i should be 1:180. In case of vid2 i should be 1:200.
for i = 1:180
    frame = step(videoReader); % read the next video frame
    %figure; imshow(frame); title('Video Frame');
    foreground = step(foregroundDetector, frame);
end

%figure; imshow(frame); title('Video Frame');
%figure; imshow(foreground); title('Foreground');

se = strel('square', 3);
filteredForeground = imopen(foreground, se);
%figure; imshow(filteredForeground); title('Clean Foreground');
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', true, ...
    'MinimumBlobArea', 2000);
[areas, centroids, bbox] = step(blobAnalysis, filteredForeground);



result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
numCars = size(bbox, 1);
result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
    'FontSize', 14);
%figure; imshow(result); title('Detected Moving Objects');
videoPlayer = vision.VideoPlayer('Name', 'Detected Moving Objects');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
se = strel('square', 3); % morphological filter for noise removal
x = 0;
%release(videoReader); % close the video file
%videoReader = vision.VideoFileReader('vid2.mp4');
while ~isDone(videoReader)
    x = x + 1;
    frame = step(videoReader); % read the next video frame

    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);

    % Use morphological op?ove noise in the foreground
    filteredForeground = imopen(foreground, se);

    % Detect the?????????
%ents with the specified
%minimum area, and
    % compute their bounding boxes
    [areas, centroids, bbox] = step(blobAnalysis, filteredForeground);
    if(mod(x,100) == 0)
        %disp(x);
        %for i=1:size(bbox, 1)
        %centroids(i,:) = [ bbox(i,1)+bbox(i,3)/2 ; bbox(i,2)+bbox(i,4)/2 ];
        %areas(i,1) =  bbox(i,3)*bbox(i,4);
        %areas(:,1)
        %disp('hello');
        cropped = imcrop(frame, bbox);
        
        %imshow(cropped);
        cd 'test'
        cropped = imresize(cropped,[200 113]);
        imwrite(cropped,'cropped.jpg');
        %testgui;
        cd ..
        %info = imfinfo('cropped.jpg');
        
        %cropped = imread('cropped.jpg');
        %figure; imshow(cropped); title('cropped');
        %{
        height = size(cropped,1);
        width = size(cropped,2);
        
        
        ar = width/ height;
        disp(ar);
        %}
        
        pause(3);
        %disp('hello222');
        %end
        
    end
    
    % Draw bounding boxes around the detected cars
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

    % Display the number of cars found in the video frame
    numCars = size(bbox, 1);
    result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);

    step(videoPlayer, result);  % display the results
end

release(videoReader); % close the video file






syntheticDir   = fullfile('train');
handwrittenDir = fullfile('test');
trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%testSet = cropped;
countEachLabel(trainingSet);
%countEachLabel(testSet)


%figure;
%{
subplot(2,2,1);
imshow(trainingSet.Files{9});

subplot(2,2,2);
imshow(trainingSet.Files{4});

%subplot(2,3,3);
%imshow(trainingSet.Files{7});

subplot(2,2,3);
imshow(testSet.Files{1});

subplot(2,2,4);
imshow(testSet.Files{2});

%subplot(2,3,6);
%imshow(testSet.Files{97});
%}
exTestImage = readimage(testSet,2);
%exTestImage = cropped;
processedImage = imbinarize(rgb2gray(exTestImage));

%figure;
%{
subplot(1,2,1)
imshow(exTestImage)

subplot(1,2,2)
imshow(processedImage)
%}
img = readimage(trainingSet, 99);
%figure;imshow(img);

% Extract HOG features and HOG visualization
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
[hog_32x32, vis32x32] = extractHOGFeatures(img,'CellSize',[32 32]);

% Show the original image
%figure;
%subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
%{
subplot(2,3,4);
plot(vis8x8);
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

subplot(2,3,5);
plot(vis16x16);
title({'CellSize = [16 16]'; ['Length = ' num2str(length(hog_16x16))]});

subplot(2,3,6);
plot(vis32x32);
title({'CellSize = [32 32]'; ['Length = ' num2str(length(hog_32x32))]});
%}
cellSize = [16 16];
hogFeatureSize = length(hog_16x16);
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);

    img = rgb2gray(img);

    % Apply pre-processing steps
    img = imbinarize(img);

    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
%disp(trainingLabels)


% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.

classifier = fitcecoc(trainingFeatures, trainingLabels);
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.

[testFeatures,testLabels] = helperExtractHOGFeaturesFromImageSet(testSet,hogFeatureSize,cellSize);

% Make class predictions using the test features.

predictedLabels = predict(classifier, testFeatures);

%disp("From test folder :");
%for i = 1:size(predictedLabels)
       
   if(predictedLabels(3) == "motorcycle")
        h = msgbox('The object detected is a motorcycle\n');
        %fprintf("The object detected is a motorcycle\n"); 
        
   else
     
       h1= msgbox('The object detected is not a motorcycle\n');
   end
   
%end

%testgui;

if predictedLabels(3)== "motorcycle"

    

X2 = imcrop(cropped, [0 0 113 40]);
%figure;
%imshow(X2);title('cropped head')
cd 'test1'
croppedhead = imresize(X2,[40 113]);
imwrite(X2,'croppedhead.jpg');
%testgui;
cd ..




syntheticDir   = fullfile('train2');
handwrittenDir = fullfile('test1');
trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%testSet = cropped;
countEachLabel(trainingSet)
%countEachLabel(testSet)

%{
figure;

subplot(2,2,1);
imshow(trainingSet.Files{9});

subplot(2,2,2);
imshow(trainingSet.Files{4});

%subplot(2,3,3);
%imshow(trainingSet.Files{7});

subplot(2,2,3);
imshow(testSet.Files{1});

subplot(2,2,4);
imshow(testSet.Files{2});

%subplot(2,3,6);
%imshow(testSet.Files{97});

exTestImage = readimage(testSet,2);
%exTestImage = cropped;
processedImage = imbinarize(rgb2gray(exTestImage));

figure;

subplot(1,2,1)
imshow(exTestImage)

subplot(1,2,2)
imshow(processedImage)
%}
img = readimage(trainingSet, 20);
%figure;imshow(img);

% Extract HOG features and HOG visualization
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
%{
% Show the original image
figure;
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
subplot(2,3,4);
plot(vis8x8);
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

subplot(2,3,5);
plot(vis16x16);
title({'CellSize = [16 16]'; ['Length = ' num2str(length(hog_16x16))]});

subplot(2,3,6);
plot(vis4x4);
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
%}
cellSize = [16 16];
hogFeatureSize = length(hog_16x16);
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);

    img = rgb2gray(img);

    % Apply pre-processing steps
    img = imbinarize(img);

    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
%disp(trainingLabels)


% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.

classifier = fitcecoc(trainingFeatures, trainingLabels);
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.

[testFeatures,testLabels] = helperExtractHOGFeaturesFromImageSet(testSet,hogFeatureSize,cellSize);

% Make class predictions using the test features.

predictedLabels = predict(classifier, testFeatures);
%disp(predictedLabels(3))

disp("From test folder:");
for i = 1:size(predictedLabels)
    disp(predictedLabels)
end
   %predictedLabels(3)="helmet";   
   if(predictedLabels(3) == "helmet")
        h2 = msgbox('The rider is wearing a helmet :)'); 
   else
       h3 = msgbox('The rider is NOT wearing a helmet !');
   end
   
%end

testgui;


if (predictedLabels(3)=="head")
%X3 = imcrop(cropped, [20 65 113 70]);
testgui2;
%figure;
%imshow(X3);title('license plate');
%else
 %   disp("Person is wearing helmet.");
end

%else
%    disp("Not a mototrcycle. Exiting.....")
end



