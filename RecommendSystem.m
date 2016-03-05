
%% Initialization
clear ; close all; clc

% convert data from .csv to .mat if needed
% convertCSV2MAT('datatest.csv', 'datatest.mat');
% X = load('X.mat');             % feature vector of each item 属性特征
% Theta = load('Theta.mat');     % parameter vector of every user 用户偏好
% Y = load('Y.mat');             % The rating user j gaves to item i
% R = load('R.mat');             % 1 if user j gave a rating to item i

%  Load data
load ('data_test.mat');

%  get mean rate for the n-th Y
% n = 3;
% fprintf('Average rating for %d : %f / 5\n\n', ...
%        n, mean(Y(n, R(n, :))));

% "visualize" the ratings matrix by plotting it with imagesc
% imagesc(Y);
% ylabel('Y');
% xlabel('Users');
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

% get logic matrix R
R =  Y ~= 0;

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_items = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_items, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];


%% ================== This part is only for testing ====================

%  Reduce the data for testing use with faster speed
% num_users = 4; num_items = 5; num_features = 3;
% X = X(1:num_items, 1:num_features);
% Theta = Theta(1:num_users, 1:num_features);
% Y = Y(1:num_items, 1:num_users);
% R = R(1:num_items, 1:num_users);

%  Check gradients by running checkNNGradients
% fprintf('\nChecking Gradients (without regularization) ... \n');
% checkCostFunction;
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============ This part is for optimaizing lambda =============
% 不适合此例，由于数据较少，很容易overfitting，lambda取值偏大较好
% 
% % Selected values of lambda
% lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
% 
% % You need to return these variables correctly.
% error_train = zeros(length(lambda_vec), 1);
% % error_val = zeros(length(lambda_vec), 1);     % no cross validation set
%                                                 % in this example
% 
% % optimize lambda
% for i = 1:length(lambda_vec)
%     lambda = lambda_vec(i);
%     
%     theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_items, ...
%         num_features, lambda)), ...
%         initial_parameters, options);
%     
%     % Unfold the returned theta back into U and W
%     X = reshape(theta(1:num_items*num_features), num_items, num_features);
%     Theta = reshape(theta(num_items*num_features+1:end), ...
%         num_users, num_features);
% 
%     %  cost function
%     error_train(i) = cofiCostFunc([X(:) ; Theta(:)], Ynorm, R, ...
%         num_users, num_items, num_features, lambda);
% end;
% 
% plot(lambda_vec, error_train);
% legend('Train');
% xlabel('lambda');
% ylabel('Error');
% 
% fprintf('lambda\t\tTrain Error\tValidation Error\n');
% for i = 1:length(lambda_vec)
% 	fprintf(' %f\t%f\n', ...
%             lambda_vec(i), error_train(i));
% end
% 
% %  Check gradients with new lambda value
% checkCostFunction(lambda);
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ================ This part is for Learning Ratings ==================
fprintf('\nTraining collaborative filtering...\n');

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_items, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_items*num_features), num_items, num_features);
Theta = reshape(theta(num_items*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% ============== This part is for Recommendation ================

user_id = 5;
p = X * Theta';
my_predictions = p(:, user_id) + Ymean;

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for user id = %d:\n', user_id);

num_recommend = 10;
for i = 1:num_recommend
    j = ix(i);
    fprintf('Predicting rating %.1f for item id = %d\n', my_predictions(j), ...
        j);
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =========== This part is for Visualizing recommed =============

% "visualize" the ratings matrix by plotting it with imagesc
% imagesc(p);
% ylabel('Predictions');
% xlabel('Users');






