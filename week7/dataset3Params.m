function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



lowest_error = 1;

for C1 = [.01 .03 .1 .3 1 3 10 30]
  for sigma1 = [.01 .03 .1 .3 1 3 10 30]
    % Create model
    model = svmTrain(X, y, C1, ...
      @(x1, x2) gaussianKernel(x1, x2, sigma1));

    % Get predictions for selected C1 and sigma1
    predictions = svmPredict(model, Xval);
    
    % Calculate error for selected C1 and sigma1
    error = mean(double(predictions ~= yval));
    
    % If new error is less than previous, assign new values to C and sigma
    if error < lowest_error
      C = C1;
      sigma = sigma1;      
      lowest_error = error;
    endif
  endfor
endfor




% =========================================================================

end
