function [all_theta] = oneVsAll(X, y, num_labels, lambda)


%J=zeros(10,1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables



m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

initial_theta = zeros(n+1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);


%他把數字零的y值設成一了
for i=1:num_labels

    [all_theta(i,:)] = ...
      fmincg (@(t)(lrCostFunction(t, X, (y==i), lambda)), ...
      initial_theta, options)
  %J(i)=-1*class(:,i)*log(h)-(1-class(:,i))*log(1-h)


  
endfor



%用資源區的test case試過之後 alltheta也是正確的  但是程式還是0分 不知道為何  後:原因就是我不小心多加了load('ex3data1.mat') 這會直接導致上傳作業的時候爆開 因為她會覆蓋掉我們test case的xy 值  
 


% =========================================================================


end
