D=load('ex2data1.txt');
X=D(:,[1, 2]); y = D(:, 3);
X=[ones(length(X),1),X];
[m,n]=size(X);
initial_theta = zeros(n , 1);
test_theta = [-24; 0.2; 0.2];
% Compute and display initial cost and gradient
[J, grad] = costFunction(test_theta, X, y)

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

 theta
 cost
 
 
 plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;