cd C:\Users\zxzx5\Desktop\MachineLearning\assignment\machine-learning-ex1\machine-learning-ex1\ex1
d=load('ex1data1.txt');
x=d(:,1);
y=d(:,2);
m=length(y);
X=[ones(m,1),x];
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;
v1=[0;]
v2=[0;]
for i=1:iterations,
  for j=1:m,
    h= theta(1,1)+theta(2,1)*X(j,1);
    f=(h-y(j,1))*X(j,1);
    g=(h-y(j,1))*X(j,2);
    v1=[v1;f];
    v2=[v2;f];
   end
  theta(1,1)=theta(1,1)-alpha/m*sum(v1)
  theta(2,1)=theta(2,1)-alpha/m*sum(v2)
end

computeCost(X, y, theta)
