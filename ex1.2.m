cd C:\Users\zxzx5\Desktop\MachineLearning\assignment\machine-learning-ex1\machine-learning-ex1\ex1
d=load('ex1data1.txt');
x=d(:,1);
y=d(:,2);
m=length(y);
X=[ones(m,1),x];
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;



for i=1:iterations,
  h=X*theta
  GradientSummary1=sum((h-y).*X(:,1))  
  GradientSummary2=sum((h-y).*X(:,2))
  theta(1,1)=theta(1,1)-(alpha/m)*GradientSummary1
  theta(2,1)=theta(2,1)-(alpha/m)*GradientSummary2
end


%sum完變成0  但是h 跟 h'-y   (h'-y).*x 結果都看起來挺正常
computeCost(X,y,theta)