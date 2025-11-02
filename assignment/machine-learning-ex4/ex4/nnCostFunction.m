function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
                                   %https://www.coursera.org/learn/machine-learning/resources/Uuxg6 可以參考這個網址 這個作業真的有夠混亂 例如不知道怎麼vectorize、h跟y的Output型式都不知道原理(現在還是不知道) 
                                   
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
Xnobias=X;
m = size(X, 1);
yRE=zeros(m,num_labels);
%temp=ones(num_labels,1);
%temp2=ones(1,m+1);

sumtemp=zeros(num_labels,1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
,%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X=[ones(size(X,1),1),X];
z2=Theta1*X';  
a2=sigmoid(z2);
a2=[ones(1,size(a2',1));a2];%a2第一列就是第一m的a2
h=sigmoid(Theta2*a2); %a3(h)第一列是第一m結果


%debugging 的嘗試

zerotest=find(a2==0);
zerotest2=find(h==0);
zerotest3=find(h==1); %h不能=0或1 現在h沒有=0 但是有很多=1  這代表a3有很多數值無限大的東西???..............  

%for loop版本 好像就是課程要我們做的事情 但我沒完成
%for i=1:m
  %for k=1:num_labels
    %if k==y    
     % temp(k)=-1'*log(h(k,i));
    %else 
   %   temp(k)=-1*log(1-h(k,i));
     
  %  endif
 % temp2(1,i)=sum(temp);

%endfor


%matrix相乘版本1  (y的改寫方法1號)
%for k=1:num_labels
%  indeks=find(y==k);
  
%  ytemp=zeros(num_labels,1);
%  ytemp(k)=1;

  
%  log(1-h(:,indeks));
%  costmk=-1*ytemp'*log(h(:,indeks))-(1-ytemp)'*log(1-h(:,indeks));  %a row of the cost of all the m whose y==k  p.s summing a row is ok   %-(1-ytemp)'*log(1-h(:,indeks))出問題了 會有NaN 好像源自於log下去之後 有些數值會是-inf (下界 超小的值得意思吧 意思是h可能趨近於0)  p.s除以0的話也會得到-Inf
  
%  sumtemp(k)=sum(costmk)
%endfor
%  J=sum(sumtemp)/m;
%  J=trace(sumtemp)


%matrx y的改寫方法2號
for k=1:num_labels
  indeks=find(y==k);
  yRE(indeks,k)=1;

endfor

%vectorize參考網站https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA

%method1
costtemp=-1*yRE'.*log(h)-(1-yRE)'.*log(1-h);

J=sum(sum((costtemp)))/m+(lambda/(2*m))*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)));


%method2
%costtemp=-1*yRE.*log(h)-(1-yRE).*log(1-h)

%J=trace(costtemp)/m; %trace是對角線總和 如果matrix不是正方形 他也有奇怪的方法可以判斷 反正我是不懂啦................

%========================================================================
%part 2 backpropagation 
delta3=h-y';  %size 跟y依樣 也就是numlables*m matrix 

%z2=[ones(1,m);sigmoidGradient(z2)];%為什麼z2竟然也需要加bias bias到底是做什麼用的 課程什麼都沒講 ........ A: 的確不用加 參考https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw  Q1第二點 
delta2=(Theta2(:,2:end)'*delta3).*sigmoidGradient(z2);  %第一列就是第一m的delta2 

size(delta2);  %delta 25*5000 

%∆(l) = ∆(l) + δ(l+1)(a(l))

DELTA1=delta2*X;
DELTA2=delta3*a2';  %sigmoid(z2)=沒有BIAS的a2;
Theta1_grad=(DELTA1)/m;
Theta2_grad=(DELTA2)/m;

size(X); %5000*401
size(a2); %26*5000
size(Theta1_grad);  %25*401 跟Theta1 應該是同大小的 
size(Theta2_grad);  %10*26 跟Theta2 同大小 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
