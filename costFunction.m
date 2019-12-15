function [J, grad] = costFunction(theta, X, y)
  
m = length(y); % number of training examples
grad = zeros(size(theta));

J = (-1/m)*[y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta))]; %cost function
g = (1/m)*[X'*(sigmoid(X*theta) - y)];
grad = g;

end  
