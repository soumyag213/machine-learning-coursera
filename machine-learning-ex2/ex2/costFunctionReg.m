function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
	a = X([i],[1:size(X,2)]);
	g = sigmoid(sum(theta'.*a));
	J += -1*[y(i)*log(g)+ (1-y(i))*log(1-g)];
endfor
s=0;
for j=2:length(theta)
	s += theta(j)^2;
endfor 

J = (J+((lambda*s)/2))/m;
temp = theta;
temp(1)=0;

for j = 1:size(X,2)
	K=0;
	for i = 1:m
		a = X([i],[1:size(X,2)]);
		g = sigmoid(sum(theta'.*a));
		K+= (g-y(i))*X(i,j);
	endfor	
	K = K + (lambda/m)*temp(j);
	grad(j)=K;

endfor
% =============================================================

end
