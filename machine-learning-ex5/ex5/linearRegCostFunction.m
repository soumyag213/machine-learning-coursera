function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTcdFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%size(theta)
J = sum((X*(theta) - y).^2);
temp = theta;
temp(1,:)=0;
reg = lambda*sum(temp.*temp);
J = (0.5/m)*(J+reg);


K= J;
%size(grad)
grad(1,1) = (1/m)*sum( ((X*theta)-y) .* (X(:,1)) );
for K = 2:size(theta)
grad(K,1) = (1/m)*sum( ((X*theta)-y) .* (X(:,K)) ) + (lambda/m)*theta(K,1);
endfor
% =========================================================================

grad = grad(:);

end
