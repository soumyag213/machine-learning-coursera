function [theta, J_history] = gradidesc(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	
	L=0; N=0;
	m = length(y); % number of training examples
	i = 1	
	function grads(m,i, L, N)
		if m==0
			return
		else
			s = sum(theta'.*X(i,:));
			L+= (s - y(i))*(X([i],[1]));
			N+= (s - y(i))*(X([i],[2]));			
			grads(m--, i++, L, N);
	end
	l = (alpha*L)/m;
	n = (alpha*N)/m;
	theta(1) = theta(1) - l;
	theta(2) = theta(2) - n;
	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
