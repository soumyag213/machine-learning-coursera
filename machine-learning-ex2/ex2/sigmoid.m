function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
[s,t] = size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for j =1:s	
	for i = 1:t
		d = e.^((-1)*z(j,i));
		g(j,i) = 1/(1+d);
	endfor

endfor
% =============================================================

end
