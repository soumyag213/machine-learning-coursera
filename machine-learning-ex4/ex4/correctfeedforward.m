%Part 1: implementing feedforward 
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2*Theta2';
htheta = sigmoid(z3);

for k = 1:num_labels
	yk = y==k;
	hthetak = htheta(:,k);
	Jk = (1/m)*sum(-yk.*log(hthetak) - (1-yk).*log(1-hthetak));
	J+=Jk;
endfor

reg = (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)))* (lambda/(2*m));

J = J+reg;

