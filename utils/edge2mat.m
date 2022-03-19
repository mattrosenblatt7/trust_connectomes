function A = edge2mat(edges)
% convert edges into matrix  (assumes symmetric)
num_nodes=(1+sqrt(1-4*-2*length(edges)))/2;
A = ones(num_nodes);
A = 1-tril(A);

A(A>0) = edges;
A = A+A';
