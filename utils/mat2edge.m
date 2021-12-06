function edges=mat2edge(mat)
% convert funcitonal connectivity matrices to edges
num_nodes = size(mat, 1);
num_sub = size(mat, 3);
a = ones(num_nodes);
a = triu(a, 1);
edges = zeros(num_nodes*(num_nodes-1)/2, num_sub);
for idx = 1:num_sub
    mat_temp = mat(:, :, idx);
    edges(:, idx) = mat_temp(a==1);
end
