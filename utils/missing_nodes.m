function bad_idx = missing_nodes(all_mats)

ntask = size(all_mats, 4);
nsub = size(all_mats, 3);

sub_check_all = zeros(nsub, 1);
for sub_idx = 1:nsub
    sub_check = 0;
    for task_idx = 1:ntask
        temp = all_mats(:, :, sub_idx, task_idx);
        if length(find(sum(temp==0)==268))>0
            sub_check = sub_check + 1;
        end
    end
    sub_check_all(sub_idx) = sub_check;
end

bad_idx = find(sub_check_all>0);

end