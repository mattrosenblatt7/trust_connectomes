function mat_data = timeseries2mat(timeseries)

nsub = size(timeseries, 1);
nnode = size(timeseries, 2);

mat_data = zeros(nnode, nnode, nsub);

for sub_idx = 1:nsub
   mat_data(:, :, sub_idx) =  corr(squeeze(timeseries(sub_idx, :, :))');
end

mat_data(mat_data==1) = 0.99999;
mat_data = atanh(mat_data);