function [X_ts_enhanced, y_predict, q_sq, r, ts_corr, edge_corr] = time_series_enhancement(X_ts, y, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enhancement connectome data for a behavioral prediction.
% Please use responsibly
%   REQUIRED INPUTS
%        dataset            dataset name that you want to load. you will
%                           need to alter these paths (see in code below)
%        seed_start         start random seed
%        seed_end           end random seed. for only one iteration (seed),
%                           set seed_end=seed_start
%
%   OUTPUTS
%       Save a .mat file for each seed in save_path (set this path below).
%       The file contains the following outputs:
%           all_corr        correlations between predicted and meaured
%                           behavior (Pearson's r)
%           all_q           prediction R^2
%           all_r_rank   	Spearman's rank
%           edge_corr       correlations (Pearson's) between original and
%                           enhanced connectome edges
%           noise_abs_mean  mean absolute change of edges that were changed
%           noise_abs_max   max absolute change of edges that were changed
%           perc_change_all calculation of how large enhancement pattern is
%                           relative to entire connectome
%           all_sex_acc     self-reported sex classification accuracy for
%                           connectomes that were enhnaced for another
%                           prediction
%
% example call:
%    connectome_enhancement(mat_rest1, hcp_iq, 'per_enhance', 0.2, 'scale', 0.01,...
%    'train_model', 1, 'per_model', 0.1, 'show_results', 1, 'seed', 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('../utils'))

%% parse inputs
p=inputParser;
addRequired(p,'X_ts',@isnumeric);
addRequired(p,'y',@isnumeric);
addParameter(p,'scale',0.1,@isnumeric);
addParameter(p,'train_model',1, @isnumeric);
addParameter(p,'per_model',0.1,@isnumeric);
addParameter(p,'show_results',0.1,@isnumeric);
addParameter(p,'seed',1,@isnumeric);
addParameter(p,'v_alpha',1e-6,@isnumeric);
addParameter(p,'lambda',NaN,@isnumeric);
addParameter(p,'kfolds',10,@isnumeric);

parse(p, X_ts, y, varargin{:});

scale_factor = p.Results.scale;
train_model = p.Results.train_model;
per_model = p.Results.per_model;
show_results = p.Results.show_results;
seed = p.Results.seed;
v_alpha = p.Results.v_alpha;
lambda = p.Results.lambda;
kfolds = p.Results.kfolds;

%% enhance data
nsub = size(X_ts, 1);
nnode = size(X_ts, 2);
ntime = size(X_ts, 3);

% alter data, running over different scales
% scale 0 is original pred
rng(seed)

% get matrix from time series
X_mat = timeseries2mat(X_ts);

% get edge data and correlate with IQ
edge_data = mat2edge(X_mat);
corr_edge_iq = corr(edge_data', y);

% rank correlation strengths
corr_edge_iq_mat = edge2mat(corr_edge_iq);
corr_edge_iq_mat(tril(ones(nnode))==1) = 0;
[val, loc] = sort(abs(corr_edge_iq_mat(:)), 'descend');
% val = val/sum(val);  % sum to 0
[row, col] = ind2sub([nnode, nnode], loc);

% select noise pattern (e.g., sinusoid)
npat = sin(2*pi*(1:ntime) / (ntime/4));
npat = npat(:);

X_ts_enhanced = X_ts;

% loop over all nodes and add pattern
for loc_idx=1:length(row)
    
    if mod(loc_idx, 1000)==0
        disp(loc_idx)
    end
    
    row_temp=row(loc_idx); col_temp=col(loc_idx);
    original_iq_corr_val = corr_edge_iq_mat(row_temp, col_temp);
    
    if val(loc_idx)~=0  % shortens loop time (only upper triangle)
        
        % increase positive correlations by both adding to same node
        if original_iq_corr_val>=0
            for sub_idx = 1:nsub
                X_ts_enhanced(sub_idx, row_temp, :) =  squeeze(X_ts_enhanced(sub_idx, row_temp, :)) + ...
                    scale_factor*val(loc_idx)*y(sub_idx)*npat;
                
                X_ts_enhanced(sub_idx, col_temp, :) =  squeeze(X_ts_enhanced(sub_idx, col_temp, :)) + ...
                    scale_factor*val(loc_idx)*y(sub_idx)*npat;
            end
            
            % decrease negative correlations by adding to one and subtracting
            % from other node
        elseif original_iq_corr_val<0
            for sub_idx = 1:nsub
                X_ts_enhanced(sub_idx, row_temp, :) =  squeeze(X_ts_enhanced(sub_idx, row_temp, :)) + ...
                    scale_factor*val(loc_idx)*y(sub_idx)*npat;
                
                X_ts_enhanced(sub_idx, col_temp, :) =  squeeze(X_ts_enhanced(sub_idx, col_temp, :)) - ...
                    scale_factor*val(loc_idx)*y(sub_idx)*npat;
            end
        end
    end
end

X_mat_enhanced = timeseries2mat(X_ts_enhanced);
%% run predictions

if train_model
    [q_sq, r, ~, y_predict, ~, ~] = ridgeCPM(X_mat_enhanced, y, 'per_feat',...
        per_model, 'v_alpha', v_alpha, 'kfolds', kfolds, 'seed', seed, 'lambda', lambda);
end


% finding similarity between real and enhanced time series
ts_corr = zeros(nsub, nnode);
for node_idx = 1:nnode
    ts_corr(:, node_idx) =...
        diag(corr(squeeze(X_ts(:, node_idx, :))',...
        squeeze(X_ts_enhanced(:, node_idx, :))'));
end


% find edge correlation between original and enhanced connectomes
edge_corr = diag(corr(edge_data, mat2edge(X_mat_enhanced)));


if show_results
    % plotting different time-series correlations
    figure;
    ts_corr_all = sort(ts_corr(:), 'ascend');
    n_ts_corr_all = length(ts_corr_all);
    min_corr = min(ts_corr_all);
    [sub, node] = find(ts_corr==min_corr);
    subplot(2, 2, 1),
    plot(squeeze(X_ts(sub, node, :))), hold on,
    plot(squeeze(X_ts_enhanced(sub, node, :))), hold off
    legend('Original', 'Enhanced')
    title(['Min original/enhanced corr. (r=', num2str(round(min_corr, 2)), ')'],...
        'FontSize', 8)
    xlabel('Time'), ylabel('Signal')
    
    corr25 = ts_corr_all(round(.25*n_ts_corr_all));
    [sub, node] = find(ts_corr==corr25);
    subplot(2, 2, 2),
    plot(squeeze(X_ts(sub, node, :))), hold on,
    plot(squeeze(X_ts_enhanced(sub, node, :))), hold off
    legend('Original', 'Enhanced')
    title(['25th % original/enhanced corr. (r=', num2str(round(corr25, 2)), ')'],...
        'FontSize', 8)
    xlabel('Time'), ylabel('Signal')
    
    corr50 = ts_corr_all(round(.5*n_ts_corr_all));
    [sub, node] = find(ts_corr==corr50);
    subplot(2, 2, 3),
    plot(squeeze(X_ts(sub, node, :))), hold on,
    plot(squeeze(X_ts_enhanced(sub, node, :))), hold off
    legend('Original', 'Enhanced')
    title(['Median original/enhanced corr. (r=', num2str(round(corr50, 2)), ')'],...
        'FontSize', 8)
    xlabel('Time'), ylabel('Signal')
    
    corr75 = ts_corr_all(round(.75*n_ts_corr_all));
    [sub, node] = find(ts_corr==corr75);
    subplot(2, 2, 4),
    plot(squeeze(X_ts(sub, node, :))), hold on,
    plot(squeeze(X_ts_enhanced(sub, node, :))), hold off
    legend('Original', 'Enhanced')
    title(['75th % original/enhanced corr. (r=', num2str(round(corr75, 2)), ')'],...
        'FontSize', 8)
    xlabel('Time'), ylabel('Signal')
end

