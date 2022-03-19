function [X_enhanced, y_predict, q_sq, r, edge_corr] = connectome_enhancement(X, y, varargin)
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
addRequired(p,'X',@isnumeric);
addRequired(p,'y',@isnumeric);
addParameter(p,'per_enhance',0.1,@isnumeric);
addParameter(p,'scale',0.1,@isnumeric);
addParameter(p,'train_model',1, @isnumeric);
addParameter(p,'per_model',0.1,@isnumeric);
addParameter(p,'show_results',0.1,@isnumeric);
addParameter(p,'seed',1,@isnumeric);
addParameter(p,'v_alpha',1e-6,@isnumeric);
addParameter(p,'lambda',NaN,@isnumeric);
addParameter(p,'kfolds',10,@isnumeric);

parse(p, X, y, varargin{:});

per_enhance = p.Results.per_enhance;
scale_factor = p.Results.scale;
train_model = p.Results.train_model;
per_model = p.Results.per_model;
show_results = p.Results.show_results;
seed = p.Results.seed;
v_alpha = p.Results.v_alpha;
lambda = p.Results.lambda;
kfolds = p.Results.kfolds;

%% initialize some parameters and run enhancement
nsub = size(X, 3);
nnodes = size(X, 1);
nedges = (nnodes*(nnodes-1))/2;

% alter data, running over different scales
% scale 0 is original pred
rng(seed)
% noise is proportional to each individuals iq score
sub_scale_noise = zscore(y)*scale_factor;
sub_scale_noise = repmat(sub_scale_noise', [nedges, 1]);

% random sign flips/edge selections
% NOTE: you can make these edges not random to enforce a specific
% neuroscientific interpretation (see pre-print)
random_edges_to_change = rand(size(sub_scale_noise, 1), 1) - .5;
% pick random 20% edges to alter
thresh_tmp = (1-per_enhance)/2;
random_edges_to_change(random_edges_to_change>=thresh_tmp) = 1;
random_edges_to_change(random_edges_to_change<=-thresh_tmp) = -1;
random_edges_to_change(abs(random_edges_to_change)<.4) = 0;

% add or subtract noise for the 20% of edges
sub_scale_noise(random_edges_to_change==1, :) = sub_scale_noise(random_edges_to_change==1, :);
sub_scale_noise(random_edges_to_change==-1, :) = -1*sub_scale_noise(random_edges_to_change==-1, :);
sub_scale_noise(random_edges_to_change==0, :) = 0;

noise_mat = zeros(size(X));
for noise_idx = 1:nsub
    noise_mat(:, :, noise_idx) = edge2mat(sub_scale_noise(:, noise_idx));
end

X_enhanced = X + noise_mat;

if train_model
    [q_sq, r, ~, y_predict, ~, ~] = ridgeCPM(X_enhanced, y, 'per_feat',...
        per_model, 'v_alpha', v_alpha, 'kfolds', kfolds, 'seed', seed, 'lambda', lambda);
end

% find edge correlation between original and enhanced connectomes
edges_original=mat2edge(X);
edges_enhanced=mat2edge(X_enhanced);
edge_corr = (diag(corr(edges_original, edges_enhanced)));

if show_results  % plot matrices
    figure;
    nplot = 3;
    plt_sub = randi(nsub, [nplot, 1]);
    for plt_idx = 1:nplot
        subplot(nplot, 2, 2*(plt_idx-1)+1),
        imagesc(edge2mat(mat2edge(X(:, :, plt_sub(plt_idx)))));  % remove diagonals
        if plt_idx==1
            title('Original')
        end
        
        subplot(nplot, 2, 2*(plt_idx)),
        imagesc(edge2mat(mat2edge(X_enhanced(:, :, plt_sub(plt_idx)))));  % remove diagonals
        if plt_idx==1
            title('Enhanced')
        end
    end
end

