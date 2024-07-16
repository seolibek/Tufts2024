

% Load data
data = load('data.mat');
X = data.X;
GT = data.GT;
grid_size = ceil(sqrt(size(X, 1)));
disp(['Grid size: ' num2str(grid_size)]);

Hyperparameters = struct();
Hyperparameters.Sigma = 1.0;
Hyperparameters.DensityNN = 10;
Hyperparameters.DiffusionNN = 10;

Hyperparameters.Sigma0 = 1.0;  
% Hyperparameters.SpatialParams = struct();
% Hyperparameters.SpatialParams.GraphSpatialRadius = 12;
% Hyperparameters.SpatialParams.ImageSize = [grid_size, grid_size];

[p, ~] = KDE(X, Hyperparameters);

Graph = extract_graph(X, Hyperparameters);

t = 43;

[C, K, Dt] = LearningbyUnsupervisedNonlinearDiffusion(X, t, Graph, p);

disp(['Number of clusters: ', num2str(K)]);



function plot_clusters(X, labels, title_text, filename)
    figure;  
    scatter(X(:,1), X(:,2), 50, labels, 'filled');
    title(title_text);
    colormap('parula'); 
    colorbar;
    axis equal;
    grid on;
    saveas(gcf, filename);  
    close; 
end

plot_clusters(X, C, 'lund','lund.png');
plot_clusters(X,GT, 'ground truth scipy x data','gt.png')