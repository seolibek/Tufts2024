[X,Y] = extract_salinasA();
data_name = 'SalinasA';
load('salinasA-HP.mat')

Clusterings = M_LUND(X, Hyperparameters);

% Get the dimensions of the image
M = 83;
N = 86;
GT = reshape(Y, M, N);

% Define colors
colors = {'k', '#4658F8', '#2896EB', '#13BEB8', '#80CA57', '#FCBB3D', '#F8FA13'};
rgb_key = [[0,0,0]; [0.275,0.345,0.973]; [15.7,58.8, 92.2]./100; [7.5, 74.5, 72.2]./100; [50.2,79.2,34.1]./100; [98.8, 73.3, 23.9]./100; [97.3, 98,7.5]./100];

% Create figure
figure

% Plot Ground Truth
subplot(1,2,1)
c_data = zeros(M,N,3);
for i = 1:M
    for j = 1:N
        c_data(i,j,:) = rgb_key(GT(i,j),:);
    end
end
image(c_data)
title('Salinas A Ground Truth', 'interpreter', 'latex')
xticks([])
yticks([])
pbaspect([1,1,1])
set(gca,'FontName', 'Times', 'FontSize', 20)

% Plot M_LUND Clustering
subplot(1,2,2)
best_clustering = Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx);
clustering_image = reshape(best_clustering, M, N);
c_data_clustering = zeros(M,N,3);
for i = 1:M
    for j = 1:N
        if clustering_image(i,j) == 0
            c_data_clustering(i,j,:) = [1,1,1];  % white for background
        else
            c_data_clustering(i,j,:) = rgb_key(mod(clustering_image(i,j)-1, size(rgb_key,1))+1,:);
        end
    end
end
image(c_data_clustering)
title(['M-LUND Clustering (t = ' num2str(Clusterings.TimeSamples(Clusterings.TotalVI.Minimizer_Idx)) ')'], 'interpreter', 'latex')
xticks([])
yticks([])
pbaspect([1,1,1])
set(gca,'FontName', 'Times', 'FontSize', 20)

% Adjust the layout
set(gcf, 'Position', [100, 100, 1000, 400])

% Display additional information
disp(['Number of clusters: ' num2str(Clusterings.K(Clusterings.TotalVI.Minimizer_Idx))])
disp(['Total VI: ' num2str(Clusterings.TotalVI.Vector(Clusterings.TotalVI.Minimizer_Idx))])
