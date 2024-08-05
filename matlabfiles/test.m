% % Load SalinasA data
% [X,Y] = extract_salinasA();
% data_name = 'SalinasA';
%
% % Extract graph and compute KDE
% G = extract_graph(X, Hyperparameters);
% p = KDE(X, Hyperparameters);
%
% % Run M-LUND
% Clusterings = M_LUND(X, Hyperparameters, G, p);
%
% % Get the best clustering (at the TotalVI minimizer index)
% best_clustering = Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx);
%
% % Visualization function
% function plot_salinas(labels, title_str)
%     M = 83;  % Height of SalinasA image
%     N = 86;  % Width of SalinasA image
%
%     labels_2d = reshape(labels, M, N);
%
%     imagesc(labels_2d);
%     colormap(lines(max(labels)));
%     title(title_str, 'Interpreter', 'latex');
%     axis equal tight off;
%     colorbar;
%     set(gca, 'FontSize', 12);
% end
%
% % Create figure
% figure('Position', [100, 100, 1000, 400]);
%
% % Plot Ground Truth
% subplot(1, 2, 1);
% plot_salinas(Y, 'SalinasA Ground Truth');
%
% % Plot M-LUND result
% subplot(1, 2, 2);
% plot_salinas(best_clustering, 'M-LUND Clustering of SalinasA');
%
% % Overall title
% sgtitle('SalinasA: Ground Truth vs M-LUND Clustering', 'Interpreter', 'latex');
%

% Load SalinasA data
[X,Y] = extract_salinasA();
data_name = 'SalinasA';

% Extract graph and compute KDE
G = extract_graph(X, Hyperparameters);
p = KDE(X, Hyperparameters);

% Run M-LUND
Clusterings = M_LUND(X, Hyperparameters, G, p);

% Get the best clustering (at the TotalVI minimizer index)
best_clustering = Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx);

% Define color scheme
colors = {'k', '#4658F8', '#2896EB', '#13BEB8', '#80CA57', '#FCBB3D', '#F8FA13'};
rgb_key = [[0,0,0]; [0.275,0.345,0.973]; [15.7,58.8, 92.2]./100; [7.5, 74.5, 72.2]./100; [50.2,79.2,34.1]./100; [98.8, 73.3, 23.9]./100; [97.3, 98,7.5]./100];

% Visualization function
function plot_salinas(labels, title_str, rgb_key)
M = 83; % Height of SalinasA image
N = 86; % Width of SalinasA image
labels_2d = reshape(labels, M, N);

% Create RGB image
rgb_image = zeros(M, N, 3);
for i = 1:M
    for j = 1:N
        if labels_2d(i,j) == 0
            rgb_image(i,j,:) = [1,1,1];  % white for background
        else
            rgb_image(i,j,:) = rgb_key(mod(labels_2d(i,j)-1, size(rgb_key,1))+1,:);
        end
    end
end

image(rgb_image);
title(title_str, 'Interpreter', 'latex');
axis equal tight off;
set(gca, 'FontSize', 12);
end

% Create figure
figure('Position', [100, 100, 1000, 400]);

% Plot Ground Truth
subplot(1, 2, 1);
plot_salinas(Y, 'SalinasA Ground Truth', rgb_key);

% Plot M-LUND result
subplot(1, 2, 2);
plot_salinas(best_clustering, ['M-LUND Clustering of SalinasA (t = ' num2str(Clusterings.TimeSamples(Clusterings.TotalVI.Minimizer_Idx)) ')'], rgb_key);

% Overall title
sgtitle('SalinasA: Ground Truth vs M-LUND Clustering', 'Interpreter', 'latex');
