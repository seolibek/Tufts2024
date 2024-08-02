[X,Y] = extract_salinasA();
data_name = 'SalinasA';
load('salinasA-HP.mat');

% G = extract_graph(X, Hyperparameters);
% p = KDE(X,Hyperparameters);

Clusterings = M_LUND(X, Hyperparameters);

accuracy = calculate_aligned_accuracy(Y, Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx));
% Get the best clustering (at the TotalVI minimizer index)
best_clustering = Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx);

% Calculate accuracy
accuracy_two = calculate_aligned_accuracy(Y, best_clustering);
% Create a new figure
% fig = figure;

plot_clusters(Clusterings, 'M-LUND Assignments for SalinasA');

% Plot M-LUND Clustering result
% subplot(1, 2, 1);
% plot_clusters(best_clustering, sprintf('M-LUND Assignments\nAccuracy: %.2f%%', accuracy * 100));

% Plot Ground Truth
% subplot(1, 2, 2);
% plot_clusters(X, Y, 'Ground Truth');
% 
% % Adjust the overall figure
% sgtitle('M-LUND Clustering vs Ground Truth', 'Interpreter', 'latex');
% set(gcf, 'Position', get(0, 'Screensize')); % Make figure full screen
% 
% 





% 
% % Plot LUND Clustering
% figure;
% % plot_clusters(X, Ct(:,end), sprintf('LUND Clustering (Aligned Accuracy: %.2f, diffusion time: %f)', accuracy, timesamples(end)));
% plot_clusters(X, Clusterings, sprintf('LUND Clustering\nAccuracy: %.2f%%', accuracy * 100));
% 
% % Plot Ground Truth
% figure;
% % plot_clusters(X, Y, 'Ground Truth');
% plot_clusters(X, Y, 'Ground Truth');
% 



% n = length(X);
% T = full(ceil(log( log(Hyperparameters.Tau*min(G.StationaryDist)/2)/log(G.EigenVals(2)))/log(Hyperparameters.Beta)));
% timesamples = [0, Hyperparameters.Beta.^(0:T)];
% 
% % Initialize
% Ct = zeros(n,T+2);
% Kt = zeros(T+2,1);
% Dt = zeros(n,T+2);
% for i = 1:T+2
%     [Ct(:,i),Kt(i), Dt(:,i)] = LearningbyUnsupervisedNonlinearDiffusion(X, timesamples(i), G, p);
% end
% 
% accuracy = sum(Y == Ct(:,end),'all')/numel(Y);
% accuracy = accuracy * 100;
% disp("Accuracy is " + accuracy);
