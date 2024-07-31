[X,Y] = extract_salinasA();
data_name = 'SalinasA';
load('salinasA-HP.mat');

% whos


G = extract_graph(X, Hyperparameters);
p = KDE(X,Hyperparameters);

n = length(X);
T = full(ceil(log( log(Hyperparameters.Tau*min(G.StationaryDist)/2)/log(G.EigenVals(2)))/log(Hyperparameters.Beta)));
timesamples = [0, Hyperparameters.Beta.^(0:T)];

% Initialize
Ct = zeros(n,T+2);
Kt = zeros(T+2,1);
Dt = zeros(n,T+2);
for i = 1:T+2
    [Ct(:,i),Kt(i), Dt(:,i)] = LearningbyUnsupervisedNonlinearDiffusion(X, timesamples(i), G, p);
end

accuracy = sum(Y == Ct(:,end),'all')/numel(Y);
disp("Accuracy is " + accuracy);

% Plot LUND Clustering
figure;
plot_clusters(X, Ct(:,end), sprintf('LUND Clustering (Aligned Accuracy: %.2f, diffusion time: %f)', accuracy, timesamples(end)));

% Plot Ground Truth
figure;
plot_clusters(X, Y, 'Ground Truth');
 