% function plot_clusters(X, labels, title_str, data_name)
%     if nargin < 4
%         data_name = '';
%     end
% 
%     if strcmpi(data_name, 'SalinasA')
%         M = 83;
%         N = 86;
%         imagesc(reshape(labels, [], N));
%         colorbar;
%         title(title_str, 'Interpreter', 'latex');
%         axis equal tight;
%         set(gca, 'XTick', [], 'YTick', []);
%     else
%         if size(X, 2) == 2
%             scatter(X(:,1), X(:,2), 36, labels, 'filled');
%         elseif size(X, 2) == 3
%             scatter3(X(:,1), X(:,2), X(:,3), 36, labels, 'filled');
%             view([-4, -20, 7]);
%         else
%             error('Data should be 2D or 3D for visualization');
%         end
%         colormap(lines(max(labels)));
%         colorbar;
%         title(title_str, 'Interpreter', 'latex');
%         xlabel('Feature 1');
%         ylabel('Feature 2');
%         if size(X, 2) == 3
%             zlabel('Feature 3');
%         end
%         set(gca, 'XTick', [], 'YTick', []);
%         axis equal tight;
%     end
% 
%     set(gca, 'FontSize', 20, 'FontName', 'Times');
%     box on;
% end
% 
% function plot_clusters(X, labels, title_str)
%     scatter(X(:,1), X(:,2), 20, labels, 'filled');
%     colormap(lines(max(labels)));
%     title(title_str);
%     xlabel('Feature 1');
%     ylabel('Feature 2');
%     colorbar;
% end

% function plot_clusters(labels, title_str)
%     M = 83;  % SalinasA image height
%     N = 86;  % SalinasA image width
% 
%     % Reshape labels into 2D image
%     labels_2d = reshape(labels, M, N);
% 
%     % Plot the clustered image
%     imagesc(labels_2d);
%     colormap(lines(max(labels)));  % Use a discrete colormap
%     title(title_str, 'Interpreter', 'latex');
%     axis equal tight;
%     colorbar;
% 
%     % Remove axis ticks
%     xticks([]);
%     yticks([]);
% 
%     % Set font properties
%     set(gca, 'FontSize', 20, 'FontName', 'Times');
% end


% function plot_clusters(X, Clusterings, data_name)
% 
% strcmp(data_name, 'SalinasA')
% 
% n = length(X);
% M = 83;
% N = 86;
% 
% load('salinasA_gt');
% [~,GT_idx] = sort(reshape(salinasA_gt,M*N,1));
% V = Clusterings.Graph.EigenVecs(GT_idx,:);
% Vinv = pinv(V);
% lambda = Clusterings.Graph.EigenVals;
% 
% 
% 
% nt_K = unique(Clusterings.K(and(Clusterings.K>=2, Clusterings.K<n/2)));
% n_nt_K = length(nt_K);
% 
% ts = zeros(n_nt_K,1);
% for k = 1:n_nt_K
%     tk = find(Clusterings.K == nt_K(k), 2,'first');
%     ts(k) = tk(1);
% end
% ts = ts(end:-1:1);
% 
% if Clusterings.K(2) == 1 || Clusterings.K(2) >= n/2
%     ts = [2, ts'];
% end
% if Clusterings.K(end) == 1 || Clusterings.K(end) >= n/2
%     ts = [ ts, length(Clusterings.K)];
% end    
% 
% n_rows = length(ts);
% plt_idx = 0;    
% 
% for row = 1:n_rows
% 
%     t = ts(row);
% 
%     % Plot cluster assignments at time t
%     plt_idx = plt_idx+1;
%     subplot(n_rows, n_cols, plt_idx);
%     imagesc(reshape(Clusterings.Labels(:,t), M,N))
%     pbaspect([1,1,1])
%     title('LUND Assignments', 'interpreter', 'latex')
%     xticks([])
%     yticks([])
%     pbaspect([1,1,1])
%     set(gca,'FontSize', 20, 'FontName', 'Times')
% 
%     % Plot transition matrix at time t
%     plt_idx = plt_idx+1;
%     subplot(n_rows, n_cols, plt_idx);
%     Dt = diag(lambda.^Clusterings.TimeSamples(t));
%     Pt = V*Dt*Vinv;
%     imagesc(real(log10(Pt)))
% 
%     title('$\log_{10}[P^t]$', 'interpreter', 'latex')
%     xticks([])
%     yticks([])
%     colorbar
%     pbaspect([1,1,1])
%     set(gca,'FontSize', 20, 'FontName', 'Times')
% 
% 
%     title('$\log_{10}[P^t]$', 'interpreter', 'latex')
%     xticks([])
%     yticks([])
%     colorbar
%     pbaspect([1,1,1])
%     set(gca,'FontSize', 20, 'FontName', 'Times')
% 
% 
%     % Plot eigenvalue decay at time t
%     plt_idx = plt_idx+1;
%     subplot(n_rows, n_cols, plt_idx);
%     plot(Clusterings.Graph.EigenVals.^Clusterings.TimeSamples(t), 'LineWidth', 2)
%     title('$\lambda_k^t$', 'interpreter', 'latex')
%     xlabel('$k$', 'interpreter', 'latex')
%     xticks(1:5)
%     yticks(0:0.1:1)
%     ylim([0,1])
%     pbaspect([1,1,1])
%     set(gca,'FontSize', 20, 'FontName', 'Times')
% 
%     % Plot Dt(x) with modes
%     plt_idx = plt_idx+1;
%     subplot(n_rows, n_cols, plt_idx);
%     Dt = Clusterings.Dt(:,t);
%     X = reshape(Dt,M,N);
%     [~, m_sorting] = sort(Dt, 'descend');
%     K = Clusterings.K(t);
% 
%     modes = zeros(K,2);
%     for l = 1:K
%         [modes(l,2), modes(l,1)] = find(X == Dt(m_sorting(l)));
%     end
% 
%     imagesc(log10(X))
%     hold on
%     scatter(modes(:,1), modes(:,2), 72, 'ro', 'filled')
%     pbaspect([1,1,1])
%     title('Data with modes, colored by $\log_{10}[\mathcal{D}_t(x)]$', 'interpreter', 'latex')
%     xticks([])
%     yticks([])
%     set(gca,'FontName', 'Times', 'FontSize', 20)
%     colorbar
% 
%     if sc_on
% 
%         plt_idx = plt_idx+1;
%         subplot(n_rows, n_cols, plt_idx);
% 
%         [delta, lambda, kappa] = StochasticComplement(Clusterings.Graph.P, Clusterings.Labels(:,t));
% 
%         for i = 1:length(epsilons)
%             epsilon = epsilons(i);
%             top = log(2*kappa)-log(epsilon);
%             bottom = -log(lambda);
%             lb(i) = top/bottom;
%             ub(i) = epsilon/(2*delta);
%         end
%         title('Bounds of $\mathcal{I}_\epsilon^{(\ell)}$', 'interpreter', 'latex')
%         loglog(epsilons, lb, 'LineWidth', 2)
%         hold on
%         loglog(epsilons, ub, 'LineWidth', 2)
%         hold off
%         ylim([min([lb,ub]')/10, max([lb,ub]')*10])
%         l = ceil(log10(min([lb,ub]')));
%         u = floor(log10(max([lb,ub]')));
%         yticks(10.^(-4:2:8))
%         pbaspect([1,1,1])
%         xlim([min(epsilons) , max(epsilons)])
%         legend({'$\frac{\ln(2\kappa^{(\ell)}/\epsilon)}{\ln(1/|\lambda_{K_\ell+1}^{(\ell)}|)}$', '$\frac{\epsilon}{2\delta^{(\ell)}}$'}, 'interpreter', 'latex', 'location', 'southeast')
%         xlabel('$\epsilon$', 'interpreter', 'latex')
% 
%         xtickangle(45)
%         title('Lower and Upper Bounds of $\mathcal{I}_\epsilon^{(\ell)}$ as a function of $\epsilon$', 'interpreter', 'latex')
% 
%         set(gca,'FontSize', 20, 'FontName', 'Times')
% 
%         box on
% 
%     end
% 
% end
% end


function plot_clusters(Clusterings, title_str)
    M = 83;
    N = 86;

    % Get the best clustering (at the TotalVI minimizer index)
    best_clustering = Clusterings.Labels(:, Clusterings.TotalVI.Minimizer_Idx);

    % Plot cluster assignments
    figure;
    imagesc(reshape(best_clustering, M, N));
    colormap(lines(max(best_clustering)));
    colorbar;
    
    title(title_str, 'Interpreter', 'latex');
    xticks([]);
    yticks([]);
    pbaspect([1,1,1]);
    set(gca, 'FontSize', 20, 'FontName', 'Times');
end