% Truncation based estimation experiments
% Model 1: The extended Ishigami function with interaction of order 2
% Model 2: The extended Ishigami function with product interaction of order
% 14
close all;clearvars;clc

%% ---------------- Input PARAMETERS ----------------
% ----- A model with interaction order of 2
mdl = @(X) ishigami_ext(X); sname = 'milddummy'; 
% ----- A model with higher-order interactions
%mdl = @(X) ishigami_ext_interaction(X); sname = 'intct500'; 

numRepeat = 10;

k_ext = 14;            % Max number of extra inputs (add more dimensions) 15
d = 3 + k_ext;        % total input dimensions
disp(['Total input dimensions = ',num2str(d)])

Tlist = 1:1:d; % Torder list
if Tlist(end)~=d
    Tlist(end+1) = d; % Ensure the last element is d if not already present
end

%   CORRELATED NORMAL INPUTS
mu = zeros(1,d);
Sigma = 0.5 * ones(d);
Sigma(1:d+1:end) = 1;
%   FEASIBILITY THRESHOLD (based on normal pdf)
% Multivariate normal pdf
mypdf = @(Z) mvnpdf(Z, mu, Sigma);
% Analytical threshold for 5% infeasibility
alpha = 0.05;     % fraction of infeasible points (target)
c = chi2inv(1 - alpha, d);
feasi_threshold = (2*pi)^(-d/2) * det(Sigma)^(-1/2) * exp(-0.5*c);
%% ---------------- Baseline x0 and Counterfactual x1 ----------------
% Define input vectors of length d
x1 = [2, 1, 1, 0.5*ones(1,k_ext)];          % extend as needed
x0 = [0,0,0,zeros(1,k_ext)];   % ---- change Baseline point here
%x0 = [0.1,0.1,0.1,0.1*ones(1,k_ext)];

% ---------------- MODEL DEFINITIONS ----------------
mdl_feasible = @(X) mdl(X).*(mypdf(X) >= feasi_threshold) + mdl(x0).*(mypdf(X) < feasi_threshold);
fprintf('~g(x0) is %.4f .\n', mdl(x0));
fprintf('~g(x1) is %.4f .\n', mdl(x1));

%% Experiments
% ! this section may take long, so we comment it and use the saved results for plots

% phishape_DGP = NaN(length(Tlist), d, numRepeat);   % store phishape for each repetition
% elapsed_DGP_all = NaN(length(Tlist), numRepeat);   % store run times
% 
% rng(123456);
% for r = 1:numRepeat
%     fprintf('\n============================\n');
%     fprintf('   Repetition %d / %d\n', r, numRepeat);
%     fprintf('============================\n');
% 
%     for i = 1:length(Tlist)
% 
%         Torder = min(Tlist(i), d);
%         disp(['Start calculating Shapley values with Truncation order = ', num2str(Torder), '.']);
% 
%         tic;
%         [U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl_feasible, Torder);
%         elapsed = toc; %
%         %figure;bar([phishape',ffshape(1:d)'])
% 
%         % save output
%         elapsed_DGP_all(i,r) = elapsed;
%         phishape_DGP(i,:,r) = phishape;
% 
%         fprintf('Iteration (%d,%d): Running time = %.6f seconds\n', i, r, elapsed);
%     end
% end
% 
% filename = ['Ishigami_d',num2str(d),'_',sname,'_truncation.mat'];
% save(filename);
%% load data
%d = 17; sname = 'milddummy'; % for Model 1
d = 17; sname = 'intct500'; % for Model 2
filename = ['Ishigami_d',num2str(d),'_',sname,'_truncation.mat'];
load(filename)
%% Elapsed time plot
figure;
mean_t = mean(elapsed_DGP_all, 2);
std_t  = std(elapsed_DGP_all, 0, 2);
hold on;
hShade = fill([Tlist(:); flipud(Tlist(:))], ...
              [mean_t - std_t; flipud(mean_t + std_t)], ...
              [0.8 0.85 1], ...     % light blue
              'EdgeColor', 'none', ...
              'FaceAlpha', 0.35);
hLine = plot(Tlist, mean_t, '-o', ...
             'LineWidth', 2, ...
             'MarkerSize', 6, ...
             'Color', [0 0.2 0.8]);

xlabel('M');
ylabel('Elapsed Time (seconds)');
%title('Running Time vs Tlist (Mean ± Std)');
grid on;

legend([hLine, hShade], ...
       {'Mean Runtime', '±1 Std Dev'}, ...
       'Location', 'best');

hold off;
set(gca,"FontSize",15)

%% calculate estimation error for each Tlist
TrueValues = phishape_DGP(end,:,end);
TrueMat = reshape(TrueValues, [1, d, 1]);

% Absolute error:
AbsError = abs(phishape_DGP - TrueMat);
% Relative error: 
RelError = AbsError ./ abs(TrueMat);
MeanAbsError = mean(AbsError, 3);  
MeanRelError = mean(RelError, 3);  
MAE = mean(MeanAbsError, 2);   
MRE = mean(MeanRelError, 2);  

figure;
plot(Tlist, MAE, '-o', 'LineWidth', 2);
xlabel('Truncation Order');
ylabel('Mean Absolute Error');
grid on;set(gca,"FontSize",15)

figure;
plot(Tlist, MRE, '-s', 'LineWidth', 3);
xlabel('M');
ylabel('Mean Relative Error');
%title('Mean Relative Estimation Error');
grid on;
set(gca,"FontSize",15)

figure; hold on;
for j = 1:d
    plot(Tlist, MeanAbsError(:,j), '-o', 'LineWidth', 2);
end
xlabel('Truncation Order M');
ylabel('Mean Abs Error');
grid on;
legend(arrayfun(@(j) sprintf('X%d', j), 1:d, 'UniformOutput', false), ...
       'Location', 'bestoutside');
set(gca,"FontSize",15)

%%  Barplot Feasible Shapley Analytical Values
figure;
bar(1:d,phishape_DGP(end,:,end))
xticks(1:d);
xlabel('Feature Index');
ylabel('$S_i(\widetilde{\varphi })$','Interpreter','latex')
set(gca, 'FontSize', 15);
grid on;


