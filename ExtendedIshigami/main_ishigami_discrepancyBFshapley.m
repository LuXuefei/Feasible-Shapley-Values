% What causes discripency between Baseline and Feasible Shapley
% 1. counterfactual location; 
% 2. feasibility threshold lambda
% 3. input correlation; 
close all;clearvars;clc
%% ---------------- (a) Original setting ----------------
sname = 'loc';
k_ext = 0; d = 3 + k_ext;        % total input dimensions
disp(['Total input dimensions = ',num2str(d)])

%   CORRELATED NORMAL INPUTS
mu = zeros(1,d);
Sigma = 0.5 * ones(d);
Sigma(1:d+1:end) = 1;
%   Multivariate normal pdf
mypdf = @(Z) mvnpdf(Z, mu, Sigma);

%   FEASIBILITY THRESHOLD (based on normal pdf)
% Analytical threshold for 5% infeasibility
alpha = 0.05;     % fraction of infeasible points (target)
c = chi2inv(1 - alpha, d);
feasi_threshold = (2*pi)^(-d/2) * det(Sigma)^(-1/2) * exp(-0.5*c);

% ---------------- Baseline x0 and Counterfactual x1 ----------------
x1 = [2, 1, 1, 0.5*ones(1,k_ext)];  % original setting
x0 = [0,0,0,zeros(1,k_ext)];

% check true feasibility rate
rng default
N = 10000;
X = mvnrnd(mu, Sigma, N);
% portion of X with mypdf(X) less than feasi_threshold
infeasibleCount = sum(mypdf(X) < feasi_threshold);
feasibilityRate = infeasibleCount / N;
%TrueFeasiRate(i, j) = feasibilityRate;  % Store the feasibility rate for the current parameters
fprintf('Feasibility rate for alpha = %.2f is %.4f.\n', alpha, feasibilityRate);

% define models
mdl = @(X) ishigami_ext(X);
mdl_feasible = @(X) mdl(X).*(mypdf(X) >= feasi_threshold) + mdl(x0).*(mypdf(X) < feasi_threshold);

% Calculate baseline Shapley
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl);
elapsed = toc;
Bshape = phishape;

% Calculate feasible
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl_feasible);
elapsed = toc;
Fshape = phishape;
% Count number of infeasible points
infeasibleCountFeasible = sum(mypdf(DX') < feasi_threshold);
fprintf('Infeasible points count for alpha = %.2f is %d.\n', alpha, infeasibleCountFeasible);


figure
subplot(2,1,1)
plotindx = 1;plotindy=2;
ScatterPlotFvsInF(x0,x1,X,mypdf,feasi_threshold,plotindx,plotindy)
grid on
subplot(2,1,2)
bar([Bshape; Fshape]', 'grouped');
xlabel('Feature Index');
ylabel('Shapley Values');
legend({'Baseline $S_i(\varphi)$', 'Feasible $S_i(\widetilde{\varphi })$'},'Interpreter','latex');
set(gca,"FontSize",15)
grid on;
 
%% ---------------- (b) Counterfactual Location ----------------
sname = 'loc';
k_ext = 0; d = 3 + k_ext;        % total input dimensions
disp(['Total input dimensions = ',num2str(d)])

%   CORRELATED NORMAL INPUTS
mu = zeros(1,d);
Sigma = 0.5 * ones(d);
Sigma(1:d+1:end) = 1;
%   Multivariate normal pdf
mypdf = @(Z) mvnpdf(Z, mu, Sigma);

%   FEASIBILITY THRESHOLD (based on normal pdf)
% Analytical threshold for 5% infeasibility
alpha = 0.05;     % fraction of infeasible points (target)
c = chi2inv(1 - alpha, d);
feasi_threshold = (2*pi)^(-d/2) * det(Sigma)^(-1/2) * exp(-0.5*c);

% ---------------- Baseline x0 and Counterfactual x1 ----------------
%x1 = [2, 1, 1, 0.5*ones(1,k_ext)];  % original setting
x1 = [2, 2, 2, 0.5*ones(1,k_ext)];  % with infeasible points
x0 = [0,0,0,zeros(1,k_ext)];

% check true feasibility rate
rng default
N = 10000;
X = mvnrnd(mu, Sigma, N);
% portion of X with mypdf(X) less than feasi_threshold
infeasibleCount = sum(mypdf(X) < feasi_threshold);
feasibilityRate = infeasibleCount / N;
%TrueFeasiRate(i, j) = feasibilityRate;  % Store the feasibility rate for the current parameters
fprintf('Feasibility rate for alpha = %.2f is %.4f.\n', alpha, feasibilityRate);

% define models
mdl = @(X) ishigami_ext(X);
mdl_feasible = @(X) mdl(X).*(mypdf(X) >= feasi_threshold) + mdl(x0).*(mypdf(X) < feasi_threshold);

% Calculate baseline Shapley
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl);
elapsed = toc;
Bshape = phishape;

% Calculate feasible
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl_feasible);
elapsed = toc;
Fshape = phishape;
% Count number of infeasible points
infeasibleCountFeasible = sum(mypdf(DX') < feasi_threshold);
fprintf('Infeasible points count for alpha = %.2f is %d.\n', alpha, infeasibleCountFeasible);


figure
subplot(2,1,1)
plotindx = 1;plotindy=2;
ScatterPlotFvsInF(x0,x1,X,mypdf,feasi_threshold,plotindx,plotindy)
grid on
subplot(2,1,2)
bar([Bshape; Fshape]', 'grouped');
xlabel('Feature Index');
ylabel('Shapley Values');
legend({'Baseline $S_i(\varphi)$', 'Feasible $S_i(\widetilde{\varphi })$'},'Interpreter','latex');
set(gca,"FontSize",15)
grid on;



%% ---------------- (c) Feasibility Threshold Lambda ----------------
sname = 'lambda';
k_ext = 0; d = 3 + k_ext;        % total input dimensions
disp(['Total input dimensions = ',num2str(d)])

%   CORRELATED NORMAL INPUTS
mu = zeros(1,d);
rho = 0.5; % original setting 0.5 
Sigma = rho * ones(d);
Sigma(1:d+1:end) = 1;
%   Multivariate normal pdf
mypdf = @(Z) mvnpdf(Z, mu, Sigma);

%   FEASIBILITY THRESHOLD (based on normal pdf)
% Analytical threshold for 5% infeasibility 
% alpha = 0.05;     % fraction of infeasible points (target) -- original setting
alpha = 0.15;     % fraction of infeasible points (target) -- original setting
c = chi2inv(1 - alpha, d);
feasi_threshold = (2*pi)^(-d/2) * det(Sigma)^(-1/2) * exp(-0.5*c);

% ---------------- Baseline x0 and Counterfactual x1 ----------------
x1 = [2, 1, 1, 0.5*ones(1,k_ext)];  % original setting
x0 = [0,0,0,zeros(1,k_ext)];

if mypdf(x1) < feasi_threshold
    fprintf('Counterfactual x1 is infeasible.\n');
else
    fprintf('Counterfactual x1 is feasible.\n');
end

% check true feasibility rate
rng default
N = 10000;
X = mvnrnd(mu, Sigma, N);
% portion of X with mypdf(X) less than feasi_threshold
infeasibleCount = sum(mypdf(X) < feasi_threshold);
feasibilityRate = infeasibleCount / N;
%TrueFeasiRate(i, j) = feasibilityRate;  % Store the feasibility rate for the current parameters
fprintf('Feasibility rate for alpha = %.2f is %.4f.\n', alpha, feasibilityRate);

% define models
mdl = @(X) ishigami_ext(X);
mdl_feasible = @(X) mdl(X).*(mypdf(X) >= feasi_threshold) + mdl(x0).*(mypdf(X) < feasi_threshold);

% Calculate baseline Shapley
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl);
elapsed = toc;
Bshape = phishape;

% Calculate feasible
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl_feasible);
elapsed = toc;
Fshape = phishape;
% Count number of infeasible points
infeasibleCountFeasible = sum(mypdf(DX') < feasi_threshold);
fprintf('Infeasible points count for alpha = %.2f is %d.\n', alpha, infeasibleCountFeasible);


figure
subplot(2,1,1)
plotindx = 1;plotindy=2;
ScatterPlotFvsInF(x0,x1,X,mypdf,feasi_threshold,plotindx,plotindy)
grid on
subplot(2,1,2)
bar([Bshape; Fshape]', 'grouped');
xlabel('Feature Index');
ylabel('Shapley Values');
legend({'Baseline $S_i(\varphi)$', 'Feasible $S_i(\widetilde{\varphi })$'},'Interpreter','latex');
set(gca,"FontSize",15)
grid on;


%% ---------------- (d) Correlation ----------------
sname = 'corr';
k_ext = 0; d = 3 + k_ext;        % total input dimensions
disp(['Total input dimensions = ',num2str(d)])

%   CORRELATED NORMAL INPUTS
mu = zeros(1,d);
%rho = 0.5; % original setting 0.5 
rho = 0.8; % rho  = 0.9 then CF becomes infeasible
Sigma = rho * ones(d);
Sigma(1:d+1:end) = 1;
%   Multivariate normal pdf
mypdf = @(Z) mvnpdf(Z, mu, Sigma);

%   FEASIBILITY THRESHOLD (based on normal pdf)
% Analytical threshold for 5% infeasibility
alpha = 0.05;     % fraction of infeasible points (target)
c = chi2inv(1 - alpha, d);
feasi_threshold = (2*pi)^(-d/2) * det(Sigma)^(-1/2) * exp(-0.5*c);

% ---------------- Baseline x0 and Counterfactual x1 ----------------
x1 = [2, 1, 1, 0.5*ones(1,k_ext)];  % original setting
x0 = [0,0,0,zeros(1,k_ext)];

if mypdf(x1) < feasi_threshold
    fprintf('Counterfactual x1 is infeasible.\n');
else
    fprintf('Counterfactual x1 is feasible.\n');
end

% check true feasibility rate
rng default
N = 10000;
X = mvnrnd(mu, Sigma, N);
% portion of X with mypdf(X) less than feasi_threshold
infeasibleCount = sum(mypdf(X) < feasi_threshold);
feasibilityRate = infeasibleCount / N;
%TrueFeasiRate(i, j) = feasibilityRate;  % Store the feasibility rate for the current parameters
fprintf('Feasibility rate for alpha = %.2f is %.4f.\n', alpha, feasibilityRate);

% define models
mdl = @(X) ishigami_ext(X);
mdl_feasible = @(X) mdl(X).*(mypdf(X) >= feasi_threshold) + mdl(x0).*(mypdf(X) < feasi_threshold);

% Calculate baseline Shapley
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl);
elapsed = toc;
Bshape = phishape;

% Calculate feasible
tic;
[U,DX,y,ff,phi,ffshape,phishape] = feasibleShapley(x0, x1, mdl_feasible);
elapsed = toc;
Fshape = phishape;
% Count number of infeasible points
infeasibleCountFeasible = sum(mypdf(DX') < feasi_threshold);
fprintf('Infeasible points count for alpha = %.2f is %d.\n', alpha, infeasibleCountFeasible);


figure
subplot(2,1,1)
plotindx = 1;plotindy=2;
ScatterPlotFvsInF(x0,x1,X,mypdf,feasi_threshold,plotindx,plotindy)
grid on
subplot(2,1,2)
bar([Bshape; Fshape]', 'grouped');
xlabel('Feature Index');
ylabel('Shapley Values');
legend({'Baseline $S_i(\varphi)$', 'Feasible $S_i(\widetilde{\varphi })$'},'Interpreter','latex');
set(gca,"FontSize",15)
grid on;
