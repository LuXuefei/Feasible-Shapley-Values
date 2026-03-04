function y = ishigami_ext_interaction(X_normal)
%% ---------------- MODEL DEFINITIONS ----------------

% Convert all inputs to uniform(0,1) using erfc (faster than calling normcdf)
U01 = 0.5 * erfc(-X_normal ./ sqrt(2));

% Convert uniform(0,1) to uniform(-pi, pi)
X_ish = (U01 - 0.5) * 2*pi;

% number of extra inputs and scalar a
k_ext = size(X_ish,2) - 3;
a = 500 * (k_ext > 0);

% row product of first three columns
rowProduct = prod(X_ish(:,1:3), 2);

% cache sin and sin^2
s = sin(rowProduct);
s2 = s .* s;

% product of extra columns only if present
if k_ext > 0
    prod_ext = prod(X_ish(:,4:end), 2);
else
    prod_ext = 0; % contributes nothing when a==0
end

% final Ishigami-like function
y = s .* (1 + 0.1 * rowProduct.^4) + 7 * s2 + a .* prod_ext;

end

% % % Convert all inputs to uniform(0,1)
%  U01 = normcdf(X_normal); % preserves the dependence (copula) structure,
%  % Convert uniform(0,1) to uniform(-pi, pi)
%  X_ish = (U01 - 0.5) * 2*pi;
%  k_ext = size(X_ish(:,4:end),2);
%  if k_ext ~=0
%      a = 500;%1/k_ext; % a = 0.01, 10^3
%  else
%      a=0;
%  end
%  % row product of X_ish
%  % Compute the row product of X_ish
%  rowProduct = prod(X_ish(:,1:3), 2);
%  % Define the extended Ishigami function
%  y= sin(rowProduct) .* (1 + 0.1 *  rowProduct.^4) + ...
%      7 * sin(rowProduct).^2 + a.* prod(X_ish(:,4:end),2);
