function y = ishigami_ext(X_normal)
% Convert all inputs to uniform(0,1) (faster than calling normcdf)
U01 = 0.5 * erfc(-X_normal ./ sqrt(2));

% Map to uniform(-pi,pi)
X_ish = (U01 - 0.5) * (2*pi);

% number of extra columns (>=0) and scalar a
nCols = size(X_ish,2);
k_ext = max(0, nCols - 3);
a = 0.01 * (k_ext > 0); % scalar

% compute main terms
s1 = sin(X_ish(:,1));           % sin(x1)
s2 = sin(X_ish(:,2));           % sin(x2)
x3_4 = X_ish(:,3).^4;           % x3^4

% compute extra-sum only if present
if k_ext > 0
    extras = sum(X_ish(:,4:end), 2);
else
    extras = 0;                 % scalar 0 broadcasts correctly
end

% final output
y = s1 .* (1 + 0.1 * x3_4) + 7 * (s2.^2) + a .* extras;
end

% % % Convert all inputs to uniform(0,1)
% U01 = normcdf(X_normal); % preserves the dependence (copula) structure,
% % Convert uniform(0,1) to uniform(-pi, pi)
% X_ish = (U01 - 0.5) * 2*pi;
% k_ext = size(X_ish(:,4:end),2);
% if k_ext ~=0
% a = 0.01;%1/k_ext; % a = 0
% else
% a=0;
% end
% 
% % Define the extended Ishigami function
% y= sin(X_ish(:,1)) .* (1 + 0.1 * X_ish(:,3).^4) + ...
% 7 * (sin(X_ish(:,2))).^2 + a.* sum(X_ish(:,4:end),2);