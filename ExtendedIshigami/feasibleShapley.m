function [U,DX,y,ff,phi,ffshape,phishape]=feasibleShapley(x0,x1,mdl,Torder)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Inputs
%   x0: baseline point
%   x1: counterfactual point
%   mdl: \Tilde g
%   Torder: truncation order
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%   U: design matrix
%   DX: design matrix in original space
%   yy: function evaluaiton at DX
%   ff: first-order and interactions corresponding to U
%   phi: Total-order effects
%   ffshape: shapley version ff
%   phishape: Shapley version total order effects =  Feasible Shapley
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dx=x1-x0;
n=size(x0,2);
if nargin < 4
    % fewer than 3 inputs: use n
    Torder = n;
else
    Torder = min(Torder,n);
end
m=sumbincoeffcut(n,Torder);
U=zeros(n,m);
for i=1:n
    for j=1:m
        if i==j
            U(i,j)=1;
        end
    end 
end

k=n;
for l = 2:Torder %l=2:n
    a=combnk(1:n,l);
    b=binomial(n,l);
    counter=k;
    for k=k+1:k+b;
        % if k==m-1
        %     U(:,k)=1;
        % elseif k==m
        %     U(:,k)=0;
        % else        
            for z=1:n
                for w=1:size(a,2)
                    if z==a(k-counter,w)
                    U(z,k)=1;
                    end
                end
            end
        %end
    end
end
U;
%Base case value is in the last column of U

% Creazione della matrice dU
for i=1:k+1
    for j=1:n
        dU(j,i)=dx(j)*U(j,i);
    end
end
% Creazione della matrice DX
for i=1:k+1
    for j=1:n
        DX(j,i)=dx(j)*U(j,i)+x0(j);
    end
end
DX;

y=zeros(1,m);
%Valutazione del modello esterno
for i=1:m
    %y(i)=prod(DX(:,i)');
    y(i)=mdl(DX(:,i)');
end
y;
% Ortogonalizzazione fij
i=0;
ff=zeros(1,m);
count=0;
k=0;
for l=1:Torder
    a=combnk(1:n,l);
    b=binomial(n,l);
    counter=k; 
    for i=k+1:k+b
        ff(i)=y(i);
        for u=1:k
            ep=0;
            for j=1:n
                ep=U(j,i)*U(j,u)+ep;
            end
            %ep=ep
            %i=i
            %u=u
            %pause
            if ep==sum(U(:,u))
                ind=1;
            else
                ind=0;
            end
            ff(i)=ff(i)-ind*ff(u);
        end
    ff(i)=ff(i)-y(m);
    end
    k=k+b;
end
ff;
%Importance
phi=zeros(1,n);
for j=1:n
    for i=1:m
        phi(j)=phi(j)+ff(i)*U(j,i);
    end
end
phi;

ffshape = ff; % the same with zeros(1,m) since the last one [0,0] is always zero
for i = 1:m
    if sum(U(:,i)) ~= 0
        ffshape(:,i) = ff(:,i)/sum(U(:,i));
    end
end

% 
% ffshape=np.zeros((1,m));
% for i in range(1,m+1):
%     if sum(U[:,i-1]) !=0:
%         ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) 


phishape = zeros(1,n);
for j = 1:n
    for i = 1:m
        if U(j,i)~=0
            phishape(:,j) = phishape(:,j) + ff(:,i)*U(j,i)/sum(U(:,i));
        end 
    end
end

% 
% phishape=np.zeros((1,n))
% for j in range(1,n+1):
%     for i in range(1,m+1):
%         if U[j-1,i-1] !=0:
%             phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])

