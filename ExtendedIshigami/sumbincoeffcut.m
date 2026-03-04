function [sum_n_over_k]=sumbincoeffcut(n,Torder)
sum_n_over_k=0;
for i=0:Torder
sum_n_over_k=(factorial(n))/(factorial(i)*factorial(n-i))+sum_n_over_k;
end
sum_n_over_k;
