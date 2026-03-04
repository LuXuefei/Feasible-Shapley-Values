function [binom]=binomial(n,k)
binom=(factorial(n))/(factorial(k)*factorial(n-k));