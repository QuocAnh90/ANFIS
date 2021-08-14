function r_n = autocorrelation_divide_n(v1, v2)
% Pour avoir un signal centre zero
% et correlation normalisee
Z1 = (v1 - mean(v1))/std(v1);
Z2 = (v2 - mean(v2))/std(v2);
n = length(v1);
r_n = zeros(1);
for k = 1:n-1
    A = Z1(1:n-k);
    B = Z2(1+k:n);
    r_n(k) = sum(A.*B)/n;
end

