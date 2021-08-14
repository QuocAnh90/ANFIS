function [R2, RMSE, Slope, IA, R] = ComputePaperBackfillQi2019(y, y_star)
% y % experimental
% y_star% predicted


n = length(y);

A = sum((y-mean(y)).*(y_star-mean(y_star)));
B1 = sqrt(sum((y-mean(y)).^2));
B2 = sqrt(sum((y_star-mean(y_star)).^2));

R = A/B1/B2;

R2 = R^2;

obs = y;
sim = y_star;
IA = 1 - sum( (obs - sim).^2 ) ...
    / sum( ( abs(sim - mean(obs)) + abs(obs - mean(obs)) ).^2 ) ;

Slope = polyfit(y, y_star, 1);
Slope = Slope(1);

Errors=y-y_star;
MSE=mean(Errors.^2);
RMSE=sqrt(MSE);


