function x_moving_average = MovingAverageFiltering(x, n, Demand_Plot)
x_moving_average = x;
for i = n+1:size(x, 1)-n
    x_moving_average(i) = mean(x(i-n:i+n));
end

if Demand_Plot == 1
    figure;
    hold on
    plot(x)
    plot(x_moving_average, 'r')
end
