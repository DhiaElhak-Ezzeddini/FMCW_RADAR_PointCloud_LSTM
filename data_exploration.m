base_path = "data/arm_to_left/arm_to_left/";

% Initialize arrays to store means
x_means = zeros(400, 1);
y_means = zeros(400, 1);
vel_means = zeros(400, 1);
range_means = zeros(400, 1);

for i = 1:400
    % Construct file name
    file_name = sprintf("%sgesture_%d.csv", base_path, i);
    
    % Check if the file exists (optional but recommended)
    if isfile(file_name)
        data = readtable(file_name);

        % Compute means
        x_means(i) = mean(data.x);
        y_means(i) = mean(data.y);
        vel_means(i) = mean(data.Velocity);
        range_means(i) = mean(data.Range);
    else
        warning('File not found: %s', file_name);
        x_means(i) = NaN;
        y_means(i) = NaN;
        vel_means(i) = NaN;
        range_means(i) = NaN;
    end
end



%%


figure;

subplot(2,2,1);
boxplot(x_means,'Labels',{'x'})
title('Boxplot of x Mean Across Gesture Samples');
ylabel('Mean Value');
grid on ; 

subplot(2,2,2);
boxplot(y_means,'Labels',{'y'})
title('Boxplot of y Mean Across Gesture Samples');
ylabel('Mean Value');
grid on ; 

subplot(2,2,3);
boxplot(range_means,'Labels',{'Range'})
title('Boxplot of Range Mean Across Gesture Samples');
ylabel('Mean Value');
grid on ; 

subplot(2,2,4);
boxplot(vel_means,'Labels',{'Velocity'})
title('Boxplot of Velocity Mean Across Gesture Samples');
ylabel('Mean Value');
grid on ; 

