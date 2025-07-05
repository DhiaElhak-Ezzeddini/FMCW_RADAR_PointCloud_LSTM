% Load the CSV file
data = readtable("data/arm_to_right/arm_to_right/gesture_2.csv");
frames = unique(data.FrameNumber);
x = data.x ; 
y = data.y ; 
% Create a figure for animation
figure;  % Adjust axis limits based on your data
hold on ;  
grid on ; 
xlim([-0.4,0.4])
ylim([0,0.65])
for i = 1:length(frames)
    frame_id = frames(i);
    
    % Filter rows for this frame
    idx = data.FrameNumber == frame_id;
    x_vals = data.x(idx);
    y_vals = data.y(idx);
    
    % Plot the points
    scatter(x_vals, y_vals,'filled');
    
    % Annotate
    text(x_vals + 0.005, y_vals, string(data.ObjectNumber(idx)), 'FontSize', 8);
    
    title(sprintf('Frame %d', frame_id));
    xlabel('x');
    ylabel('y');
    
    pause(0.05);  % Pause between frames (adjust speed)
end


%%
%scatter3(data.x , data.FrameNumber , data.y , "filled" , "ColorVariable","Diastolic")%

normPower = 20 * (data.PeakValue - min(data.PeakValue)) /  (max(data.PeakValue) - min(data.PeakValue)) + 5;

% Use, for example, Range for color
figure ; 
scatter3(data.x, data.FrameNumber, data.y, ...
         normPower, ...         % size (from PeakValue)
         data.Range, ...        % color (can be changed to Velocity or PeakValue)
         'filled','h','MarkerEdgeColor','flat');

xlabel('x');
ylabel('Frame Number');
zlabel('y');
title('Arm To Right');
%colorbar;

%%

% Load the CSV file
data = readtable("data/hand_to_right/hand_to_right/gesture_1.csv");
frames = unique(data.FrameNumber);
x = data.x; 
y = data.y;

% Create a figure for animation
figure;
hold on;  % Keep all plots on the same figure
xlabel('x');
ylabel('y');
zlabel('Frame Index');
title('Cumulative Object Positions Over Frames');
grid on;

% Set axis limits based on data (optional but useful)
xlim([-0.4 , 0.4]);
ylim([0,0.6]);
zlim([1, length(frames)]);

for i = 1:length(frames)
    frame_id = frames(i);
    
    % Filter rows for this frame
    idx = data.FrameNumber == frame_id;
    x_vals = data.x(idx);
    y_vals = data.y(idx);
    
    % Plot the points, with z = frame index
    scatter3(x_vals, repmat(i, size(x_vals)),y_vals, 'filled');
    
    % Annotate
    text(x_vals + 0.005, y_vals, repmat(i, size(x_vals)), string(data.ObjectNumber(idx)), 'FontSize', 8);
    
    pause(0.3);  % Adjust speed
end





%%

data = readmatrix('data/close_fist_perpendicularly/close_fist_perpendicularly/gesture_200.csv');

% Extract relevant columns
frames = data(:,1);
x = data(:,6);
y = data(:,7);

% Create 3D scatter plot
figure;
scatter3(frames, x, y, 36, y, 'filled'); % size 36, color based on y
xlabel('Frames');
ylabel('X position [m]');
zlabel('Y position [m]');
title('Closing a fist vertically');
colormap('blues'); % optional: try different colormaps like 'parula', 'jet', etc.
grid on;
view(-30, 20); % adjust the viewing angle to match the original