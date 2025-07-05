clear all ; clc ; 
tiradar = mmWaveRadar("TI AWR1642BOOST",ConfigPort = "COM5", DataPort = "COM6") ; 

installDir = matlabshared.supportpkg.getSupportPackageRoot ; 
tiradarCfgFileDir = fullfile(installDir,'toolbox','target', 'supportpackages', 'timmwaveradar', 'configfiles');
tiradar.ConfigFile = fullfile(tiradarCfgFileDir, 'xwr16xx_BestRangeResolution_UpdateRate_10.cfg');

[dets, timestamp, meas, overrun] = tiradar() ; 

tiradar.EnableRangeGroups = true;
tiradar.EnableDopplerGroups = true;
tiradar.AzimuthLimits = [-30 30];
tiradar.DetectionCoordinates = "Sensor rectangular";

% Create figure and other graphic objects to view the detections and range profile
fig = figure('Name','Radar Data', 'WindowState','maximized','NumberTitle','off');
tiledlayout(fig,2,2);

% Create handle to scatter plot and initialize its properties for plotting detections
ax1 = nexttile;
scatterPlotHandle = scatter(ax1,0,0,'filled','yellow');
ax1.Title.String = 'Scatter plot - Object position';
ax1.XLabel.String = 'x (m)';
ax1.YLabel.String = 'y (m)';
% Update the xlimits, ylimits and nPoints as per the scenario and Radar properties
% Y-axis limits for the scatter plot
yLimits = [0,1];
% X-axis limits for the scatter plot
xLimits = [-0.3,0.3];
% Number of tick marks in x and y axis in the scatter plot
nPoints = 10;
ylim(ax1,yLimits);
yticks(ax1,linspace(yLimits(1),yLimits(2),nPoints))
xlim(ax1,xLimits);
xticks(ax1,linspace(xLimits(1),xLimits(2),nPoints));
set(ax1,'color',[0.1 0.2 0.9]);
grid(ax1,'on')

% Create text handle to print the number of detections and time stamp
ax2 = nexttile([2,1]);
blnkspaces = blanks(1);
txt = ['Number of detected objects: ','Not available',newline newline,'Timestamp: ','Not available'];
textHandle = text(ax2,0.1,0.5,txt,'Color','black','FontSize',20);
axis(ax2,'off');

% Create plot handle and initialize properties for plotting range profile
ax3 = nexttile();
rangeProfilePlotHandle = plot(ax3,0,0,'blue');
ax3.Title.String = 'Range Profile for zero Doppler';
ax3.YLabel.String = 'Relative-power (dB)';
ax3.XLabel.String = 'Range (m)';
% Update the xlimits, ylimits and nPoints as per the scenario and Radar properties
% X-axis limits for the plot
xLimits = [0,tiradar.MaximumRange];
% Y-axis limits for the plot
yLimits = [0,250];
% Number of tick marks in x axis in the Range profile plot
nPoints = 30;
ylim(ax3,yLimits);
xlim(ax3,xLimits);
xticks(ax3,linspace(xLimits(1),xLimits(2),nPoints));

% Read radar measurements in a loop and plot the measurements for 50s (specified by stopTime)
ts = tic;
stopTime = 4;
% Data Logging %%
loggedData = []; 
frameCounter = 0;

while(toc(ts)<=stopTime)

    % Read detections and other measurements from TI mmWave Radar
    [objDetsRct,timestamp,meas,overrun] = tiradar();
    frameCounter = frameCounter + 1; %%%%
    % Get the number of detections read
    numDets = numel(objDetsRct);
    % Print the timestamp and number of detections in plot
    txt = ['Number of detected objects: ', num2str(numDets),newline newline,'Timestamp: ',num2str(timestamp),'s'];
    textHandle.String = txt;
    % Detections will be empty if the output is not enabled or if no object is
    % detected. Use number of detections to check if detections are available
    if numDets ~= 0
        % Detections are reported as cell array of objects of type objectDetection
        % Extract  x-y position information from each objectDetection object
        xpos = zeros(1,numDets);
        ypos = zeros(1,numDets);
        for i = 1:numel(objDetsRct)
            xpos(i) = objDetsRct{i}.Measurement(1);
            ypos(i) = objDetsRct{i}.Measurement(2);
        end
        [scatterPlotHandle.XData,scatterPlotHandle.YData] = deal(ypos,xpos);
    end
    % Range profile will be empty if the log magnitude range output is not enabled
    % via guimonitor command in config File
    if ~isempty(meas.RangeProfile)
        [rangeProfilePlotHandle.XData,rangeProfilePlotHandle.YData] = deal(meas.RangeGrid,meas.RangeProfile);
    end
    if ~isempty(objDetsRct)
        for i = 1:numel(objDetsRct)
            det = objDetsRct{i};
            % Range, Velocity, PeakVal
            rng = norm(det.Measurement);  % Euclidean distance
            vel = det.MeasurementNoise(1,1);  % May vary depending on radar setup
            if isfield(meas, 'SNR')
                peakVal = meas.SNR(i);
            else
                peakVal = 0;  % Fallback if not available
            end
            x = det.Measurement(1);
            y = det.Measurement(2);

            % Append row to data
            loggedData = [loggedData; frameCounter, i, rng, vel, peakVal, x, y];
        end
    end
    drawnow limitrate ; 
end
% === Save to CSV ===
header = {'FrameNumber','ObjectNumber','Range','Velocity','PeakValue','x','y'};
outputFile = 'radar_point_cloud.csv';
fid = fopen(outputFile, 'w');
fprintf(fid, '%s,', header{1,1:end-1});
fprintf(fid, '%s\n', header{1,end});
fclose(fid);
dlmwrite(outputFile, loggedData, '-append');

clear tiradar;

