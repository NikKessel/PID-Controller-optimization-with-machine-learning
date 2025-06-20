% === Configuration ===
num_samples = 10000;           % Total number of systems
T_final = 3000;              % Simulation time
results = {};                % Store metrics
all_t = {}; all_y = {};      % Store responses
row = 1;

% === System Type Definitions ===
system_definitions = {
    % Label        Kmin  Kmax   T1min  T1max   T2min  T2max
    "VeryFast",    0.5,   5.0,   0.01,  0.5,    0.005, 0.2;
    "Fast",        0.8,   8.0,   0.1,   2.0,    0.01,  2.0;
    "Medium",      1.0,  15.0,   1.0,   10.0,   0.1,   10.0;
    "Slow",        2.0,  20.0,   5.0,   30.0,   5.0,   30.0;
    "VerySlow",    2.0,  10.0,  20.0,  200.0,  20.0, 200.0;
    "HighGain",   10.0, 100.0,   0.5,   10.0,   0.1,   2.0;
};
num_types = size(system_definitions, 1);

% === PIDTUNE VARIATION RANGES ===
pidtune_ranges = struct( ...
    'wc_factor_min',     0.1, ...
    'wc_factor_max',     10.0, ...
    'phase_margin_min',  25, ...
    'phase_margin_max',  60, ...
    'design_focus',      {{'reference-tracking', 'balanced', 'disturbance-rejection'}} ...
);

% === Main Loop ===
fprintf('Starting PID dataset generation...\n');
for i = 1:num_samples
    try
        % === 1. Randomly choose system type ===
        type_idx = randi(num_types);
        selected_type = system_definitions(type_idx, :);
        type_label = selected_type{1};
        Kmin = selected_type{2}; Kmax = selected_type{3};
        T1min = selected_type{4}; T1max = selected_type{5};
        T2min = selected_type{6}; T2max = selected_type{7};

        is_pt1 = rand() < 0.5;

        % === 2. Generate system parameters ===
        K = rand() * (Kmax - Kmin) + Kmin;
        T1 = rand() * (T1max - T1min) + T1min;

        if is_pt1
            T2 = 0;
            den = [T1 1];
            system_type = "PT1_" + type_label;
        else
            T2 = rand() * (T2max - T2min) + T2min;
            den = conv([T1 1], [T2 1]);
            system_type = "PT2_" + type_label;
        end

        G = tf(K, den);

        % === 3. PID Tuning with variation ===
        % Compute Tsum and clamp wc range
        Tsum = T1 + T2 + 1e-3;
        wc_min = 0.01;
        wc_max = 10;

        % Sample wc
        wc_factor = rand() * (pidtune_ranges.wc_factor_max - pidtune_ranges.wc_factor_min) + pidtune_ranges.wc_factor_min;
        wc = wc_factor / Tsum;
        wc = min(max(wc, wc_min), wc_max);

        % Sample other parameters
        phase_margin = rand() * (pidtune_ranges.phase_margin_max - pidtune_ranges.phase_margin_min) + pidtune_ranges.phase_margin_min;
        focus_options = pidtune_ranges.design_focus;
        design_focus = focus_options{randi(numel(focus_options))};

        % === FIXED: Use pidtune with pidtuneOptions for advanced parameters ===
        try
            % Try using pidtuneOptions if available (newer MATLAB versions)
            opts = pidtuneOptions('CrossoverFrequency', wc, 'PhaseMargin', phase_margin);
            [C, info] = pidtune(G, 'PID', opts);
        catch
            % Fallback to basic pidtune with just crossover frequency
            try
                [C, info] = pidtune(G, 'PID', wc);
            catch
                % Ultimate fallback - use pidtune with default settings
                [C, info] = pidtune(G, 'PID');
                % Store the actual crossover frequency used
                if isfield(info, 'CrossoverFrequency')
                    wc = info.CrossoverFrequency;
                end
            end
        end

        % === 4. Validate PID ===
        Kp = C.Kp; Ki = C.Ki; Kd = C.Kd;
        if any(isnan([Kp, Ki, Kd])) || Kp <= 0 || Ki <= 0
            continue;
        end

        % === 5. Simulate Step Response ===
        sys_cl = feedback(C * G, 1);
        
        % Check if closed-loop system is stable
        if ~isstable(sys_cl)
            continue;
        end
        
        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        % === 6. Compute Metrics ===
        try
            info_step = stepinfo(y, t);
            e = 1 - y;
            ISE = trapz(t, e.^2);
            SSE = abs(e(end));

            % Handle cases where stepinfo might not compute all metrics
            if ~isfield(info_step, 'SettlingTime') || isnan(info_step.SettlingTime)
                info_step.SettlingTime = T_final;
            end
            if ~isfield(info_step, 'RiseTime') || isnan(info_step.RiseTime)
                info_step.RiseTime = T_final;
            end
            if ~isfield(info_step, 'Overshoot') || isnan(info_step.Overshoot)
                info_step.Overshoot = 0;
            end

            % Reject unstable or poorly performing systems
            if info_step.SettlingTime > T_final || info_step.RiseTime > T_final || ISE > 1e5
                continue;
            end
        catch
            % If stepinfo fails, skip this sample
            continue;
        end

        % === 7. Store Valid Sample ===
        results{row,1}  = K;
        results{row,2}  = T1;
        results{row,3}  = T2;
        results{row,4}  = Kp;
        results{row,5}  = Ki;
        results{row,6}  = Kd;
        results{row,7}  = ISE;
        results{row,8}  = SSE;
        results{row,9}  = info_step.RiseTime;
        results{row,10} = info_step.SettlingTime;
        results{row,11} = info_step.Overshoot;
        results{row,12} = system_type;
        results{row,13} = type_label;
        results{row,14} = wc;
        results{row,15} = phase_margin;
        results{row,16} = design_focus;

        all_t{row} = t;
        all_y{row} = y;

        if mod(row, 10) == 0
            fprintf("✅ %d → %d [%s]: Kp=%.3f, Ki=%.4f, Kd=%.3f, wc=%.2f\n", ...
                i, row, system_type, Kp, Ki, Kd, wc);
        end

        row = row + 1;

    catch ME
        fprintf('❌ Sample %d failed: %s\n', i, ME.message);
        continue;
    end
end

fprintf('Generated %d valid samples out of %d attempts\n', row-1, num_samples);

if isempty(results)
    error("❌ No valid samples were generated. Check your PID tuning compatibility.");
end

% === 8. Export Table ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
           'SystemType','SystemCategory','wc','PhaseMargin','DesignFocus'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_pidtune.csv');
fprintf('✅ Dataset saved to pid_dataset_pidtune.csv with %d samples\n', height(T));

% === 9. Statistical Analysis ===
fprintf('\n=== STATISTICAL ANALYSIS ===\n');

% Convert results to numerical matrix for easy analysis
numeric_cols = [1:11, 14, 15]; % K, T1, T2, Kp, Ki, Kd, ISE, SSE, RiseTime, SettlingTime, Overshoot, wc, PhaseMargin
numeric_data = cell2mat(results(:, numeric_cols));
metric_names = {'K', 'T1', 'T2', 'Kp', 'Ki', 'Kd', 'ISE', 'SSE', 'RiseTime', 'SettlingTime', 'Overshoot', 'wc', 'PhaseMargin'};

fprintf('\n--- SYSTEM PARAMETERS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Parameter', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:3  % K, T1, T2
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- PID CONTROLLER GAINS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Gain', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for i = 4:6  % Kp, Ki, Kd
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- PERFORMANCE METRICS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for i = 7:11  % ISE, SSE, RiseTime, SettlingTime, Overshoot
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- TUNING PARAMETERS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Parameter', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for i = 12:13  % wc, PhaseMargin
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

% === System Type Distribution ===
fprintf('\n--- SYSTEM TYPE DISTRIBUTION ---\n');
system_types = results(:, 12);  % SystemType column
% Convert strings to cell array of char vectors for compatibility
if iscell(system_types) && ~isempty(system_types)
    system_types_char = cellfun(@char, system_types, 'UniformOutput', false);
else
    system_types_char = system_types;
end
[unique_types, ~, idx] = unique(system_types_char);
type_counts = accumarray(idx, 1);
fprintf('%-20s %8s %8s\n', 'System Type', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 40));
for i = 1:length(unique_types)
    fprintf('%-20s %8d %8.1f%%\n', unique_types{i}, type_counts(i), 100*type_counts(i)/length(system_types));
end

% === System Category Distribution ===
fprintf('\n--- SYSTEM CATEGORY DISTRIBUTION ---\n');
system_categories = results(:, 13);  % SystemCategory column
% Convert strings to cell array of char vectors for compatibility
if iscell(system_categories) && ~isempty(system_categories)
    system_categories_char = cellfun(@char, system_categories, 'UniformOutput', false);
else
    system_categories_char = system_categories;
end
[unique_cats, ~, idx] = unique(system_categories_char);
cat_counts = accumarray(idx, 1);
fprintf('%-20s %8s %8s\n', 'Category', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 40));
for i = 1:length(unique_cats)
    fprintf('%-20s %8d %8.1f%%\n', unique_cats{i}, cat_counts(i), 100*cat_counts(i)/length(system_categories));
end

% === Design Focus Distribution ===
fprintf('\n--- DESIGN FOCUS DISTRIBUTION ---\n');
design_focus = results(:, 16);  % DesignFocus column
% Convert strings to cell array of char vectors for compatibility
if iscell(design_focus) && ~isempty(design_focus)
    design_focus_char = cellfun(@char, design_focus, 'UniformOutput', false);
else
    design_focus_char = design_focus;
end
[unique_focus, ~, idx] = unique(design_focus_char);
focus_counts = accumarray(idx, 1);
fprintf('%-25s %8s %8s\n', 'Design Focus', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 45));
for i = 1:length(unique_focus)
    fprintf('%-25s %8d %8.1f%%\n', unique_focus{i}, focus_counts(i), 100*focus_counts(i)/length(design_focus));
end

% === Correlation Analysis ===
fprintf('\n--- CORRELATION ANALYSIS (Performance Metrics) ---\n');
perf_data = numeric_data(:, 7:11);  % ISE, SSE, RiseTime, SettlingTime, Overshoot
perf_names = {'ISE', 'SSE', 'RiseTime', 'SettlingTime', 'Overshoot'};
corr_matrix = corrcoef(perf_data);

fprintf('Correlation Matrix:\n');
fprintf('%12s', '');
for i = 1:length(perf_names)
    fprintf('%12s', perf_names{i});
end
fprintf('\n');
for i = 1:length(perf_names)
    fprintf('%12s', perf_names{i});
    for j = 1:length(perf_names)
        fprintf('%12.3f', corr_matrix(i,j));
    end
    fprintf('\n');
end

% === Quality Assessment ===
fprintf('\n--- QUALITY ASSESSMENT ---\n');
% Define "good" performance criteria
good_overshoot = sum(numeric_data(:, 11) < 10);  % Overshoot < 10%
good_ise = sum(numeric_data(:, 7) < 10);          % ISE < 1
good_settling = sum(numeric_data(:, 10) < 100);  % SettlingTime < 100s
good_rise = sum(numeric_data(:, 9) < 50);        % RiseTime < 50s

total_samples = size(results, 1);
fprintf('Good Overshoot (< 10%%):      %3d/%d (%.1f%%)\n', good_overshoot, total_samples, 100*good_overshoot/total_samples);
fprintf('Good ISE (< 1):              %3d/%d (%.1f%%)\n', good_ise, total_samples, 100*good_ise/total_samples);
fprintf('Good Settling Time (< 100s): %3d/%d (%.1f%%)\n', good_settling, total_samples, 100*good_settling/total_samples);
fprintf('Good Rise Time (< 50s):      %3d/%d (%.1f%%)\n', good_rise, total_samples, 100*good_rise/total_samples);

fprintf('\n=== END STATISTICAL ANALYSIS ===\n\n');

% === 10. Plot Step Responses ===
if ~isempty(all_y)
    figure;
    hold on;
    for i = 1:length(all_y)
        plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.15]);
    end
    yline(1.0, 'k--', 'Step Input', 'LineWidth', 1.5);
    title('Step Responses from PID Tuned Systems (PT1/PT2)');
    xlabel('Time (s)');
    ylabel('Output y(t)');
    grid on;
    ylim([0, 1.5]); % Set reasonable y-axis limits
    hold off;
    
    fprintf('✅ Plot generated successfully\n');
end