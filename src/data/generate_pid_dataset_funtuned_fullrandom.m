% === Configuration ===
num_samples = 100000;           % Total number of systems
T_final = 3000;              % Simulation time
results = {};                % Store metrics
all_t = {}; all_y = {};      % Store responses
row = 1;

% === System Type Definitions ===
system_definitions = {
    % Label           Kmin  Kmax   T1min  T1max   T2min  T2max
    "VeryFastt",       15,   50,   0.1,   3,     0.1,   3;
    "VeryFast",        0.5,  5.0,  0.01,  0.5,   0.01,  0.5;
    "Fast",            0.8,  8.0,  0.1,   2.0,   0.1,   2.0;
    "Medium",          1.0,  15.0, 1.0,   10.0,  1.0,   10.0;
    "Slow",            2.0,  20.0, 5.0,   30.0,  5.0,   30.0;
    "VerySlow",        2.0,  10.0, 20.0,  200.0, 20.0,  200.0;
    "UltraSlow",       0.1,  5.0,  100.0, 500.0, 100.0, 500.0;
    "HighGain",        10.0, 100.0,0.5,   10.0,  0.5,   10.0;
    "ExtremeGain",     50.0, 200.0,0.1,   5.0,   0.1,   5.0;
    "LowGain",         0.01, 0.5,  0.1,   20.0,  0.1,   20.0;
    "DeadTimeDominated",0.5, 10.0, 0.1,   10.0,  0.1,   10.0;
};

num_types = size(system_definitions, 1);

% === PID Parameter Ranges for Random Sampling ===
Kp_min = 0; Kp_max = 1000;
Ki_min = 0; Ki_max = 500;
Kd_min = 0; Kd_max = 200;

% === Main Loop ===
fprintf('Starting PID dataset generation with random PID parameters...\n');
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

        % === 3. Generate random PID parameters ===
        Kp = rand() * (Kp_max - Kp_min) + Kp_min;
        Ki = rand() * (Ki_max - Ki_min) + Ki_min;
        Kd = rand() * (Kd_max - Kd_min) + Kd_min;

        % Basic sanity check
        if any(isnan([Kp, Ki, Kd])) || Kp < 0 || Ki < 0 || Kd < 0
            continue;
        end

        % === 4. Simulate Step Response ===
        C = pid(Kp, Ki, Kd);
        sys_cl = feedback(C * G, 1);

        % Check if closed-loop system is stable
        %if ~isstable(sys_cl)
            %continue;
        %end

        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        % === 5. Compute Metrics ===
        try
            info_step = stepinfo(y, t);
            e = 1 - y;
            ISE = trapz(t, e.^2);
            SSE = abs(e(end));

            % Handle missing stepinfo fields
            if ~isfield(info_step, 'SettlingTime') || isnan(info_step.SettlingTime)
                info_step.SettlingTime = T_final;
            end
            if ~isfield(info_step, 'RiseTime') || isnan(info_step.RiseTime)
                info_step.RiseTime = T_final;
            end
            if ~isfield(info_step, 'Overshoot') || isnan(info_step.Overshoot)
                info_step.Overshoot = 0;
            end

            % Reject unstable or poor performance samples
            if info_step.SettlingTime > T_final || info_step.RiseTime > T_final || ISE > 1e5
                continue;
            end
        catch
            continue;
        end
        if isnan(ISE) || isinf(ISE)
            ISE = 1e5;  % or other large penalty
        end
        if isnan(SSE) || isinf(SSE)
            SSE = 1e5;
        end
        if isnan(info_step.RiseTime) || isinf(info_step.RiseTime)
            info_step.RiseTime = T_final;
        end
        if isnan(info_step.SettlingTime) || isinf(info_step.SettlingTime)
            info_step.SettlingTime = T_final;
        end
        if isnan(info_step.Overshoot) || isinf(info_step.Overshoot)
            info_step.Overshoot = 100; % Large overshoot penalty
        end

        % === 6. Store Valid Sample ===
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
        results{row,14} = NaN;           % No wc available (used in pidtune)
        results{row,15} = NaN;           % No phase margin (used in pidtune)
        results{row,16} = "random_pid";  % Design focus placeholder

        all_t{row} = t;
        all_y{row} = y;

        if mod(row, 10) == 0
            fprintf("✅ %d → %d [%s]: Kp=%.3f, Ki=%.4f, Kd=%.3f\n", ...
                i, row, system_type, Kp, Ki, Kd);
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

% === 7. Export Table ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
           'SystemType','SystemCategory','wc','PhaseMargin','DesignFocus'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_random_pid.csv');
fprintf('✅ Dataset saved to pid_dataset_random_pid.csv with %d samples\n', height(T));

% === 8. Statistical Analysis ===
fprintf('\n=== STATISTICAL ANALYSIS ===\n');

numeric_cols = [1:6, 7,8,9,10,11]; % System & PID params + metrics
numeric_data = cell2mat(results(:, numeric_cols));
metric_names = {'K', 'T1', 'T2', 'Kp', 'Ki', 'Kd', 'ISE', 'SSE', 'RiseTime', 'SettlingTime', 'Overshoot'};

fprintf('\n--- SYSTEM PARAMETERS & PID GAINS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Parameter', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 60));
for i = 1:6
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- PERFORMANCE METRICS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 60));
for i = 7:11
    data = numeric_data(:, i);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{i}, mean(data), std(data), min(data), max(data), median(data));
end

% === 9. Plot Step Responses ===
if ~isempty(all_y)
    figure;
    hold on;
    for i = 1:length(all_y)
        plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.15]);
    end
    yline(1.0, 'k--', 'Step Input', 'LineWidth', 1.5);
    title('Step Responses from Random PID Tuned Systems (PT1/PT2)');
    xlabel('Time (s)');
    ylabel('Output y(t)');
    grid on;
    ylim([0, 1.5]);
    hold off;

    fprintf('✅ Plot generated successfully\n');
end
