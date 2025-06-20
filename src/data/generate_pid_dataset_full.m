% === Configuration ===
num_samples = 100;           % Total number of systems
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
    'wc_factor_max',     2.0, ...
    'phase_margin_min',  45, ...
    'phase_margin_max',  80, ...
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

% === 9. Plot Step Responses ===
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