%% === Configuration ===
num_samples = 100000;
T_final = max(50);  % Simulation time
results = {};
all_t = {}; all_y = {};
row = 1;

% === DEAD TIME CONFIGURATION ===
L_ratio_min = 0.0;
L_ratio_max = 3;

% === System Type Definitions (only used for your legacy PT1/PT2 branch) ===
system_definitions = {
    "VeryLowGain",     0.5, 2.8,   1.0, 50.0,   1.0, 50.0;
    "LowGain",         0.9, 4.2,   1.0, 30.0,   1.0, 30.0;
    "MediumGain",      0.8, 5.0,   6.0, 20.0,   6.0, 20.0;
    "HighGain",        1.5, 15.0,   0.10, 8.0,    2.0, 10.0;
    "VeryHighGain",    2.5, 20.0,  0.5, 5.0,    0.5, 5.0;
};
num_types = size(system_definitions, 1);

% === SYSTEM FAMILY MIX (probabilities sum to 1) ===
family_names  = {'PT1PT2_existing','PT2_osc','IT1','P'};
family_probs  = [0.70,                0.15,     0.20, 0.1];  % tune as needed
family_cum    = cumsum(family_probs / sum(family_probs));     % for sampling

% === PARAMETER RANGES FOR NEW FAMILIES ===
ranges.PT2osc.K      = [0.3, 20];
ranges.PT2osc.w0     = [1/75, 1/1];   % rad/s (i.e., T in [1..75] s)
ranges.PT2osc.zeta   = [0.05, 1.2];   % under/over/critical

ranges.IT1.K         = [0.3, 20];
ranges.IT1.T         = [0.1, 75];

ranges.P.K           = [0.3, 20];

% === PIDTUNE VARIATION RANGES ===
pidtune_ranges = struct( ...
    'wc_factor_min',     0.20, ...
    'wc_factor_max',     8.0, ...
    'phase_margin_min',  20, ...
    'phase_margin_max',  90, ...
    'design_focus',      {{'reference-tracking', 'balanced', 'disturbance-rejection'}} ...
);

fprintf('Starting PID dataset generation...\n');

%% === Main Generation Loop ===
for i = 1:num_samples
    try
        % ---- 1. Select Random System Category (for legacy PT1/PT2) ----
        type_idx = randi(num_types);
        selected_type = system_definitions(type_idx, :);
        type_label = selected_type{1};
        Kmin = selected_type{2}; Kmax = selected_type{3};
        T1min = selected_type{4}; T1max = selected_type{5};
        T2min = selected_type{6}; T2max = selected_type{7};

        % ---- 1a. Select Random Family (new) ----
        u = rand;
        fam_idx = find(u <= family_cum, 1, 'first');
        fam = family_names{fam_idx};

        % Variables to fill for table consistency
        K = NaN; T1 = NaN; T2 = NaN; w0 = NaN; z = NaN; Tchar = NaN;
        system_type = ""; system_category = type_label; % keep your labeling
        G = [];  % transfer function

        switch fam
            case 'PT1PT2_existing'
                % Your existing PT1/PT2 generator (unchanged)
                is_pt1 = rand() < 0.5;

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
                Tchar = max(T1 + T2, 1e-6);  % characteristic time for wc

            case 'PT2_osc'
                % K*w0^2 / (s^2 + 2*z*w0*s + w0^2)
                K  = exp(log(ranges.PT2osc.K(1))  + rand()*(log(ranges.PT2osc.K(2))  - log(ranges.PT2osc.K(1))));
                w0 = exp(log(ranges.PT2osc.w0(1)) + rand()*(log(ranges.PT2osc.w0(2)) - log(ranges.PT2osc.w0(1))));
                z  = ranges.PT2osc.zeta(1)       + rand()*(ranges.PT2osc.zeta(2)     - ranges.PT2osc.zeta(1));

                num = K * w0^2;
                den = [1, 2*z*w0, w0^2];
                G = tf(num, den);

                system_type = sprintf("PT2osc_z%.2f", z);
                system_category = "PT2osc";
                Tchar = 1/max(w0, 1e-6); % natural timescale

            case 'IT1'
                % K / (s*(T s + 1)) => den = [T, 1, 0]
                K = exp(log(ranges.IT1.K(1)) + rand()*(log(ranges.IT1.K(2)) - log(ranges.IT1.K(1))));
                T = exp(log(ranges.IT1.T(1)) + rand()*(log(ranges.IT1.T(2)) - log(ranges.IT1.T(1))));
                num = K; den = [T, 1, 0];
                G = tf(num, den);

                T1 = T; T2 = 0;
                system_type = "IT1";
                system_category = "IT1";
                Tchar = T;

            case 'P'
                % Static gain, but add a small mandatory delay
                K = exp(log(ranges.P.K(1)) + rand()*(log(ranges.P.K(2))-log(ranges.P.K(1))));
                G = tf(K, 1);
            
                % Force a minimum dead-time for realism
                L = rand() * 2 + 0.5;   % e.g. between 0.5 and 2s
                [num_d, den_d] = pade(L, 1);
                G = tf(num_d, den_d) * G;
            
                system_type = "P_delay";
                system_category = "P";
                Tchar = L;   % use delay as characteristic time

        end

        % ---- 2. Dead Time (Padé) ----
        % For families without clear T1/T2 (e.g., P/PT2osc), use Tsum from legacy if available, else 0
        T_sum_for_delay = 0;
        if ~isnan(T1), T_sum_for_delay = T_sum_for_delay + T1; end
        if ~isnan(T2), T_sum_for_delay = T_sum_for_delay + T2; end

        L_ratio = rand() * (L_ratio_max - L_ratio_min) + L_ratio_min;
        L = L_ratio * T_sum_for_delay;

        if L > 0
            [num_d, den_d] = pade(L, 1);  % 1st-order Padé
            delay_tf = tf(num_d, den_d);
            G = delay_tf * G;
        end

        % ---- 3. PID Tuning with Options ----
        wc_min = 1.01;
        wc_max = 15;

        wc_factor = rand() * (pidtune_ranges.wc_factor_max - pidtune_ranges.wc_factor_min) ...
                    + pidtune_ranges.wc_factor_min;

        if ~isnan(Tchar) && isfinite(Tchar) && Tchar > 0
            wc = wc_factor / Tchar;
        else
            % Fallback if no natural time scale
            wc = exp(log(0.2) + rand()*(log(10) - log(0.2)));
        end
        wc = min(max(wc, wc_min), wc_max);

        phase_margin = rand() * (pidtune_ranges.phase_margin_max - pidtune_ranges.phase_margin_min) ...
                     + pidtune_ranges.phase_margin_min;
        focus_options = pidtune_ranges.design_focus;
        design_focus = focus_options{randi(numel(focus_options))}; %#ok<NASGU> (kept for logging/ML features)

        try
            opts = pidtuneOptions('CrossoverFrequency', wc, 'PhaseMargin', phase_margin);
            [C, info] = pidtune(G, 'PID', opts); %#ok<NASGU>
        catch
            try
                [C, info] = pidtune(G, 'PID', wc); %#ok<NASGU>
            catch
                [C, info] = pidtune(G, 'PID'); %#ok<NASGU>
            end
        end

        Kp = C.Kp; Ki = C.Ki; Kd = C.Kd;
        if any(isnan([Kp, Ki, Kd])) || Kp <= 0 || Ki <= 0
            continue;
        end

        % ---- 4. Closed-Loop Stability Check ----
        sys_cl = feedback(C * G, 1);
        if ~isstable(sys_cl)
            continue;
        end

        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        % ---- 5. Compute Metrics ----
        try
            info_step = stepinfo(y, t);
            e = 1 - y;
            ISE = trapz(t, e.^2);
            SSE = abs(e(end));

            if ~isfield(info_step, 'SettlingTime') || isnan(info_step.SettlingTime)
                info_step.SettlingTime = T_final;
            end
            if ~isfield(info_step, 'RiseTime') || isnan(info_step.RiseTime)
                info_step.RiseTime = T_final;
            end
            if ~isfield(info_step, 'Overshoot') || isnan(info_step.Overshoot)
                info_step.Overshoot = 0;
            end

            % Guards against pathological cases
            if info_step.SettlingTime > T_final || info_step.RiseTime > T_final || ISE > 1e5
                continue;
            end
        catch
            continue;
        end

        % ---- 6. Store Valid Sample ----
        results{row,1}  = K;
        results{row,2}  = T1;
        results{row,3}  = T2;
        results{row,4}  = L;
        results{row,5}  = Kp;
        results{row,6}  = Ki;
        results{row,7}  = Kd;
        results{row,8}  = ISE;
        results{row,9}  = SSE;
        results{row,10} = info_step.RiseTime;
        results{row,11} = info_step.SettlingTime;
        results{row,12} = info_step.Overshoot;
        results{row,13} = string(system_type);      % SystemType (e.g., PT2osc_z0.35, IT1, PT1_*)
        results{row,14} = string(system_category);  % SystemCategory (e.g., PT2osc, IT1, P, or legacy label)
        results{row,15} = wc;
        results{row,16} = phase_margin;
        results{row,17} = design_focus;
        results{row,18} = w0;      % natural frequency (PT2osc) else NaN
        results{row,19} = z;       % zeta (PT2osc) else NaN
        results{row,20} = Tchar;   % characteristic time for wc scaling
        results{row,21} = string(fam); % Family

        all_t{row} = t;
        all_y{row} = y;

        if mod(row, 10) == 0
            fprintf("✅ %d → %d [%s/%s]: Kp=%.3f, Ki=%.4f, Kd=%.3f, wc=%.2f\n", ...
                i, row, system_category, system_type, Kp, Ki, Kd, wc);
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

%% === 8. Export Table ===
headers = {'K','T1','T2','L','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
           'SystemType','SystemCategory','wc','PhaseMargin','DesignFocus','w0','zeta','Tchar','Family'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_pidtune_extended.csv');
fprintf('✅ Dataset saved to pid_dataset_pidtune_extended.csv with %d samples\n', height(T));

%% === 9. Statistical Analysis ===
fprintf('\n=== STATISTICAL ANALYSIS ===\n');

% Convert results to numerical matrix for easy analysis
% Columns: 1:K, 2:T1, 3:T2, 4:L, 5:Kp, 6:Ki, 7:Kd, 8:ISE, 9:SSE,
% 10:Rise, 11:Settle, 12:Overshoot, 15:wc, 16:PM, 18:w0, 19:zeta, 20:Tchar
numeric_cols = [1:12, 15, 16, 18, 19, 20];
numeric_data = cell2mat(results(:, numeric_cols));
metric_names = {'K','T1','T2','L','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
                'wc','PhaseMargin','w0','zeta','Tchar'};

fprintf('\n--- SYSTEM PARAMETERS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Parameter', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for iMet = [1,2,3,4,15,16,17]  % K, T1, T2, L, w0, zeta, Tchar
    data = numeric_data(:, iMet);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{iMet}, mean(data,'omitnan'), std(data,'omitnan'), ...
        min(data,[],'omitnan'), max(data,[],'omitnan'), median(data,'omitnan'));
end

fprintf('\n--- PID CONTROLLER GAINS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Gain', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for iMet = 5:7  % Kp, Ki, Kd
    data = numeric_data(:, iMet);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{iMet}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- PERFORMANCE METRICS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for iMet = 8:12  % ISE, SSE, RiseTime, SettlingTime, Overshoot
    data = numeric_data(:, iMet);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{iMet}, mean(data), std(data), min(data), max(data), median(data));
end

fprintf('\n--- TUNING PARAMETERS ---\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n', 'Parameter', 'Mean', 'Std', 'Min', 'Max', 'Median');
fprintf('%s\n', repmat('-', 1, 70));
for iMet = 13:14  % wc, PhaseMargin
    data = numeric_data(:, iMet);
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        metric_names{iMet}, mean(data), std(data), min(data), max(data), median(data));
end

% === System Type Distribution ===
fprintf('\n--- SYSTEM TYPE DISTRIBUTION ---\n');
system_types = results(:, 13);  % SystemType
system_types_char = cellfun(@char, system_types, 'UniformOutput', false);
[unique_types, ~, idx] = unique(system_types_char);
type_counts = accumarray(idx, 1);
fprintf('%-25s %8s %8s\n', 'System Type', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 45));
for k = 1:length(unique_types)
    fprintf('%-25s %8d %8.1f%%\n', unique_types{k}, type_counts(k), 100*type_counts(k)/length(system_types));
end

% === System Category Distribution ===
fprintf('\n--- SYSTEM CATEGORY DISTRIBUTION ---\n');
system_categories = results(:, 14);  % SystemCategory
system_categories_char = cellfun(@char, system_categories, 'UniformOutput', false);
[unique_cats, ~, idx] = unique(system_categories_char);
cat_counts = accumarray(idx, 1);
fprintf('%-20s %8s %8s\n', 'Category', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 40));
for k = 1:length(unique_cats)
    fprintf('%-20s %8d %8.1f%%\n', unique_cats{k}, cat_counts(k), 100*cat_counts(k)/length(system_categories));
end

% === Family Distribution ===
fprintf('\n--- FAMILY DISTRIBUTION ---\n');
fam_col = results(:, 21); fam_char = cellfun(@char, fam_col, 'UniformOutput', false);
[unique_fam, ~, idx] = unique(fam_char);
fam_counts = accumarray(idx, 1);
fprintf('%-20s %8s %8s\n', 'Family', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 40));
for k = 1:length(unique_fam)
    fprintf('%-20s %8d %8.1f%%\n', unique_fam{k}, fam_counts(k), 100*fam_counts(k)/length(fam_char));
end

% === Design Focus Distribution ===
fprintf('\n--- DESIGN FOCUS DISTRIBUTION ---\n');
design_focus = results(:, 17);
design_focus_char = cellfun(@char, design_focus, 'UniformOutput', false);
[unique_focus, ~, idx] = unique(design_focus_char);
focus_counts = accumarray(idx, 1);
fprintf('%-25s %8s %8s\n', 'Design Focus', 'Count', 'Percent');
fprintf('%s\n', repmat('-', 1, 45));
for k = 1:length(unique_focus)
    fprintf('%-25s %8d %8.1f%%\n', unique_focus{k}, focus_counts(k), 100*focus_counts(k)/length(design_focus));
end

% === Correlation Analysis (Performance Metrics) ===
fprintf('\n--- CORRELATION ANALYSIS (Performance Metrics) ---\n');
perf_idx = [8,9,10,11,12];  % ISE, SSE, Rise, Settle, Overshoot
perf_data = cell2mat(results(:, perf_idx));
perf_names = {'ISE', 'SSE', 'RiseTime', 'SettlingTime', 'Overshoot'};
corr_matrix = corrcoef(perf_data);

fprintf('Correlation Matrix:\n');
fprintf('%12s', '');
for k = 1:length(perf_names), fprintf('%12s', perf_names{k}); end
fprintf('\n');
for r = 1:length(perf_names)
    fprintf('%12s', perf_names{r});
    for c = 1:length(perf_names)
        fprintf('%12.3f', corr_matrix(r,c));
    end
    fprintf('\n');
end

% === Quality Assessment ===
fprintf('\n--- QUALITY ASSESSMENT ---\n');
total_samples = size(results, 1);
good_overshoot = sum(cell2mat(results(:,12)) < 10);  % Overshoot < 10%%
good_ise       = sum(cell2mat(results(:,8))  < 10);  % ISE < 10 (adjust as needed)
good_settling  = sum(cell2mat(results(:,11)) < 100);
good_rise      = sum(cell2mat(results(:,10)) < 50);

fprintf('Good Overshoot (< 10%%):      %3d/%d (%.1f%%)\n', good_overshoot, total_samples, 100*good_overshoot/total_samples);
fprintf('Good ISE (< 10):             %3d/%d (%.1f%%)\n', good_ise,       total_samples, 100*good_ise/total_samples);
fprintf('Good Settling Time (< 100s): %3d/%d (%.1f%%)\n', good_settling,  total_samples, 100*good_settling/total_samples);
fprintf('Good Rise Time (< 50s):      %3d/%d (%.1f%%)\n', good_rise,      total_samples, 100*good_rise/total_samples);

fprintf('\n=== END STATISTICAL ANALYSIS ===\n\n');

% === 10. Plot Step Responses ===
if ~isempty(all_y)
    figure;
    hold on;
    for k = 1:length(all_y)
        plot(all_t{k}, all_y{k}, 'Color', [0, 0.5, 1, 0.15]);
    end
    yline(1.0, 'k--', 'Step Input', 'LineWidth', 1.5);
    title('Step Responses from PID Tuned Systems (PT1/PT2 + PT2osc + IT1 + P)');
    xlabel('Time (s)'); ylabel('Output y(t)'); grid on;
    ylim([0, 1.5]);
    hold off;
    fprintf('✅ Plot generated successfully\n');
end
