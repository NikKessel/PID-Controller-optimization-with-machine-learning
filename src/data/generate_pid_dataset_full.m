% === Configuration ===
num_samples = 100;          % Total number of systems
T_final = 50;               % Simulation time
results = {};               % To store metrics
all_t = {}; all_y = {};     % To store responses for plotting
row = 1;                    % Row index for valid data only

for i = 1:num_samples
    try
        % === Randomly choose system type ===
        system_type = randi([1, 2]);  % 1 = PT1, 2 = PT2

        % === Generate system ===
        K  = rand() * 10 + 0.5;      % Gain K ∈ [0.5, 10]

        if system_type == 1
            % --- PT1 system ---
            T1 = rand() * 49 + 1;
            T2 = 0;  % dummy for PT1
            num = K;
            den = conv([T1 1], [1]);
            sys_type_str = "PT1";
        else
            % --- PT2 system ---
            T1 = rand() * 49 + 1;
            T2 = rand() * 49 + 1;
            num = K;
            den = conv([T1 1], [T2 1]);
            sys_type_str = "PT2";
        end

        G = tf(num, den);

        % === Randomize PID tuning options ===
        PM_options = [30, 45, 60, 75];
        focus_options = {'reference-tracking', 'disturbance-rejection'};
        PM = PM_options(randi(length(PM_options)));
        focus = focus_options{randi(2)};
        log_bw = rand() * (log10(5) - log10(0.5)) + log10(0.5);
        bw = 10^log_bw;

        opts = pidtuneOptions('DesignFocus', focus, 'PhaseMargin', PM);
        C = pidtune(G, 'PID', bw, opts);

        % === Check if PID gains are valid ===
        if any([C.Kp, C.Ki, C.Kd] <= 0) || any(isnan([C.Kp, C.Ki, C.Kd]))
            warning("Skipping invalid PID gains (zero or NaN)");
            continue;
        end

        Kp = C.Kp;
        Ki = C.Ki;
        Kd = C.Kd;

        % === Simulate closed-loop system ===
        sys_cl = feedback(C * G, 1);
        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        info = stepinfo(y, t);
        e = 1 - y;
        ISE = trapz(t, e.^2);
        SSE = abs(e(end));

        % === Store results ===
        results{row,1}  = K;
        results{row,2}  = T1;
        results{row,3}  = T2;
        results{row,4}  = Kp;
        results{row,5}  = Ki;
        results{row,6}  = Kd;
        results{row,7}  = ISE;
        results{row,8}  = SSE;
        results{row,9}  = info.RiseTime;
        results{row,10} = info.SettlingTime;
        results{row,11} = info.Overshoot;
        results{row,12} = PM;
        results{row,13} = focus;
        results{row,14} = bw;
        results{row,15} = sys_type_str;

        all_y{row} = y;
        all_t{row} = t;

        % Show status
        fprintf("✅ Sample %d → Row %d: Kp=%.3f, Ki=%.4f, Kd=%.3f, ISE=%.2f\n", ...
                i, row, Kp, Ki, Kd, ISE);
        row = row + 1;

    catch ME
        disp(['❌ Sample ', num2str(i), ' failed: ', ME.message]);
        continue;
    end
end

% === Save to table ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
           'PhaseMargin', 'DesignFocus', 'Bandwidth', 'SystemType'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_pidtune.csv');
disp('✅ Dataset saved to pid_dataset_pidtune.csv');

% === Plot all step responses ===
figure;
hold on;
for i = 1:length(all_y)
    plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.25]);
end
yline(1.0, 'k--', 'Step Input');
title('All Step Responses from PID Tuned Systems');
xlabel('Time (s)');
ylabel('Output y(t)');
grid on;
hold off;
