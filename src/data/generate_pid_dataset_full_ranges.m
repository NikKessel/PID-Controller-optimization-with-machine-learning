% === Configuration ===
num_samples = 1000;       % Total number of systems
T_final = 10;              % Simulation time
results = {};               % Store metrics
all_t = {}; all_y = {};     % Store responses
row = 1;

for i = 1:num_samples
    try
        % === 1. Randomly choose system type ===
        is_pt1 = rand() < 0.5;

        % === 2. Generate random system ===
        K  = rand() * 10 + 0.1;         % [0.5, 100]
        T1 = rand() * 1 + 0.001;       % [0.01, 100]

        if is_pt1
            T2 = 0;
            den = [T1 1];                 % PT1
            system_type = "PT1";
        else
            T2 = rand() * 1 + 0.001;   % [0.01, 100]
            den = conv([T1 1], [T2 1]);   % PT2
            system_type = "PT2";
        end

        G = tf(K, den);

        % === 3. PID Tuning ===
        C = pidtune(G, 'PID');
        Kp = C.Kp;
        Ki = C.Ki;
        Kd = C.Kd;

        if any(isnan([Kp, Ki, Kd])) || Kp == 0 || Ki == 0
            continue;
        end

        % === 4. Simulation ===
        sys_cl = feedback(C * G, 1);
        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        % === 5. Metrics ===
        info = stepinfo(y, t);
        e = 1 - y;
        ISE = trapz(t, e.^2);
        SSE = abs(e(end));

        % === 6. Store results ===
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
        results{row,12} = system_type;

        all_t{row} = t;
        all_y{row} = y;

        if mod(row, 100) == 0
            fprintf("✅ Sample %d → Row %d [%s]: Kp=%.3f, Ki=%.4f, Kd=%.3f, ISE=%.2f\n", ...
                i, row, system_type, Kp, Ki, Kd, ISE);
        end

        row = row + 1;

    catch ME
        disp(['❌ Sample ', num2str(i), ' failed: ', ME.message]);
        continue;
    end
end

% === 7. Export to Table and CSV ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot','SystemType'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_pidtune.csv');
disp('✅ Dataset saved to pid_dataset_pidtune.csv');

% === 8. Plot all responses ===
figure;
hold on;
for i = 1:length(all_y)
    plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.15]);
end
yline(1.0, 'k--', 'Step Input');
title('Step Responses from PID Tuned Systems (PT1/PT2)');
xlabel('Time (s)');
ylabel('Output y(t)');
grid on;
hold off;
