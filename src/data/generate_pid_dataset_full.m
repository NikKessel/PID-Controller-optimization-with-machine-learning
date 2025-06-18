% === Configuration ===
system_definitions = {
    % Label      Kmin  Kmax  T1min  T1max  T2min  T2max
    "VeryFast",  1.0,  10.0, 0.01,  1.0,   0.005, 0.5;
    "Fast",      0.8,  8.0,  0.5,   5.0,   0.1,   2.0;
    "Medium",    2.0,  20.0, 2.0,   15.0,  0.5,   8.0;
    "Slow",      3.0,  30.0, 10.0,  40.0,  2.0,   15.0;
    "VerySlow",  5.0,  15.0, 30.0,  120.0, 5.0,   30.0;
    "HighGain",  15.0, 80.0, 1.0,   10.0,  0.2,   3.0;
};

num_per_type = 200;    % Samples per system type
T_final = 50;         % Simulation time
results = {}; all_t = {}; all_y = {};
row = 1;

failure_count = containers.Map();  % To log failures per system type

for s = 1:size(system_definitions,1)
    type_name = system_definitions{s,1};
    Kmin = system_definitions{s,2};  Kmax = system_definitions{s,3};
    T1min = system_definitions{s,4}; T1max = system_definitions{s,5};
    T2min = system_definitions{s,6}; T2max = system_definitions{s,7};

    count = 0;  % Successful samples
    attempts = 0;  % Total attempts per type
    failure_count(type_name) = 0;  % Init

    while count < num_per_type
        try
            % === Generate random system ===
            K  = rand() * (Kmax - Kmin) + Kmin;
            T1 = rand() * (T1max - T1min) + T1min;
            T2 = rand() * (T2max - T2min) + T2min;

            num = K;
            den = conv([T1 1], [T2 1]);
            G = tf(num, den);
            % === Random PID tuning options ===
            PM_options = [30, 45, 60, 75];
            focus_options = {'reference-tracking', 'disturbance-rejection'};
            PM = PM_options(randi(length(PM_options)));
            focus = focus_options{randi(2)};
            log_bw = rand() * (log10(5) - log10(0.5)) + log10(0.5);
            bw = 10^log_bw;

            C = pidtune(G, 'PID', bw, ...
                pidtuneOptions('DesignFocus', focus, 'PhaseMargin', PM));

            % === Skip invalid controllers ===
            if C.Kp == 0 || C.Ki == 0
                continue;
            end



            % === Simulate ===
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
            results{row,4}  = C.Kp;
            results{row,5}  = C.Ki;
            results{row,6}  = C.Kd;
            results{row,7}  = ISE;
            results{row,8}  = SSE;
            results{row,9}  = info.RiseTime;
            results{row,10} = info.SettlingTime;
            results{row,11} = info.Overshoot;
            results{row,12} = PM;
            results{row,13} = focus;
            results{row,14} = bw;
            results{row,15} = type_name;

            all_t{row} = t;
            all_y{row} = y;
            fprintf("✅ %s %d → Row %d: Kp=%.2f, Ki=%.2f, Kd=%.2f, ISE=%.2f\n", ...
                    type_name, i, row, C.Kp, C.Ki, C.Kd, ISE);
            row = row + 1;
            count = count + 1;


        catch ME
            disp(['❌ Sample ', num2str(i), ' failed: ', ME.message]);
            continue;
        end
    end
end

% === Save results ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot', ...
           'PhaseMargin','DesignFocus','Bandwidth','SystemCategory'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_multi_ranges.csv');
disp('✅ Dataset saved to pid_dataset_multi_ranges.csv');

% === Plot step responses ===
figure;
hold on;
for i = 1:length(all_y)
    plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.2]);
end
yline(1.0, 'k--', 'Step Input');
title('Step Responses from Multi-Range PID-Tuned Systems');
xlabel('Time (s)');
ylabel('Output y(t)');
grid on;
hold off;
