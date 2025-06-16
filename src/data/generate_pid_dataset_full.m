% === Configuration ===
num_samples = 10000;         % Total number of systems
T_final = 300;             % Simulation time
results = {};              % To store metrics
all_t = {}; all_y = {};    % To store responses for plotting

for i = 1:num_samples
    try
        % === 1. Generate random PT2 system ===
        K  = rand() * 10 + 0.5;       % K ∈ [0.5, 3]
        T1 = rand() * 49 + 0.1;          % T1 ∈ [1, 50]
        T2 = rand() * 15 + 0.01;          % T2 ∈ [5, 50]

        num = [K];
        den = conv([T1 1], [T2 1]);    % (T1*s+1)(T2*s+1)
        G = tf(num, den);

        % === 2. Tune PID controller ===
        C = pidtune(G, 'PID');
        Kp = C.Kp;
        Ki = C.Ki;
        Kd = C.Kd;

        % === 3. Closed-loop simulation ===
        sys_cl = feedback(C * G, 1);
        t = linspace(0, T_final, 1000);
        [y, t] = step(sys_cl, t);

        % === 4. Performance metrics ===
        info = stepinfo(y, t);
        e = 1 - y;
        ISE = trapz(t, e.^2);
        SSE = abs(e(end));

        % === 5. Store numeric results ===
        results{i,1}  = K;
        results{i,2}  = T1;
        results{i,3}  = T2;
        results{i,4}  = Kp;
        results{i,5}  = Ki;
        results{i,6}  = Kd;
        results{i,7}  = ISE;
        results{i,8}  = SSE;
        results{i,9}  = info.RiseTime;
        results{i,10} = info.SettlingTime;
        results{i,11} = info.Overshoot;

        % === 6. Store step response ===
        all_y{i} = y;
        all_t{i} = t;

        % Optional: Display status
        if mod(i, 10) == 0
            fprintf("✅ Sample %d: Kp=%.3f, Ki=%.4f, Kd=%.3f, ISE=%.2f\n", ...
                i, Kp, Ki, Kd, ISE);
        end

    catch ME
        disp(['❌ Sample ', num2str(i), ' failed: ', ME.message]);
        continue;
    end
end

% === Save to table ===
headers = {'K','T1','T2','Kp','Ki','Kd','ISE','SSE','RiseTime','SettlingTime','Overshoot'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, 'pid_dataset_pidtune.csv');
disp('✅ Dataset saved to pid_dataset_pidtune.csv');

% === Plot all step responses ===
figure;
hold on;
for i = 1:length(all_y)
    plot(all_t{i}, all_y{i}, 'Color', [0, 0.5, 1, 0.25]); % Transparent blue
end
yline(1.0, 'k--', 'Step Input');
title('All Step Responses from PID Tuned Systems');
xlabel('Time (s)');
ylabel('Output y(t)');
grid on;
hold off;
