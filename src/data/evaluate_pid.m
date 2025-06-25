function metrics = evaluate_pid(K, T1, T2, Kp, Ki, Kd, T_final)

    if nargin < 8
        T_final = 300;  % Default simulation time
    end

    % === 1. Define Transfer Function ===
    if T2 == 0
        den = [T1, 1];
    else
        den = conv([T1, 1], [T2, 1]);
    end
    G = tf(K, den);

    % === 2. Define PID Controller ===
    C = pid(Kp, Ki, Kd);

    % === 3. Closed-loop System ===
    sys_cl = feedback(C * G, 1);

    % === 4. Simulate Step Response ===
    t = linspace(0, T_final, 2000);
    [y, t] = step(sys_cl, t);

    % === 5. Calculate Metrics ===
    e = 1 - y;
    ISE = trapz(t, e.^2);
    SSE = abs(e(end));
    info = stepinfo(y, t);

    % Safe fallback
    if isnan(info.RiseTime), info.RiseTime = T_final; end
    if isnan(info.SettlingTime), info.SettlingTime = T_final; end
    if isnan(info.Overshoot), info.Overshoot = 100; end

    % === 6. Output ===
    metrics = struct();
    metrics.ISE = ISE;
    metrics.SSE = SSE;
    metrics.Overshoot = info.Overshoot;
    metrics.RiseTime = info.RiseTime;
    metrics.SettlingTime = info.SettlingTime;

    % === 7. Print Summary ===
    fprintf('\n=== PID Evaluation ===\n');
    fprintf('Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp, Ki, Kd);
    fprintf('ISE: %.4f\n', ISE);
    fprintf('SSE: %.4f\n', SSE);
    fprintf('Overshoot: %.2f%%\n', info.Overshoot);
    fprintf('Rise Time: %.2fs\n', info.RiseTime);
    fprintf('Settling Time: %.2fs\n', info.SettlingTime);

    % Optional: Plot response
    figure;
    plot(t, y, 'LineWidth', 1.5); hold on;
    yline(1, 'k--'); grid on;
    xlabel('Time (s)'); ylabel('Output y(t)');
    title(sprintf('Step Response (Kp=%.2f, Ki=%.2f, Kd=%.2f)', Kp, Ki, Kd));
    legend('Response', 'Reference');
end
