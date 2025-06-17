function results = run_pid_validation(K, T1, T2, Kp, Ki, Kd)
    % Inject variables into base workspace
    assignin('base', 'K', K);
    assignin('base', 'T1', T1);
    assignin('base', 'T2', T2);
    assignin('base', 'Kp', Kp);
    assignin('base', 'Ki', Ki);
    assignin('base', 'Kd', Kd);

    % Run Simulink model
    simOut = sim('C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_validation_model.slx', ...
                 'ReturnWorkspaceOutputs', 'on');

    try
        % Get timeseries signals from simOut
        y = simOut.get('yout').signals.values;
        t = simOut.get('yout').time;
        u = simOut.get('uout').signals.values;
        e = simOut.get('eout').signals.values;
    catch
        error("❌ Could not extract y/u/e from simOut. Check 'To Workspace' block names and that Save format = 'Structure With Time'.");
    end

    % === Compute metrics ===
    ise = trapz(t, e.^2);
    sse = abs(1 - y(end));
    os = max(y) - 1;

    try
        rise_idx_10 = find(y >= 0.1, 1);
        rise_idx_90 = find(y >= 0.9, 1);
        rise_time = t(rise_idx_90) - t(rise_idx_10);
    catch
        rise_time = NaN;
    end

    try
        settle_idx = find(abs(y - 1) <= 0.05, 1, 'last');
        settle_time = t(settle_idx);
    catch
        settle_time = NaN;
    end

    % Return results as struct
    results = struct( ...
        't', t, ...
        'y', y, ...
        'u', u, ...
        'e', e, ...
        'ISE', ise, ...
        'SSE', sse, ...
        'Overshoot', os, ...
        'RiseTime', rise_time, ...
        'SettlingTime', settle_time);
end
