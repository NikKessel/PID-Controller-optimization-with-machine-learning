% === simulate_and_export.m ===
% Loads controller & plant params from 'input.mat', runs Simulink model,
% saves results & performance metrics to 'results.mat'

try
    % Load parameters from input file
    input = load('input.mat');
    K  = input.K;
    T1 = input.T1;
    T2 = input.T2;
    Kp = input.Kp;
    Ki = input.Ki;
    Kd = input.Kd;

    % Assign to base workspace
    assignin('base', 'K', K);
    assignin('base', 'T1', T1);
    assignin('base', 'T2', T2);
    assignin('base', 'Kp', Kp);
    assignin('base', 'Ki', Ki);
    assignin('base', 'Kd', Kd);

    % Simulate model
    simOut = sim('C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_validation_model.slx', ...
             'ReturnWorkspaceOutputs', 'on');


    % Try fetching logged signals
    % New — ✅ correct way to extract logged signals
    y = simOut.get('yout').signals.values;
    u = simOut.get('uout').signals.values;
    e = simOut.get('eout').signals.values;
    t = simOut.get('yout').time;

    % === Performance metrics ===
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

    % === Save results ===
    save('C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\debug_signals.mat', ...
     't', 'y', 'u', 'e', 'K', 'T1', 'T2', 'Kp', 'Ki', 'Kd');

    disp("✅ Saved debug_signals.mat");

    save('results.mat', 't', 'y', 'u', 'e', ...
         'ise', 'sse', 'os', 'rise_time', 'settle_time');

catch ME
    % Save error for debugging
    fid = fopen('error_log.txt', 'w');
    fprintf(fid, 'Error: %s\n', ME.message);
    fclose(fid);
    rethrow(ME);
end
