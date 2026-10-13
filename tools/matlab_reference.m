function matlab_reference(input_file, output_file)
% Evaluate exported equations and the same fixed-step RK4 as the JAX solver.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'reference', 'matlab', 'generated'));
data = load(input_file);
P = data.P(:);
states = data.states;
count = size(states, 1);
M = zeros(count, 12, 12);
F = zeros(count, 12);
S = zeros(count, 2);
for i = 1:count
    M(i,:,:) = computeM_gen(data.times(i), states(i,:), P);
    F(i,:) = computeF_gen(data.times(i), states(i,:), P).';
    S(i,:) = computeSdot_gen(data.times(i), states(i,:), P).';
end
trajectory = zeros(data.steps + 1, 14);
trajectory(1,:) = data.x0;
for i = 1:double(data.steps)
    t = (i - 1) * data.dt;
    x = trajectory(i,:).';
    k1 = rhs(t, x, P);
    k2 = rhs(t + data.dt/2, x + data.dt*k1/2, P);
    k3 = rhs(t + data.dt/2, x + data.dt*k2/2, P);
    k4 = rhs(t + data.dt, x + data.dt*k3, P);
    trajectory(i+1,:) = (x + data.dt*(k1 + 2*k2 + 2*k3 + k4)/6).';
end
save(output_file, 'M', 'F', 'S', 'trajectory', '-v7');
end

function dx = rhs(t, x, P)
M = computeM_gen(t, x.', P);
F = computeF_gen(t, x.', P);
lambda = 1e-9 * trace(M) / size(M, 1);
solution = (M + lambda * eye(size(M))) \ F;
dx = [x(7:12); solution(1:6); computeSdot_gen(t, x.', P)];
end
