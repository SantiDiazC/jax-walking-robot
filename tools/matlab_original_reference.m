function matlab_original_reference(input_file, output_file)
% Independently derive equations from slices of the unmodified thesis script.
% No JAX algebra or simplified MATLAB exports are used in this reference.
root = fileparts(fileparts(mfilename('fullpath')));
archive = fullfile(root, 'reference', 'matlab', 'original');
addpath(fullfile(archive, 'helpers'));
source = fileread(fullfile(archive, 'RobotDynamics_mulbo_legdirstiff_stribeck_karnopp_iCVD_dam_thick.m'));
fprintf('Deriving original symbolic balances and numerical setup...\n');
eval(slice(source, 'syms t % time', '%% III: Simulation'));
ii = 1; jj = 1;
eval(slice(source, 'params1 = ', '% Functions'));
eval(slice(source, '% % % % ground stiffness', 'fF1x{ii,jj}'));
eval(slice(source, 'tau1_ = ', '% Numerical dynamics and simulation'));
folder = tempname; mkdir(folder);
cleanup = onCleanup(@()rmdir(folder, 's'));
addpath(folder);
matlabFunction(subs(M,params,params_), 'File', fullfile(folder,'original_mass'), 'Vars', {t,X});
matlabFunction(subs(F,params,params_), 'File', fullfile(folder,'original_force'), 'Vars', {t,X});
matlabFunction([sdot1_;sdot2_], 'File', fullfile(folder,'original_memory'), 'Vars', {t,X});
data = load(input_file);
count = size(data.states,1);
mass = zeros(count,12,12); force = zeros(count,12); memory = zeros(count,2);
for k=1:count
    mass(k,:,:) = original_mass(data.times(k), data.states(k,:));
    force(k,:) = original_force(data.times(k), data.states(k,:)).';
    memory(k,:) = original_memory(data.times(k), data.states(k,:)).';
end
% Same initial state and RK4 isolate the RHS from ode45 and lsqnonlin choices.
trajectory = zeros(data.steps+1,14); trajectory(1,:) = data.x0;
for k=1:double(data.steps)
    tt = (k-1)*data.dt; xx = trajectory(k,:).';
    k1 = rhs(tt,xx); k2 = rhs(tt+data.dt/2,xx+data.dt*k1/2);
    k3 = rhs(tt+data.dt/2,xx+data.dt*k2/2); k4 = rhs(tt+data.dt,xx+data.dt*k3);
    trajectory(k+1,:) = (xx+data.dt*(k1+2*k2+2*k3+k4)/6).';
end
save(output_file,'mass','force','memory','trajectory','-v7');
rmpath(folder);
end

function value = slice(source, first, last)
a = strfind(source,first); b = strfind(source,last);
assert(numel(a)==1 && numel(b)==1 && b>a,'Original source markers changed');
value = source(a:b-1);
end

function dx = rhs(t,x)
M = original_mass(t,x.'); F = original_force(t,x.');
lambda = 1e-9*trace(M)/12;
solution = (M+lambda*eye(12))\F;
dx = [x(7:12);solution(1:6);original_memory(t,x.')];
end
