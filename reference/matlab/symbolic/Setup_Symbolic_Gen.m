%% Setup_Symbolic_Gen.m
% -------------------------------------------------------------------------
% V3: Robust Physics Generation for Insect-Scale Soft Robot
% -------------------------------------------------------------------------

clc; clear; close all;
fprintf('Initializing symbolic derivation (V3 - Robust)... \n');

%% 1. DEFINE SYMBOLIC VARIABLES
syms t real
% State: q(6) + v(6) + s(2)
syms x y th1 th2 th3 th4 real
syms vx vy w1 w2 w3 w4 real
syms ax ay alpha1 alpha2 alpha3 alpha4 real
syms s1 s2 real

% Symbolic Arrays
p = {x; y; th1; th2; th3; th4};
v = {vx; vy; w1; w2; w3; w4};
a = {ax; ay; alpha1; alpha2; alpha3; alpha4};
X = [x, y, th1, th2, th3, th4, vx, vy, w1, w2, w3, w4, s1, s2];

% Derivatives for kinematics
syms xd yd th1d th2d th3d th4d real
syms vxd vyd w1d w2d w3d w4d real
pd = [xd; yd; th1d; th2d; th3d; th4d];
vd = [vxd; vyd; w1d; w2d; w3d; w4d];

%% 2. PARAMETER VECTOR (P) - Size 42 (FIXED)
syms m1 m2 m3 m4 I1 I2 I3 I4 g real
syms l1 l2 l3 l4 lleg1 lleg2 al1 al2 h1 h2 real
syms kspring damping tau_mag freq real
syms theta1_init theta2_init theta3_init real
syms ky_gnd dy_gnd real
syms mu_k_fwd mu_k_bwd mu_s_fwd mu_s_bwd real
syms vdead vscale pcoeff kt_ratio ct_ratio real
syms lambda_slip lambda_off dx_fwd dx_bwd real

% 41 physical parameters; Python appends an unused 42nd entry.
P = [m1; m2; m3; m4; I1; I2; I3; I4; g; ...        % MATLAB 1-9
     l1; l2; l3; l4; lleg1; lleg2; al1; al2; h1; h2; ... % 10-19
     kspring; damping; tau_mag; freq; ...                % 20-23
     theta1_init; theta2_init; theta3_init; ...          % 24-26
     ky_gnd; dy_gnd; ...                                 % 27-28
     mu_k_fwd; mu_k_bwd; mu_s_fwd; mu_s_bwd; ...         % 29-32
     vdead; vscale; pcoeff; kt_ratio; ct_ratio; ...      % 33-37
     lambda_slip; lambda_off; dx_fwd; dx_bwd];           % 38-41

%% 3. KINEMATICS
fprintf('  - Kinematics...\n');
cg1 = [x;y];
pin1 = cg1 + [-l1/2*cos(th1); -l1/2*sin(th1)];
cg2  = pin1 + [-l2/2*cos(th2); -l2/2*sin(th2)];
pin2 = cg2  + [-l2/2*cos(th2); -l2/2*sin(th2)];
cg3  = pin2 + [-l3/2*cos(th3); -l3/2*sin(th3)];
pin3 = cg3  + [-l3/2*cos(th3); -l3/2*sin(th3)];
cg4  = pin3 + [-l4/2*cos(th4); -l4/2*sin(th4)];

joint1 = cg1 + [(l1/2-h1)*cos(th1); (l1/2-h1)*sin(th1)];
joint2 = cg4 + [-(l4/2-h2)*cos(th4); -(l4/2-h2)*sin(th4)];

foot1 = joint1 + [lleg1*cos(al1-th1); -lleg1*sin(al1-th1)];
foot2 = joint2 + [lleg2*cos(al2-th4); -lleg2*sin(al2-th4)];

%% 4. EXACT CORIOLIS & DYNAMICS
fprintf('  - Dynamics & Accelerations...\n');
% Velocity Helper Function
diffT_ = @(f, q, dq) jacobian(f, q) * dq;

v2 = subs(diffT_(cg2, [p{:}].', [v{:}].'), pd, v);
v3 = subs(diffT_(cg3, [p{:}].', [v{:}].'), pd, v);
v4 = subs(diffT_(cg4, [p{:}].', [v{:}].'), pd, v);
vfoot1 = subs(diffT_(foot1, [p{:}].', [v{:}].'), pd, v);
vfoot2 = subs(diffT_(foot2, [p{:}].', [v{:}].'), pd, v);

% Acceleration Helper
a2 = subs(diffT_(v2, [p{:}, v{:}].', [v{:}, a{:}].'), [pd;vd], [v{:}, a{:}].');
a3 = subs(diffT_(v3, [p{:}, v{:}].', [v{:}, a{:}].'), [pd;vd], [v{:}, a{:}].');
a4 = subs(diffT_(v4, [p{:}, v{:}].', [v{:}, a{:}].'), [pd;vd], [v{:}, a{:}].');

a_total = [ax; ay; alpha1; a2; alpha2; a3; alpha3; a4; alpha4];
a_cencori = a_total;

% Isolate Mass Matrix coefficients
coeffs = sym(zeros(12, 6));
% a_total has 12 elements (ax,ay,al1, a2x,a2y,al2, a3x,a3y,al3, a4x,a4y,al4)
% We must expand a2, a3, a4 into scalar components for the loop
a_total_scalar = [ax; ay; alpha1; a2(1); a2(2); alpha2; a3(1); a3(2); alpha3; a4(1); a4(2); alpha4];
a_cencori_scalar = a_total_scalar;
a_vars = [ax, ay, alpha1, alpha2, alpha3, alpha4];

for ii = 1:6
    coeffs(:,ii) = diff(a_cencori_scalar, a_vars(ii));
    a_cencori_scalar = a_cencori_scalar - coeffs(:,ii)*a_vars(ii);
end

%% 5. FRICTION MODEL
fprintf('  - Friction Model...\n');
F1y = piecewise(foot1(2) < 0, -ky_gnd * foot1(2) - dy_gnd * vfoot1(2), 0);
F2y = piecewise(foot2(2) < 0, -ky_gnd * foot2(2) - dy_gnd * vfoot2(2), 0);
F1y = max(0, F1y); F2y = max(0, F2y); % Clamp

% Foot 1
v_rel1 = vfoot1(1);
mu_k1 = piecewise(v_rel1 <= 0, mu_k_bwd, mu_k_fwd);
mu_s1 = piecewise(v_rel1 <= 0, mu_s_bwd, mu_s_fwd);
dx_1  = piecewise(v_rel1 <= 0, dx_bwd, dx_fwd);
mu_str1 = mu_k1 + (mu_s1 - mu_k1) * exp(-(abs(v_rel1)/vscale)^pcoeff);
sgn1 = tanh(v_rel1 / vdead);

kt1 = kt_ratio * ky_gnd; ct1 = ct_ratio * dy_gnd;
F_stick1 = kt1*s1 + ct1*v_rel1;
is_stick1 = (abs(v_rel1) < vdead) & (abs(F_stick1) <= mu_s1 * F1y) & (F1y > 0);
F_slip1 = -(mu_str1*F1y*sgn1 + dx_1*v_rel1);
F1x = piecewise(F1y <= 0, 0, is_stick1, -F_stick1, F_slip1);
sdot1 = piecewise(F1y <= 0, -lambda_off*s1, is_stick1, v_rel1, -lambda_slip*s1);

% Foot 2
v_rel2 = vfoot2(1);
mu_k2 = piecewise(v_rel2 <= 0, mu_k_bwd, mu_k_fwd);
mu_s2 = piecewise(v_rel2 <= 0, mu_s_bwd, mu_s_fwd);
dx_2  = piecewise(v_rel2 <= 0, dx_bwd, dx_fwd);
mu_str2 = mu_k2 + (mu_s2 - mu_k2) * exp(-(abs(v_rel2)/vscale)^pcoeff);
sgn2 = tanh(v_rel2 / vdead);

kt2 = kt_ratio * ky_gnd; ct2 = ct_ratio * dy_gnd;
F_stick2 = kt2*s2 + ct2*v_rel2;
is_stick2 = (abs(v_rel2) < vdead) & (abs(F_stick2) <= mu_s2 * F2y) & (F2y > 0);
F_slip2 = -(mu_str2*F2y*sgn2 + dx_2*v_rel2);
F2x = piecewise(F2y <= 0, 0, is_stick2, -F_stick2, F_slip2);
sdot2 = piecewise(F2y <= 0, -lambda_off*s2, is_stick2, v_rel2, -lambda_slip*s2);

%% 6. FORCE BALANCE
fprintf('  - Assembly...\n');
M_diag = diag([m1; m1; I1; m2; m2; I2; m3; m3; I3; m4; m4; I4]);

phase = 2 * pi * freq * t;
tau1_act = kspring*(th2 - th1 - theta1_init) + damping*(w2-w1) + tau_mag*sin(phase);
tau2_act = kspring*(th3 - th2 - theta2_init) + damping*(w3-w2) + tau_mag*sin(phase);
tau3_act = kspring*(th4 - th3 - theta3_init) + damping*(w4-w3) + tau_mag*sin(phase);

F_vec = [F1x; F1y - m1*g;
         tau1_act - F1x*(foot1(2)-cg1(2)) + F1y*(foot1(1)-cg1(1));
         0; -m2*g; -tau1_act + tau2_act;
         0; -m3*g; -tau2_act + tau3_act;
         F2x; F2y - m4*g;
         -tau3_act - F2x*(foot2(2)-cg4(2)) + F2y*(foot2(1)-cg4(1))];

F_total = F_vec - M_diag * a_cencori_scalar;
M_gen = M_diag * coeffs;

% Constraint Matrix (Rows=Equations, Cols=Pin Forces)
% Pin 1 (Body 1 & 2), Pin 2 (Body 2 & 3), Pin 3 (Body 3 & 4)
Constraint_Matrix = ...
    [1, 0, 0, 0, 0, 0;
     0, 1, 0, 0, 0, 0;
     l1/2*sin(th1), -l1/2*cos(th1), 0, 0, 0, 0;
    -1, 0, 1, 0, 0, 0;
     0,-1, 0, 1, 0, 0;
     l2/2*sin(th2), -l2/2*cos(th2), l2/2*sin(th2), -l2/2*cos(th2), 0, 0;
     0, 0,-1, 0, 1, 0;
     0, 0, 0,-1, 0, 1;
     0, 0, l3/2*sin(th3), -l3/2*cos(th3), l3/2*sin(th3), -l3/2*cos(th3);
     0, 0, 0, 0,-1, 0;
     0, 0, 0, 0, 0,-1;
     0, 0, 0, 0, l4/2*sin(th4), -l4/2*cos(th4)];

M_final = [M_gen, Constraint_Matrix];

%% 7. EXPORT
fprintf('  - Generating Files...\n');
% Keep exports beside the maintained bounded initial-pose solver.
output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'generated');
matlabFunction(M_final, 'File', fullfile(output_dir, 'computeM_gen'), 'Vars', {t, X, P}, 'Optimize', false);
matlabFunction(F_total, 'File', fullfile(output_dir, 'computeF_gen'), 'Vars', {t, X, P}, 'Optimize', false);
matlabFunction([sdot1; sdot2], 'File', fullfile(output_dir, 'computeSdot_gen'), 'Vars', {t, X, P}, 'Optimize', false);
% solveInitialPose_gen.m is maintained separately; it uses P(24:26).
fprintf('Done. Generated equations in %s.\n', output_dir);
