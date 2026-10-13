function matlab_parameter_reference(output_file)
% Evaluate only the original numerical setup, without symbolic dynamics or GUI.
root = fileparts(fileparts(mfilename('fullpath')));
archive = fullfile(root, 'reference', 'matlab', 'original');
addpath(fullfile(archive, 'helpers'));
source = fileread(fullfile(archive, 'RobotDynamics_mulbo_legdirstiff_stribeck_karnopp_iCVD_dam_thick.m'));
first = strfind(source, 'para.Body.ha =');
last = strfind(source, '%% III: Simulation');
assert(numel(first) == 1 && numel(last) == 1 && last > first, 'Original setup markers changed');
eval(source(first:last-1));
reference.mass_kg = para.body.m_front1;
reference.inertia_kg_m2 = para.body.I_front1;
reference.spring_n_m_rad = para.Total.kspring;
reference.damping_n_m_s_rad = para.Total.damping;
reference.torque_n_m = para.Force.tau;
reference.frequency_hz = para.freqs;
reference.front_leg_offset_m = para.leg.hfront;
reference.hind_leg_offset_m = para.leg.hhind;
reference.front_leg_angle_rad = para.leg.theta_front;
reference.hind_leg_angle_rad = para.leg.theta_hind;
reference.front_contact_stiffness_n_m = para.Ground.kyf(1);
reference.hind_contact_stiffness_n_m = para.Ground.kyh(1);
reference.contact_damping_n_s_m = para.Ground.dy;
reference.initial_foot_clearance_m = 1e-6 + 0.0005; % objectiveF target + X0 drop.
save(output_file, 'reference', '-v7');
end
