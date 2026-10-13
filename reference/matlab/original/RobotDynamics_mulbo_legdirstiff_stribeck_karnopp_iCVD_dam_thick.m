%%
clc
clear
close all

filepath='/home/hri-david/code_winter_workshop/matlab_code';
%%
syms t % time

% States
syms x y th1 th2 th3 th4
syms vx vy w1 w2 w3 w4
syms ax ay alpha1 alpha2 alpha3 alpha4
syms s1 s2
p = {x; y; th1; th2; th3; th4}; % front link CG x (m), CGy (m), angle (rad), rear link angle (rad)
v = {vx; vy; w1; w2; w3; w4}; % time derivatives of p
a = {ax; ay; alpha1; alpha2; alpha3; alpha4}; % time deriviatives of v
X = [x, y, th1, th2, th3, th4, vx, vy, w1, w2, w3, w4, s1, s2]; % state vector                             

syms xd yd th1d th2d th3d th4d
syms vxd vyd w1d w2d w3d w4d
pd = [xd; yd; th1d; th2d; th3d; th4d]; % temporary time derivatives of p
vd = [vxd; vyd; w1d; w2d; w3d; w4d]; % temporary time derivatives of v

% Parameters 
% syms l10x l10y l12 l21 l20x l20y % kinematic parameters
syms l1 l2 l3 l4 h1 h2 lleg1 lleg2 al1 al2  % kinematic parameters
syms m1 m2 m3 m4 I1 I2 I3 I4 % mass parameters

% Forces
syms F1x F1y F2x F2y F3x F3y tau1 tau2 tau3 g % force parameters

numBody = 4;
cg1 = [x;y]; % front link CG point

pin1 = cg1+ [-l1/2*cos(th1); % pin joint point
             -l1/2*sin(th1)];

cg2 = pin1 + [-l2/2*cos(th2); % pin joint point
             -l2/2*sin(th2)];

pin2 = cg2 + [-l2/2*cos(th2); % pin joint point
             -l2/2*sin(th2)];

cg3 = pin2 + [-l3/2*cos(th3); % rear link CG point
             -l3/2*sin(th3)];

pin3 = cg3 + [-l3/2*cos(th3); % pin joint point
             -l3/2*sin(th3)];

cg4 = pin3 + [-l4/2*cos(th4); % rear link CG point
             -l4/2*sin(th4)];

edge1 = cg1 + [+l1/2*cos(th1); % rear link CG point
             +l1/2*sin(th1)];

edge2 = cg4 + [-l4/2*cos(th4); % rear link CG point
             -l4/2*sin(th4)];

joint1 = cg1 + [+(l1/2-h1)*cos(th1); % rear link CG point
             +(l1/2-h1)*sin(th1)];

joint2= cg4 + [-(l4/2-h2)*cos(th4); % rear link CG point
             -(l4/2-h2)*sin(th4)];

foot1 = joint1 + [lleg1*cos(al1-th1); -lleg1*sin(al1-th1)];
foot2 = joint2 + [lleg2*cos(al2-th4); - lleg2*sin(al2-th4)];

% Velocities
vpin1 = diffT(pin1,p,1); % 2nd link
vpin1 = subs(vpin1,pd,v);

v2 = diffT(cg2,p,1); % 2nd link
v2 = subs(v2,pd,v);

vpin2 = diffT(pin2,p,1); % 2nd link
vpin2 = subs(vpin2,pd,v);

v3 = diffT(cg3,p,1); % rear link CG velocity
v3 = subs(v3,pd,v);

vpin3 = diffT(pin3,p,1); % 2nd link
vpin3 = subs(vpin3,pd,v);

v4 = diffT(cg4,p,1); % rear link CG velocity
v4 = subs(v4,pd,v);

vfoot1 = subs(diffT(foot1,p,1),pd,v); % front foot velocity
vfoot2 = subs(diffT(foot2,p,1),pd,v); % rear foot velocity

% Accelerations
a2 = diffT(v2,[p;v],1); % rear link CG acceleration
a2 = subs(a2,[pd;vd],[v;a]);

a3 = diffT(v3,[p;v],1); % rear link CG acceleration
a3 = subs(a3,[pd;vd],[v;a]);

a4 = diffT(v4,[p;v],1); % rear link CG acceleration
a4 = subs(a4,[pd;vd],[v;a]);

coeffs = vpa(zeros(numBody*3,length(a)));
a_total = [ax; ay; alpha1; a2; alpha2; a3; alpha3; a4; alpha4];
a_cencori=a_total;

for ii = 1:length(a)
    coeffs(:,ii) = diff(a_cencori,a(ii));
    a_cencori = a_cencori - coeffs(:,ii)*a(ii); %centripetal과 coriolis 구하기.
end

%%
% Dynamics ----------------------------------------------------------------
% Force & torque balance matrices: M*x = F
% where x = [ax1; ay1; alpha1; ax2; ay2; alpha2]
% Only diagonal term is considered for simple calculation and modeling.
M = diag([m1; m1; I1; m2; m2; I2; m3; m3; I3; m4; m4; I4]);

% Exerted forcing terms are considered except for internal forces.
% M*const = 
F = [F1x;
    F1y - m1*g;
    tau1 - F1x*(foot1(2)-cg1(2)) + F1y*(foot1(1)-cg1(1));
    0;
    -m2*g;
    -tau1+tau2;
    0;
    -m3*g;
    -tau2+tau3;
    F2x;
    F2y - m4*g;
    -tau3 - F2x*(foot2(2)-cg4(2)) + F2y*(foot2(1)-cg4(1))]- M*a_cencori;

% Transforming into minimal coordinates M*z = F
% where z = [ax1; ay1; alpha1; alpha2; Fpx; Fpy]
% Jacobian terms multiplied to Mass matrix inducing 6 x 4 matrix.
M = M*coeffs;

% Pin joint x and y reaction forces respectively
% adding internal forcing terms.
M = [M, ...
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
     0, 0, 0, 0, l4/2*sin(th4), -l4/2*cos(th4)]];

%% II: Simulate with numerical parameters =================================
% 1. Material & Geometry Parameters
para.Body.ha = 80e-6;  para.Body.Ya = 2.8e9;  para.Body.ra = 1853;
para.Body.hp = 75e-6;  para.Body.Yp = 4.0e9;  para.Body.rp = 1461;
para.Body.had = 1e-6;  para.Body.Yad = 61e6;  para.Body.rad = 1000; % iCVD
% para.Body.had = [1e-6, 5e-6, 10e-6,30e-6, 50e-6];
para.Body.had = 1e-6;
% para.Body.had = 25e-6;  para.Body.Yad = 330e3;  para.Body.rad = 1071; % OCA
% para.Body.had = 5e-6;  para.Body.Yad = 1355e6;  para.Body.rad = 1131; % UTT

para.Total.width = 15e-3;        % Width of entire body [m]
para.Total.length = 25e-3;       % Length of entire body [m]
para.Total.theta_deg = 12;
para.Total.theta = deg2rad(para.Total.theta_deg); % Initial angle of two body [rad]

para.Total.theta1=para.Total.theta/3;
para.Total.theta2=para.Total.theta/3;
para.Total.theta3=para.Total.theta/3;

para.leg.hfront = 2.8e-3;       % Initial distance of front leg fixed point from front body's end point [m]
para.leg.hhind = 2.5e-3;       % Initial distance of hind leg fixed point from hind body's end point [m]
para.leg.lfront = 2.5e-3; % Length of front leg [m]
para.leg.lhind= 2.5e-3; % Length of hind leg [m]
para.leg.tfront = 25e-6;  % Thickness of front leg [m]
para.leg.thind = 25e-6;  % Thickness of hind leg [m]
para.leg.Ifront = 1/12 * para.Total.width * (para.leg.tfront)^3;
para.leg.Ihind = 1/12 * para.Total.width * (para.leg.thind)^3;

para.leg.theta_front_deg = 95; % Initial angle between front body and front leg [deg]
para.leg.theta_hind_deg = 125; % Initial angle between hind body and hind leg [deg]
para.leg.theta_front = deg2rad(para.leg.theta_front_deg);  % Initial angle between front body and front leg [rad]
para.leg.theta_hind = deg2rad(para.leg.theta_hind_deg); % Initial angle between hind body and hind leg [rad]
para.leg.hinfront = para.leg.lfront/10;
para.leg.hinhind = para.leg.lhind/10;

para.Total.zeta = 0.01;
% para.Total.zeta = linspace(0.01,0.09,5); 

para.Total.d31 = -22e-12;
para.Force.volt = 200;
% para.Force.volt = linspace(50,200,4);

para.Total.rhoA = para.Total.width * (para.Body.ra * para.Body.ha + para.Body.rp * para.Body.hp + para.Body.rad * para.Body.had); % mass per length 
para.Total.y_bar = parallel_axis(para.Total.width, para.Body.Ya, para.Body.Yp, para.Body.Yad, para.Body.ha, para.Body.hp, para.Body.had);
para.Total.EI = inertia_2(para.Total.width, para.Body.Ya, para.Body.Yp, para.Body.Yad, para.Body.ha, para.Body.hp, para.Body.had, para.Total.y_bar);
para.Total.beta1 = 4.73;
para.Total.omega = (para.Total.beta1^2) * sqrt(para.Total.EI / (para.Total.rhoA * para.Total.length^4));
para.Total.freq_frefre = para.Total.omega / (2*pi);

para.body.length1 = para.Total.length / 4;
para.body.length2 = para.Total.length / 4;
para.body.length3 = para.Total.length / 4;
para.body.length4 = para.Total.length / 4;

para.body.m_front1 = para.Total.rhoA .* para.body.length1;
para.body.m_front2 = para.Total.rhoA .* para.body.length2;
para.body.m_front3 = para.Total.rhoA .* para.body.length3;
para.body.m_front4 = para.Total.rhoA .* para.body.length4;

% para.body.m_hind = para.Total.rhoA * para.Total.length - para.body.m_front;
% para.body.l_front = para.Total.length / 2; % half length of body.
% para.body.l_hind = para.Total.length - para.body.l_front;

para.body.I_front1 = (1/12) * para.body.m_front1 * (para.body.length1)^2;
para.body.I_front2 = (1/12) * para.body.m_front2 * (para.body.length2)^2;
para.body.I_front3 = (1/12) * para.body.m_front3 * (para.body.length3)^2;
para.body.I_front4 = (1/12) * para.body.m_front4 * (para.body.length4)^2;

% para.Total.I_eff = para.body.I_front1;
para.Total.kspring = 2*(para.Total.EI)/(para.body.length1)/2;
% para.Total.kspring = (2 * pi * para.Total.freq_frefre)^2 * para.Total.I_eff;
para.Total.damping = 2 * para.Total.zeta' * para.Total.kspring ./ para.Total.omega;
para.Total.omega_2nd = 22.373/(para.Total.length)^2 .*sqrt(para.Total.EI./para.Total.rhoA);
para.Total.kspring_2nd = (para.Total.EI) ./(para.body.length1) ;
%% Leg Stiffness
para.Force.g = 9.81;
para.Force.Mpiezo = MomentPiezo(para.Total.width, para.Total.d31, para.Force.volt, para.Body.Ya, para.Body.ha, para.Body.hp, para.Body.had, para.Total.y_bar);
para.Force.tau = para.Force.Mpiezo;

% forward
% angle between hind-leg axis and ground normal [deg]
phi_degf = 180-para.leg.theta_front_deg+rad2deg(para.Total.theta1)+rad2deg(para.Total.theta2)/2;
% avoid singularity when phi -> 0 deg
denf = sind(phi_degf).^2;
denf(denf < sind(0.5)^2) = sind(0.5)^2;   % clamp at 0.5 deg (tune if needed)
para.Ground.kyf1 = 3*(para.leg.Ihind * para.Body.Yp) / (para.leg.lfront)^3 .* (1./denf).*linspace(1,5,5);       % ground stiffness (N/m)
para.Ground.kyf1 = ceil(para.Ground.kyf1);

para.Ground.kyf2 = 3*(para.leg.Ihind * para.Body.Yp) / (para.leg.lfront)^3 .*linspace(1,5,5);       % ground stiffness (N/m)
para.Ground.kyf2 = ceil(para.Ground.kyf2);

% hind
% angle between hind-leg axis and ground normal [deg]
phi_degh = 180-para.leg.theta_hind_deg+rad2deg(para.Total.theta3)+rad2deg(para.Total.theta2)/2;

% avoid singularity when phi -> 0 deg
denh = sind(phi_degh).^2;
denh(denh < sind(0.5)^2) = sind(0.5)^2;   % clamp at 0.5 deg (tune if needed)

para.Ground.kyh1 = 3*(para.leg.Ihind * para.Body.Yp) / (para.leg.lhind)^3 .* (1./denh).*linspace(1,5,5);       % ground stiffness (N/m)
para.Ground.kyh1 = ceil(para.Ground.kyh1);

para.Ground.kyh2 = 3*(para.leg.Ihind * para.Body.Yp) / (para.leg.lhind)^3 .*linspace(1,5,5);       % ground stiffness (N/m)
para.Ground.kyh2 = ceil(para.Ground.kyh2);

% para.Ground.ky = 10E0;         % ground stiffness (N/m)
% para.Ground.dy = 5E-3;          % ground damping [N/(m/s)]

% Ground parameters
para.Total.foldlinef = 65e-6;
para.Total.foldlineh = 40e-6;
para.Total.foldlinef_plus = para.Total.foldlinef*0.5;
para.Total.foldlinef_min = para.Total.foldlinef*2;
para.Total.foldlineh_plus = para.Total.foldlineh*0.5;
para.Total.foldlineh_min = para.Total.foldlineh*2;

para.Ground.D_strip = para.Body.Yp*para.leg.tfront^3/12;
para.Ground.hingef = para.Ground.D_strip * para.Total.width / para.Total.foldlinef;
para.Ground.hingeh = para.Ground.D_strip * para.Total.width / para.Total.foldlineh;
para.Ground.hingef_plus = para.Ground.D_strip * para.Total.width / para.Total.foldlinef_plus;
para.Ground.hingef_min = para.Ground.D_strip * para.Total.width / para.Total.foldlinef_min;
para.Ground.hingeh_plus = para.Ground.D_strip * para.Total.width / para.Total.foldlineh_plus;
para.Ground.hingeh_min = para.Ground.D_strip * para.Total.width / para.Total.foldlineh_min;

para.Ground.compliancef_plus = 1./para.Ground.kyf2 + (para.leg.lfront)^2/para.Ground.hingef_plus;
para.Ground.compliancef_min = 1./para.Ground.kyf2 + (para.leg.lfront)^2/para.Ground.hingef_min;
para.Ground.complianceh_plus = 1./para.Ground.kyh2 + (para.leg.lfront)^2/para.Ground.hingeh_plus;
para.Ground.complianceh_min = 1./para.Ground.kyh2 + (para.leg.lfront)^2/para.Ground.hingeh_min;

para.Ground.stifff_plus = 1./(para.Ground.compliancef_plus).*(1./denf);
para.Ground.stifff_min = 1./(para.Ground.compliancef_min).*(1./denf);
para.Ground.stiffh_plus = 1./(para.Ground.complianceh_plus).*(1./denh);
para.Ground.stiffh_min = 1./(para.Ground.complianceh_min).*(1./denh);

para.Ground.kyf = ceil((para.Ground.stifff_plus + para.Ground.stifff_min) / 2);
para.Ground.kyh = ceil((para.Ground.stiffh_plus + para.Ground.stiffh_min) / 2);
%% Friction
para.Ground.mu_forward = 0.03;          % coefficient of kinetic friction (ratio)
para.Ground.mu_backward = 0.4;          % coefficient of kinetic friction (ratio)
para.Ground.mus_forward = 0.03;         % coefficient of static friction (ratio)
para.Ground.mus_backward = 0.8;         % coefficient of static friction (ratio)
% para.Ground.mus_backward = linspace(0.5,0.9,5);
para.Ground.dy = 8e-5;          % ground damping [N/(m/s)]
para.Ground.dx_backward = 3e-3;           % x ground damping for smoothing [N/(m/s)]
para.Ground.dx_forward = 1e-4;           % x ground damping for smoothing [N/(m/s)]

% Actuators parameters
% para.freqs = logspace(2,3,8);   % Acutator frequencies to simulate (Hz)
% para.freqs = linspace(360,365,2);   % Acutator frequencies to simulate (Hz)
% para.freqs1 = 363;   % Acutator frequencies to simulate (Hz) iCVD 1um
% para.freqs10 = 388;   % Acutator frequencies to simulate (Hz) iCVD 10um
% para.freqs20 = 416;   % Acutator frequencies to simulate (Hz) iCVD 20um
% para.freqs30 = 443;   % Acutator frequencies to simulate (Hz) iCVD 30um
% para.freqs40 = 470;   % Acutator frequencies to simulate (Hz) iCVD 40um
% para.freqs50 = 497;   % Acutator frequencies to simulate (Hz) iCVD 50um
% para.freqsto =[para.freqs1,para.freqs10, para.freqs20, para.freqs30, para.freqs40, para.freqs50];
para.freqs = ceil(para.Total.omega_2nd/(2*pi));

% para.freqs = 380;   % Acutator frequencies to simulate (Hz) UTT

para.Ground.v_threshold1 = 1e-3; % v_eps 예시 값
para.Ground.v_threshold2 = 50e-3; % static to kinetic
para.Ground.coeffi = 2; 
% para.Ground.v_threshold2 = linspace(50e-3,250e-3,5);

% para.variable1=para.Ground.kyf;
% para.variable2=para.Ground.kyh;

para.variable1=para.Total.zeta;
para.variable2=para.Body.had;

para.trajectories = cell(length(para.variable1),length(para.variable2),2);
%% III: Simulation ========================================================
h = waitbar(0, 'Simulation running...');
total = length(para.variable1) * length(para.variable2);
count = 0;
para.tend = 1.0;
para.framerate = 5000;   % Framerate (hz)
for ii = 1:length(para.variable1)
    fprintf([num2str(ii),': ']);
    for jj =1:length(para.variable2)
        fprintf([num2str(jj),': ']);
        count = count + 1;
        tic
        %%
        % Parameters derived from those specified above
        params1 =  [m1, m2, m3, m4, I1, I2, I3, I4, g, l1, l2, l3, l4, lleg1, al1, lleg2, al2, h1, h2];
        params1_ = [para.body.m_front1(jj), para.body.m_front2(jj), para.body.m_front3(jj), para.body.m_front4(jj), para.body.I_front1(jj), para.body.I_front2(jj), para.body.I_front3(jj), para.body.I_front4(jj), ...
            para.Force.g, para.body.length1, para.body.length2, para.body.length3, para.body.length4, para.leg.lfront, para.leg.theta_front, ... 
            para.leg.lhind, para.leg.theta_hind, para.leg.hfront, para.leg.hhind];
        % hNumerical expressions ---------------------------------------------------
        foot1_ = subs(foot1,params1,params1_);
        pin1_ = subs(pin1,params1,params1_);
        pin2_ = subs(pin2,params1,params1_);
        pin3_ = subs(pin3,params1,params1_);
        cg2_ = subs(cg2,params1,params1_);
        cg3_ = subs(cg3,params1,params1_);
        cg4_ = subs(cg4,params1,params1_);
        foot2_ = subs(foot2,params1,params1_);
        
        edge1_ = subs(edge1,params1,params1_);
        edge2_ = subs(edge2,params1,params1_);
        joint1_ = subs(joint1,params1,params1_);
        joint2_ = subs(joint2,params1,params1_);
        
        vfoot1_ = subs(vfoot1,params1,params1_);
        vpin1_ = subs(vpin1,params1,params1_);
        vpin2_ = subs(vpin2,params1,params1_);
        vpin3_ = subs(vpin3,params1,params1_);
        
        vcg2_ = subs(v2,params1,params1_);
        vcg3_ = subs(v3,params1,params1_);
        vcg4_ = subs(v4,params1,params1_);
        vfoot2_ = subs(vfoot2,params1,params1_);
        
        % Functions
        ffoot1 = dynamicsFunction_leg(foot1_, X, 'compute_ffoot1');
        fpin1 = dynamicsFunction_leg(pin1_, X, 'compute_fpin1');
        fpin2 = dynamicsFunction_leg(pin2_, X, 'compute_fpin2');
        fpin3 = dynamicsFunction_leg(pin3_, X, 'compute_fpin3');
        
        fcg2 = dynamicsFunction_leg(cg2_, X, 'compute_fcg2');
        fcg3 = dynamicsFunction_leg(cg3_, X, 'compute_fcg3');
        fcg4 = dynamicsFunction_leg(cg4_, X, 'compute_fcg4');
        
        ffoot2 = dynamicsFunction_leg(foot2_, X, 'compute_ffoot2');
        ffoot1y = dynamicsFunction_leg(foot1_(2), X, 'compute_ffoot1y');
        ffoot2y = dynamicsFunction_leg(foot2_(2), X, 'compute_ffoot2y');
        
        fedge1 = dynamicsFunction_leg(edge1_, X, 'compute_fedge1');
        fedge2 = dynamicsFunction_leg(edge2_, X, 'compute_fedge2');
        
        fjoint1 = dynamicsFunction_leg(joint1_, X, 'compute_fjoint1');
        fjoint2 = dynamicsFunction_leg(joint2_, X, 'compute_fjoint2');
        
        fvpin1 = dynamicsFunction_leg(vpin1_(1), X, 'compute_fvpin1');
        fvpin2 = dynamicsFunction_leg(vpin2_(1), X, 'compute_fvpin2');
        fvpin3 = dynamicsFunction_leg(vpin3_(1), X, 'compute_fvpin3');
        
        fvfoot1y = dynamicsFunction_leg(vfoot1_(2), X, 'compute_fvfoot1y');
        fvfoot2y = dynamicsFunction_leg(vfoot2_(2), X, 'compute_fvfoot2y');
        fvfoot1x = dynamicsFunction_leg(vfoot1_(1), X, 'compute_fvfoot1x');
        fvfoot2x = dynamicsFunction_leg(vfoot2_(1), X, 'compute_fvfoot2x');

        % Solve for initial conditions
        objectiveF = matlabFunction([foot2_(2)-1E-6; foot1_(2)-1E-6; th2 - th1 - para.Total.theta1; th3 - th2 - para.Total.theta2; th4 - th3 - para.Total.theta3; foot1_(1)], ...
            'Vars',{x,y,th1,th2,th3,th4});
        % initial pose
        pos0 = lsqnonlin(@(x)objectiveF(x(1),x(2),x(3),x(4),x(5),x(6)), [0;0;0;0;0;0]);

        % Forces
        min_s = @(i1,i2) heaviside(i1-i2)*i2 + heaviside(i2-i1)*i1;
        
        % 이거는 piecewise에 (ii)가 들어가서 계산이 안 됨.



        % % % % ground stiffness : ky & Hybrid friction model
        F1y_ = heaviside(-foot1_(2)) * heaviside(-para.Ground.kyf(1) * foot1_(2) - para.Ground.dy * vfoot1_(2)) * (-para.Ground.kyf(1)  * foot1_(2) - para.Ground.dy * vfoot1_(2));
        % F1x_ = -sign(vfoot1_(1)) * ( min_s( abs( para.Ground.dx * vfoot1_(1) ) , abs( para.Ground.mu_f * F1y_ ) ) )
        F2y_ = heaviside(-foot2_(2)) * heaviside(-para.Ground.kyh(1) * foot2_(2) - para.Ground.dy * vfoot2_(2)) * (-para.Ground.kyh(1)  * foot2_(2) - para.Ground.dy * vfoot2_(2));
        % F2x_ = -sign(vfoot2_(1)) * ( min_s( abs( para.Ground.dx * vfoot2_(1) ) , abs( para.Ground.mu_f * F2y_ ) ) )

        % 7) Stribeck + Karnopp stick window friction model
        % 속도 작을때는 스틱 모드를 적용.
        % --- parameters you already have ---
        vdead    = para.Ground.v_threshold1;     % Karnopp deadband (stick window)
        vscale   = para.Ground.v_threshold2;     % Stribeck speed scale
        pcoeff   = para.Ground.coeffi;           % Stribeck exponent (e.g., 2)
        bleed    = 0.5;                          % slip 시 micro-spring bleed (0~1)
        kt_ratio_f = 0.2;                        % tangential micro-stiffness ratio vs ky (front)
        kt_ratio_h = 0.2;                        % (hind)
        ct_ratio   = 0.1;                        % tangential micro-damping ratio vs dy

        para.Ground.lambda_slip = 200;    % 슬립/작동 중 s 이완 속도
        para.Ground.lambda_off  = 1000;   % 비접촉 시 빠른 이완
        % syms lambda_slip lambda_off real
        v_rel1x_sym = vfoot1_(1);              % 상대 속도
        ground_contact1 = piecewise(F1y_ > 0, 1, 0);
        % 마찰 계수 (symbolic)
        mu_k1_sym = piecewise((v_rel1x_sym) <= 0, para.Ground.mu_backward, (v_rel1x_sym) > 0, para.Ground.mu_forward);
        mu_s1_sym = piecewise((v_rel1x_sym) <= 0, para.Ground.mus_backward, (v_rel1x_sym) > 0, para.Ground.mus_forward);
        dx_1sym = piecewise((v_rel1x_sym) <= 0, para.Ground.dx_backward, (v_rel1x_sym) > 0, para.Ground.dx_forward);
        % Stribeck mu(v)
        mu_str1 = mu_k1_sym + (mu_s1_sym - mu_k1_sym) * exp( - (abs(v_rel1x_sym)/vscale)^pcoeff );
        
        % smooth sign for slip branch
        sgn1 = tanh(v_rel1x_sym / vdead);
        
        % normal force magnitude (already >=0 by your F1y_ build)
        N1 = abs(F1y_) * ground_contact1;
        
        % tangential micro compliance (stick branch)
        kt1 = kt_ratio_f * para.Ground.kyf(1);
        ct1 = ct_ratio   * para.Ground.dy;
                                     
        Fstick1 = kt1*s1 + ct1*v_rel1x_sym;
        stick1  = (abs(v_rel1x_sym)<vdead) & (abs(Fstick1)<=mu_s1_sym*N1) & (ground_contact1==1);
        Fslip1  = -( mu_str1*N1*sgn1 + dx_1sym*v_rel1x_sym ) * ground_contact1;

        % final friction force (piecewise)
        F1x_    = piecewise(ground_contact1==0, 0, stick1, -Fstick1, Fslip1);
        
        % state update (symbolic) — evaluate numerically each step
        sdot1_  = piecewise(ground_contact1==0, -para.Ground.lambda_off*s1, stick1, v_rel1x_sym, -para.Ground.lambda_slip*s1);
        
        % --- foot2 (동일 패턴) ---
        v_rel2x_sym   = vfoot2_(1);
        ground_contact2 = piecewise(F2y_ > 0, 1, 0);
        
        mu_k2_sym = piecewise(v_rel2x_sym<=0, para.Ground.mu_backward, v_rel2x_sym>0, para.Ground.mu_forward);
        mu_s2_sym = piecewise(v_rel2x_sym<=0, para.Ground.mus_backward, v_rel2x_sym>0, para.Ground.mus_forward);
        dx_2sym   = piecewise(v_rel2x_sym<=0, para.Ground.dx_backward, v_rel2x_sym>0, para.Ground.dx_forward);
        
        mu_str2 = mu_k2_sym + (mu_s2_sym - mu_k2_sym) * exp( - (abs(v_rel2x_sym)/vscale)^pcoeff );
        sgn2    = tanh(v_rel2x_sym / vdead);
        N2      = abs(F2y_) * ground_contact2;
        
        kt2 = kt_ratio_h * para.Ground.kyh(1);
        ct2 = ct_ratio   * para.Ground.dy;
        
        Fstick2 = kt2*s2 + ct2*v_rel2x_sym;
        
        stick2  = (abs(v_rel2x_sym)<vdead) & (abs(Fstick2)<=mu_s2_sym*N2) & (ground_contact2==1);
        Fslip2  = -( mu_str2*N2*sgn2 + dx_2sym*v_rel2x_sym ) * ground_contact2;
        
        F2x_    = piecewise(ground_contact2==0, 0, stick2, -Fstick2, Fslip2);
        
        sdot2_  = piecewise(ground_contact2==0, -para.Ground.lambda_off*s2, stick2, v_rel2x_sym,-para.Ground.lambda_slip*s2);

        fF1x{ii,jj} = dynamicsFunction_leg(F1x_, X, ['computeF1x_', num2str(ii), '_', num2str(jj)]);
        fF1y{ii,jj} = dynamicsFunction_leg(F1y_, X, ['computeF1y_', num2str(ii), '_', num2str(jj)]);

        fF2x{ii,jj} = dynamicsFunction_leg(F2x_, X, ['computeF2x_', num2str(ii), '_', num2str(jj)]);
        fF2y{ii,jj} = dynamicsFunction_leg(F2y_, X, ['computeF2y_', num2str(ii), '_', num2str(jj)]);

        % NEW: sdot 함수
        fs1dot{ii,jj} = dynamicsFunction_leg(sdot1_, X, ['computeS1dot_', num2str(ii),'_',num2str(jj)]);
        fs2dot{ii,jj} = dynamicsFunction_leg(sdot2_, X, ['computeS2dot_', num2str(ii),'_',num2str(jj)]);
        % % Motor torque
        % tau_ = ( para.Total.kspring * (th2 - th1 - para.Total.theta) + para.Force.tau * ( sin(2 * pi * para.freqs * t) ) + para.Total.damping(jj) * (w2-w1));
        
        % Motor torque
        tau1_ = ( para.Total.kspring(jj) * (th2 - th1 - para.Total.theta1) + para.Force.tau(jj) * ( sin(2 * pi * para.freqs(jj) * t) ) + para.Total.damping(ii,jj) * (w2-w1));
        tau2_ = ( para.Total.kspring(jj) * (th3 - th2 - para.Total.theta2) + para.Force.tau(jj) * ( sin(2 * pi * para.freqs(jj) * t) ) + para.Total.damping(ii,jj) * (w3-w2));
        tau3_ = ( para.Total.kspring(jj) * (th4 - th3 - para.Total.theta3) + para.Force.tau(jj) * ( sin(2 * pi * para.freqs(jj) * t) ) + para.Total.damping(ii,jj) * (w4-w3));
    
        params =  [params1, F1x, F1y, F2x, F2y,  tau1, tau2, tau3];
        params_ = [params1_,F1x_,F1y_,F2x_,F2y_, tau1_, tau2_, tau3_];
    
        % Numerical dynamics and simulation ---------------------------------------
        M_ = subs(M,params,params_); % numerical M matrix
        F_ = subs(F,params,params_); % numerical F vector

        % fF = dynamicsFunction_leg(F_,X); % F vector function    
        % fM = dynamicsFunction_leg(M_,X); % M matrix function
        fF = dynamicsFunction_leg(F_,X, 'computeF'); % F 벡터 함수
        fM = dynamicsFunction_leg(M_,X, 'computeM'); % M 행렬 함수

        fv = @(t,X) X(7:12); % function for velocities
    
        fprintf(['functions ',num2str(toc,3),' s. '])
    
        X0 = [pos0 + [0;0.0005;0;0;0;0]; 0;0;0;0;0;0; 0;0];    % initial state 0.5mm above the ground
        options = odeset('reltol', 1e-6, 'abstol', 1e-8);        % ODE solver options
        ts = [0,para.tend];                             % simulation duration (s)
    
        % Run the simulation
        tic
        [tout,xout] = ode45(@(t,X)MFdynamicsOrderedkarnopp(t,X,fM,fF,fv,fs1dot{ii,jj},fs2dot{ii,jj}), ts, X0, options);
        disp(['Simulated ', num2str(length(tout)),' steps in ',num2str(toc),' s'])
    
        para.trajectories(ii,jj,:) = {tout, xout};
        
        n= ii;
        m= jj;
        % Initialize arrays of uninterpolated raw timing output values
        cg2s{n,m} = zeros(length(tout),2); % total system CG location (m)
        cg3s{n,m} = zeros(length(tout),2); % back body CG location (m)
        cg4s{n,m} = zeros(length(tout),2); % back body CG location (m)
        vcg2s{n,m} = zeros(length(tout),2); % back body CG velocity (m/s)
        vcg3s{n,m} = zeros(length(tout),2); % back body CG velocity (m/s)
        vcg4s{n,m} = zeros(length(tout),2); % back body CG velocity (m/s)
        xpins{n,m} = zeros(length(tout),3); % back body CG velocity (m/s)
        ypins{n,m} = zeros(length(tout),3); % back body CG velocity (m/s)
        vpins{n,m} = zeros(length(tout),3); % back body CG velocity (m/s)
        yvpins{n,m} = zeros(length(tout),3); % back body CG velocity (m/s)
        fys{n,m} = zeros(length(tout),2); % front and back foot heights (m)
        vfys{n,m} = zeros(length(tout),2); % front and back foot y velocities (m/s)
        vfxs{n,m} = zeros(length(tout),2); % front and back foot x velocities (m/s)
        % KEs{n,m} = zeros(length(tout),2); % front and back body kinetic energy (J)
        % PEs{n,m} = zeros(length(tout),2); % front and back body potential energy (J)
        % SPEs{n,m} = 0.5*para.Total.kspring * (xout(:,4) - xout(:,3) - para.Total.theta).^2; % hinge spring potential energy (J)
    
        F1sx{n,m} = zeros(length(tout),1); % Front foot x force (N)
        F1sy{n,m} = zeros(length(tout),1); % Front foot y force (N)
        F2sx{n,m} = zeros(length(tout),1); % Back foot x force (N)
        F2sy{n,m} = zeros(length(tout),1); % Back foot y force (N)
        
        ptsx{n,m} = zeros(length(tout),13); % [front foot, front edge, front joint, front CG, pin, back CG, back edge2 back joint back foot] for plotting
        ptsy{n,m} = zeros(length(tout),13); % [front foot, front edge, front joint, front CG, pin, back CG, back edge2 back joint back foot] for plotting
    
        % Populate uninterpolated raw timing output values
        for kk = 1:length(tout)
            % cg2s{n,m}(ii,:) = fcg2(tout(ii),xout(ii,:))';
            % cg3s{n,m}(ii,:) = fcg3(tout(ii),xout(ii,:))';
            % cg4s{n,m}(ii,:) = fcg4(tout(ii),xout(ii,:))';
            % v2s{n,m}(ii,:) = fvcg2(tout(ii),xout(ii,:))';
            % cgs{n,m}(ii,:) = (para.body.m_front*xout(ii,1:2) + para.body.m_hind*cg2s{n,m}(ii,:))/(para.body.m_front+para.body.m_hind);
            fys{n,m}(kk,:) = [ffoot1y(tout(kk),xout(kk,:)), ffoot2y(tout(kk),xout(kk,:))];
            vfys{n,m}(kk,:) = [fvfoot1y(tout(kk),xout(kk,:)), fvfoot2y(tout(kk),xout(kk,:))];
            vfxs{n,m}(kk,:) = [fvfoot1x(tout(kk),xout(kk,:)), fvfoot2x(tout(kk),xout(kk,:))];
            pinst = [fpin1(tout(kk),xout(kk,:)), fpin2(tout(kk),xout(kk,:)), fpin3(tout(kk),xout(kk,:))];
            vpinst = [fvpin1(tout(kk),xout(kk,:)), fvpin2(tout(kk),xout(kk,:)), fvpin3(tout(kk),xout(kk,:))];
            % KEs{n,m}(ii,:) = [0.5 * para.body.m_front * sum(xout(ii,5:6).^2) + 0.5 * para.body.I_front * xout(ii,7).^2, ...
                % 0.5 * para.body.m_hind * sum(v2s{n,m}(ii,:).^2) + 0.5 * para.body.I_hind * xout(ii,8).^2];
            % PEs{n,m}(ii,:) = [para.body.m_front * para.Force.g * xout(ii,2), para.body.m_hind * para.Force.g * cg2s{n,m}(ii,2)];

            % pt = [ffoot1(tout(ii),xout(ii,:)), fjoint1(tout(ii),xout(ii,:)), fedge1(tout(ii),xout(ii,:)), xout(ii,1:2)', fpin1(tout(ii),xout(ii,:)), ...
                % fcg2(tout(ii),xout(ii,:)), fpin2(tout(ii),xout(ii,:)), fcg3(tout(ii),xout(ii,:)), fpin3(tout(ii),xout(ii,:)), ...
                % fcg4(tout(ii),xout(ii,:)), fedge2(tout(ii),xout(ii,:)), fjoint2(tout(ii),xout(ii,:)), ffoot2(tout(ii),xout(ii,:))];
            % ptsx{n,m}(ii,:) = pt(1,:);
            % ptsy{n,m}(ii,:) = pt(2,:);
            xpins{n,m}(kk,:)=pinst(1,:);
            ypins{n,m}(kk,:)=pinst(2,:);
            vpins{n,m}(kk,:)=vpinst(1,:);
            % yvpins{n,m}(ii,:)=vpinst(2,:);
            F1sx{n,m}(kk) = fF1x{n,m}(tout(kk),xout(kk,:));
            F1sy{n,m}(kk) = fF1y{n,m}(tout(kk),xout(kk,:));
            F2sx{n,m}(kk) = fF2x{n,m}(tout(kk),xout(kk,:));
            F2sy{n,m}(kk) = fF2y{n,m}(tout(kk),xout(kk,:));
        end
        % cgvs{n,m} = (para.body.m_hind * v2s{n,m} + para.body.m_front * xout(:,[5,6])) / (para.body.m_front1 + para.body.m_front2 + para.body.m_front3+ para.body.m_front4);
    
        % Interpolated timings at constant framerate for plotting.
        % ODE 45를 풀어서 tout과 framerate별 시간 time이 다름.
        % 시각화를 하려면 frame별로 맞추기 위해 interpolation을 사용함.
        tint{n,m} = [tout(1):(1/para.framerate):tout(end), tout(end)];
        xint{n,m} = interp1(tout,xout,tint{n,m});
    
        ptisx{n,m} = zeros(length(tint{n,m}),13);
        ptisy{n,m} = zeros(length(tint{n,m}),13);
        cgis{n,m} = zeros(length(tint{n,m}),2);
        F1isx{n,m} = zeros(length(tint{n,m}),1);
        F1isy{n,m} = zeros(length(tint{n,m}),1);
        F2isx{n,m} = zeros(length(tint{n,m}),1);
        F2isy{n,m} = zeros(length(tint{n,m}),1);
    
        % Populate interpolated timings
        for kk = 1:length(tint{n,m})
            pt = [ffoot1(tint{n,m}(kk),xint{n,m}(kk,:)), fjoint1(tint{n,m}(kk),xint{n,m}(kk,:)), fedge1(tint{n,m}(kk),xint{n,m}(kk,:)), xint{n,m}(kk,1:2)', fpin1(tint{n,m}(kk),xint{n,m}(kk,:)), ...
                fcg2(tint{n,m}(kk),xint{n,m}(kk,:)), fpin2(tint{n,m}(kk),xint{n,m}(kk,:)), fcg3(tint{n,m}(kk),xint{n,m}(kk,:)), fpin3(tint{n,m}(kk),xint{n,m}(kk,:)), ...
                fcg4(tint{n,m}(kk),xint{n,m}(kk,:)), fedge2(tint{n,m}(kk),xint{n,m}(kk,:)), fjoint2(tint{n,m}(kk),xint{n,m}(kk,:)), ffoot2(tint{n,m}(kk),xint{n,m}(kk,:))];
            ptisx{n,m}(kk,:) = pt(1,:);
            ptisy{n,m}(kk,:) = pt(2,:);
            cgis{n,m}(kk,:) = sum(pt(:,[4,6,8,10]).*[para.body.m_front1(m), para.body.m_front2(m), para.body.m_front3(m), para.body.m_front4(m); para.body.m_front1(m), para.body.m_front2(m), para.body.m_front3(m), para.body.m_front4(m)],2)...
                / (para.body.m_front1(m) + para.body.m_front2(m) + para.body.m_front3(m)+ para.body.m_front4(m));
    
            F1isx{n,m}(kk) = fF1x{n,m}(tint{n,m}(kk),xint{n,m}(kk,:));
            F1isy{n,m}(kk) = fF1y{n,m}(tint{n,m}(kk),xint{n,m}(kk,:));
            F2isx{n,m}(kk) = fF2x{n,m}(tint{n,m}(kk),xint{n,m}(kk,:));
            F2isy{n,m}(kk) = fF2y{n,m}(tint{n,m}(kk),xint{n,m}(kk,:));
        end
    
        % Gestures
        t_window = (tout > 0.2); % Times at which to extract gestures
        contracting = xout(:,11) > xout(:,10); % Contraction-expansion phase
        front_touch = F1sy{n,m} > 0; % Front foot on the ground, = 0 이면 지면에서 떨어짐.
        back_touch = F2sy{n,m} > 0; % Back foot on the ground, = 0 이면 지면에서 떨어짐.
    
        dt{n,m} = tout(2:end) - tout(1:(end-1)); % time step (s)
    
        % Gesture arrays
        % To classify the locomotion gait with time. 
        aci{n,m} = contracting & ~front_touch & ~back_touch & t_window; % Aerial contracting
        aei{n,m} = ~contracting & ~front_touch & ~back_touch & t_window; % Aerial expanding
        fci{n,m} = contracting & front_touch & ~back_touch & t_window; % Front-touching contracting
        fei{n,m} = ~contracting & front_touch & ~back_touch & t_window; % Front-touching expanding
        rci{n,m} = contracting & ~front_touch & back_touch & t_window; % Rear-touching contracting
        rei{n,m} = ~contracting & ~front_touch & back_touch & t_window; % Rear-touching expanding
        bci{n,m} = contracting & front_touch & back_touch & t_window; % Both-touching contracting
        bei{n,m} = ~contracting & front_touch & back_touch & t_window; % Both-touching expanding
        
        % Direction of foot movement
        fr{n,m} = vfxs{n,m}(:,1) < 0 & front_touch & ~back_touch & t_window; % Front foot drags back
        ff{n,m} = vfxs{n,m}(:,1) > 0 & front_touch & ~back_touch & t_window; % Front foot drags forward
        br{n,m} = vfxs{n,m}(:,2) < 0 & ~front_touch & back_touch & t_window; % Rear foot drags back
        bf{n,m} = vfxs{n,m}(:,2) > 0 & ~front_touch & back_touch & t_window; % Rear foot drags forward
    
        % Count gestures and transitions between gestures
        gestureVector{n,m} = [aci{n,m}, aei{n,m}, fci{n,m}, fei{n,m}, rci{n,m}, rei{n,m}, bci{n,m}, bei{n,m}];
        gestureTransitions{n,m} = find(gestureVector{n,m}(1,:));
        gestureMatrix{n,m} = zeros(8,8);
        for kk = 2:length(tout)
            prev = find(gestureVector{n,m}(kk-1,:));
            next = find(gestureVector{n,m}(kk,:));
            if prev ~= next
                gestureMatrix{n,m}(next,prev) = gestureMatrix{n,m}(next,prev)+1;
                gestureTransitions{n,m} = [gestureTransitions{n,m}; next];
            end
        end
        
        % Duty cycles for each gesture
        ac{n,m} = sum(dt{n,m}.*aci{n,m}(1:(end-1)));
        ae{n,m} = sum(dt{n,m}.*aei{n,m}(1:(end-1)));
        fc{n,m} = sum(dt{n,m}.*fci{n,m}(1:(end-1)));
        fe{n,m} = sum(dt{n,m}.*fei{n,m}(1:(end-1)));
        rc{n,m} = sum(dt{n,m}.*rci{n,m}(1:(end-1)));
        re{n,m} = sum(dt{n,m}.*rei{n,m}(1:(end-1)));
        bc{n,m} = sum(dt{n,m}.*bci{n,m}(1:(end-1)));
        be{n,m} = sum(dt{n,m}.*bei{n,m}(1:(end-1)));
        t_win{n,m} = t_window;

        waitbar(count / total, h, sprintf('Progress: %d / %d', count, total));
    end
end
close(h);

%% V: Plotting ============================================================
% title format

% title_format = 'kfy %.2f & khy %.2f';
% title_format2 ='kfy %.2f & khy %.2f %.2f ms';
% title_format = 'zeta %.2f & volt %.2f';
% title_format2 ='zeta %.2f & volt  %.2f \n %.1f ms';

title_format = 'zeta %.2f & thick %.1fμm' ;
title_format2 ='zeta %.2f & thick %.1fμm %.1f ms';
normunit = 1e6;
timegap= 35e-4;
saturated_ratio = 0.9;
para.tendgraph = para.tend*saturated_ratio;
para.tstartgraph = para.tendgraph-timegap;
%% 1. ============= Gait Analysis ====================
colors = initializeColors();
gr.barWidth=0.2*2; % bar width default : 0.8
gr.LineWidth=1;
% ax=gca;
xname = categorical({'AC','AE','FC','FE','RC','RE','BC','BE'});
tt=figure(1);
nrow = length(para.variable1);
ncol = length(para.variable2);
ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.07 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1820 1000]);
for ii=1:1:length(para.variable1)
    for jj=1:1:length(para.variable2)
        axes(ha(jj+ncol*(ii-1)));  % jj번째 subplot에 접근
        % ac{n}, ae{n} 등은 시간별로 제스처에 해당하는 구간을 나타내는 논리 벡터 (true/false)
        % sum(dt .* t_window(...))는 총 분석 시간 (초 단위)
        % 따라서 ac{n} 같은 벡터를 dt로 가중합하면 → 해당 제스처의 시간 비중
        % for n = toPlot
            % stem((n-1)/(2*length(para.variable1))+(1:8),100*[ac{n},ae{n},fc{n},fe{n},rc{n},re{n},bc{n},be{n}]/ ...
            %     sum(dt.*t_window(1:(end-1))))
        % end
        data = 100 * [ac{ii,jj}, ae{ii,jj}, fc{ii,jj}, fe{ii,jj}, rc{ii,jj}, re{ii,jj}, bc{ii,jj}, be{ii,jj}] / sum(dt{ii,jj}.*t_win{ii,jj}(1:end-1));
        ba = bar(xname, data,'EdgeColor',getColor(colors, 'area', 8, 1),'LineWidth',gr.LineWidth,'BarWidth', gr.barWidth);
        % function customizeBarPlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, ystart, yend, ysadd, yeadd, divisiony, gridOn)
        % customizeBarPlot(1200, 1000, 140, 100, 0.86, 0.80, 1,     0, 50, 0, 5, 5, 'off')
        ax=gca;
        ax.XTickLabelRotation=0; % 최대 최소 값을 수직으로 보여주기. 각도에따라 기울여져서 보임.
        xlabel('Duty cycle');
        ylabel('Gesture');
        ylim([0 55])
        set(gca,'fontsize',14)
        hold off
        % title(['Ground Stiff ',num2str(para.variable1(ii)),'& Damping ratio ',num2str(para.variable2(jj))]);
        % title(['Spring kf ', num2str(para.variable1(ii)),' & kh ', num2str(para.variable2(jj))]);
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
        % title(sprintf('Freq %.1f & Zeta %.2f', para.variable1(ii),para.variable2(jj)  ) );
        title(sprintf(title_format, para.variable1(ii) , para.variable2(jj)*normunit) );
        gaitdata{ii,jj}=data;
    end
end
% filename=['Gait Analysis LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Spring kf %.1f to %.1f & kh %.1f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Gait Analysis LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];

savefig(tt, fullfile(filepath, filename));

%% 2. ============= Trajectory ====================
tt=figure(10);
nrow = length(para.variable1);
ncol = length(para.variable2);
ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1860 1100]);
xmax_steady=zeros(nrow,ncol);
ymax_steady=zeros(nrow,ncol);
for ii=1:1:length(para.variable1)
    for jj=1:1:length(para.variable2)
        axes(ha(jj+ncol*(ii-1)))  % jj번째 subplot에 접근
        tout = para.trajectories{ii,jj,1};
        xout = para.trajectories{ii,jj,2};
        plot(tout,xout(:,1:2)) % x,y축 방향으로 cg1의 위치.
        hold on
        plot(tout,xpins{ii,jj}(:,2)) % x축 방향으로 cg2의 위치.
        hold on
        plot(tout,ypins{ii,jj}(:,2)) % y축 방향으로 cg2의 위치.
        % plot(tout,cgs{ii,jj}(:,1),'r') % x축 방향 total cg의 위치
        % plot(tout,cgs{ii,jj}(:,2),':g') % y축 방향 total cg의 위치
        % hold on
        % legend('x','y','cgx','cgy','Location','northwest');
        xlabel('Time (s)')
        ylabel('Distance (m)')
        set(gca,'fontsize',12)
        xlim([0 para.tend])
        ylim([-0.1 1.0]);
        % title(['Mu_',num2str(para.variable1(ii)),'_dx_',num2str(para.variable2(jj))]);
        % title(['Ground Stiff ',num2str(para.variable1(ii)),'& Damping ratio ',num2str(para.variable2(jj))]);
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
        title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
        xmax_steady(ii,jj) = max(xout(end-ceil(para.framerate/para.freqs(jj)*500):end,1));
        ymax_steady(ii,jj) = max(xout(end-ceil(para.framerate/para.freqs(jj)*500):end,2));
    end
end
% filename=['Trajectory LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Spring kf %.1f to %.1f & kh %.1f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Trajectory LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

[~, linear_index] = max(xmax_steady(:));
[para.drow_index, para.dcol_index] = ind2sub(size(xmax_steady), linear_index);
% max만 plot 및 trajectory 그리고 gait 저장.
tt=figure(11);
ii=para.drow_index;
jj=para.dcol_index;

ha = tight_subplot(1, 1, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1820 1000]);
axes(ha(1)) % jj번째 subplot에 접근
tout = para.trajectories{ii,jj,1};
xout = para.trajectories{ii,jj,2};
plot(tout,xout(:,1:2)) % x,y축 방향으로 cg1의 위치.
% plot(tout,cgs{ii,jj}(:,1),'r') % x축 방향 total cg의 위치
% plot(tout,cgs{ii,jj}(:,2),':g') % y축 방향 total cg의 위치
% hold on
legend('x','y','Location','northwest');
xlabel('Time (s)')
ylabel('Distance (m)')
set(gca,'fontsize',12)
xlim([0 para.tend])
ylim([-0.1 1.0]);
% title(['Mu_',num2str(para.variable1(ii)),'_dx_',num2str(para.variable2(jj))]);
% title(['Ground Stiff ',num2str(para.variable1(ii)),'& Damping ratio ',num2str(para.variable2(jj))]);
% title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
filename=['Trajectory Magni LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

%% 3. ============= Velocity ====================
tt=figure(15);
nrow = length(para.variable1);
ncol = length(para.variable2);
ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1860 1100]);
vxmax_steady=zeros(nrow,ncol);
vymax_steady=zeros(nrow,ncol);
for ii=1:1:length(para.variable1)
    for jj=1:1:length(para.variable2)
        axes(ha(jj+ncol*(ii-1)))  % jj번째 subplot에 접근
        tout = para.trajectories{ii,jj,1};
        xout = para.trajectories{ii,jj,2};
        plot(tout,xout(:,8),'-', 'Color',getColor(colors, 'area', 1, 2), 'LineWidth', 2) % pin 1 y velocity
        hold on
        plot(tout,xout(:,7),'-', 'Color',getColor(colors, 'area', 3, 2), 'LineWidth', 2) % pin 1 x velocity
        hold on
        plot(tout,vpins{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 4, 2), 'LineWidth', 2) % entire cg x velocity 
        hold on
        % legend('vx','vy','v-cgx','v-cgy','Location','northwest');
        % plot(tout,yvpins(:,2))
        % hold on
        xlabel('Time (s)')
        ylabel('Velocity (m/s)')
        xlim([0 para.tend])
        ylim([-1.5 1.5])
        set(gca,'fontsize',12)
        % title(['Mu_',num2str(para.variable1(ii)),'_dx_',num2str(para.variable2(jj))]);
        % title(['Ground Stiff ',num2str(para.variable1(ii)),'& Damping ratio ',num2str(para.variable2(jj))]);
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
        title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
        vxmax_steady(ii,jj) = max(xout(end-ceil(para.framerate/para.freqs(jj)*500):end,7));
        vymax_steady(ii,jj) = max(xout(end-ceil(para.framerate/para.freqs(jj)*500):end,8));
    end
end
% filename=['Velocity LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Spring kf %.1f to %.1f & kh %.1f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Velocity LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];

savefig(tt, fullfile(filepath, filename));

[~, linear_index] = max(vxmax_steady(:));
[para.vrow_index, para.vcol_index] = ind2sub(size(vxmax_steady), linear_index);
% max 속도만 plot
tt=figure(16);
ii=para.vrow_index;
jj=para.vcol_index;

ha = tight_subplot(1, 1, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1820 1000]);
axes(ha(1))  % jj번째 subplot에 접근
tout = para.trajectories{ii,jj,1};
xout = para.trajectories{ii,jj,2};
plot(tout,xout(:,8),'-', 'Color',getColor(colors, 'area', 1, 2), 'LineWidth', 2); hold on;
plot(tout,xout(:,7),'-', 'Color',getColor(colors, 'area', 3, 2), 'LineWidth', 2); hold on;
plot(tout,vpins{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 4, 2), 'LineWidth', 2) % entire cg x velocity 
xlabel('Time (s)')
ylabel('Velocity (m/s)')
xlim([0 para.tend])
ylim([-1.5 1.5])
% title(['Mu_',num2str(para.variable1(ii)),'_dx_',num2str(para.variable2(jj))]);
% title(['Ground Stiff ',num2str(para.variable1(ii)),'& Damping ratio ',num2str(para.variable2(jj))]);
% title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
% filename=['Velocity LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Spring kf %.1f to %.1f & kh %.1f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Velocity Magni LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit ,para.variable2(end)*normunit ) ,'.fig'];

savefig(tt, fullfile(filepath, filename));

%%
% 4. ============= Friction Force ====================
for ii = 1:length(para.variable1)
    
    figure(20 + ii)
    % ax = gca;
    % set(ax, 'Units', 'pixels');
    nrow = length(para.variable2);
    ncol = 1;

    % 생성: subplot 배열 (세로 간격 0.05, 여백 0.05)
    ha = tight_subplot(nrow, ncol, [0.05 0.04], [0.07 0.07], [0.04 0.02]);  
    % [gap between rows/cols], [bottom & top margin], [left & right margin]
    set(gcf,'OuterPosition',[30 30 1820 1000]);
    for jj = 1:nrow
        axes(ha(jj));  % jj번째 subplot에 접근
        plot(tint{ii, jj}, F1isx{ii, jj}, 'r'); hold on;
        plot(tint{ii, jj}, F2isx{ii, jj}, 'b');
        
        if jj == nrow
            xlabel('Time (s)');
        else
            set(gca, 'xticklabel', []);  % 중복되는 xlabel 제거
        end
        ylabel('Friction force (N)');
        set(gca, 'fontsize', 12);
        xlim([para.tstartgraph para.tendgraph])
        ylim([-15*1e-4 15*1e-4])
        
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
        title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
    end
end

% 5. ============= Normal Force ====================
for ii = 1:length(para.variable1)
    
    figure(30 + ii)
    % ax = gca;
    % set(ax, 'Units', 'pixels');
    nrow = length(para.variable2);
    ncol = 1;

    % 생성: subplot 배열 (세로 간격 0.05, 여백 0.05)
    ha = tight_subplot(nrow, ncol, [0.05 0.04], [0.07 0.07], [0.04 0.02]);  
    % [gap between rows/cols], [bottom & top margin], [left & right margin]
    set(gcf,'OuterPosition',[30 30 1820 1000]);
    for jj = 1:nrow
        axes(ha(jj))  % jj번째 subplot에 접근
        plot(tint{ii, jj}, F1isy{ii, jj}); hold on;
        plot(tint{ii, jj}, F2isy{ii, jj});
        
        if jj == nrow
            xlabel('Time (s)');
        else
            set(gca, 'xticklabel', []);  % 중복되는 xlabel 제거
        end
        ylabel('Normal force (N)');
        set(gca, 'fontsize', 12);
        xlim([para.tstartgraph para.tendgraph])
        ylim([-5*1e-4 40*1e-4])
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii),para.variable2(jj)  ) );
        title(sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit  ) );
    end
end

%% 6. ============= FOOT Velcotiy and Height (REAl, Not interpolated) ====================
num_var1 = length(para.variable1);
num_var2 = length(para.variable2);
dif=0;
mag=2;
coeffi=1000;
colors = initializeColors();

% gait cell array (당신 코드의 gait 결과들로 교체)
gaitDataAll = {aei, aci, fci, bci, rci, fei, bei, rei,};

for ii = 1:num_var1
    figure(40 + ii);
    set(gcf,'OuterPosition',[30 30 1820 1000]);
    tiledlayout((num_var1-dif)*2, 1, 'TileSpacing', 'tight', 'Padding', 'compact'); % 타일 레이아웃 생성
    
    for jj = 1:num_var2
        % Foot Velocity Plot
        nexttile; % 다음 타일로 이동
        ylimVel = [min([vfxs{ii,jj}(:); vfys{ii,jj}(:); vpins{ii,jj}(:,2)]) * mag, max([vfxs{ii,jj}(:); vfys{ii,jj}(:); vpins{ii,jj}(:,2)]) * mag];
        
        % Gait 상태 배경색 표시
        [index_start_vel, index_end_vel] = getIndexRange(para.trajectories{ii, jj, 1}, para.tstartgraph-timegap*5, para.tendgraph+timegap*5);
        for k = 1:length(gaitDataAll)
            gait = gaitDataAll{k}{ii,jj};
            plotGaitBackground(para.trajectories{ii, jj, 1}, gait, index_start_vel, index_end_vel, ylimVel, getColor(colors, 'area_gait', k, 1));
            hold on;
        end
        plot(para.trajectories{ii, jj, 1}, vfys{ii,jj}(:,1),'-', 'Color',getColor(colors, 'area', 3, 2), 'LineWidth', 2); hold on; % y - front (초록색)
        plot(para.trajectories{ii, jj, 1}, vfys{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 6, 2), 'LineWidth', 2); hold on % y - hind (보라색)
        plot(para.trajectories{ii, jj, 1}, vfxs{ii,jj}(:,1),'-', 'Color',getColor(colors, 'area', 1, 2), 'LineWidth', 2); hold on; % x - front (빨간색)
        plot(para.trajectories{ii, jj, 1}, vfxs{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 4, 2), 'LineWidth', 2); hold on % x - hind (파란색)
        % plot(para.trajectories{ii, jj, 1}, vpins{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 8, 2), 'LineWidth', 2); hold on % pin2
        ylabel('Velocity (m/s)');
        title(sprintf('%s (Velocity)', sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit )));
        ylim([-1 3])
        xlim([para.tstartgraph para.tendgraph]);
        % if jj < num_var2
        %     set(gca, 'xticklabel', []);
        % end

        % Foot Height Plot
        nexttile; % 다음 타일로 이동
        ylimHeight = [min(min(fys{ii,jj}(:,:)))* mag, max(max(fys{ii,jj}(:,:))) * mag]*coeffi;
        
        % Gait 상태 배경색 표시
        [index_start_height, index_end_height] = getIndexRange(para.trajectories{ii, jj, 1}, para.tstartgraph-timegap*5, para.tendgraph+timegap*5);
        for k = 1:length(gaitDataAll)
            gait = gaitDataAll{k}{ii,jj};
            plotGaitBackground(para.trajectories{ii, jj, 1}, gait, index_start_height, index_end_height, ylimHeight, getColor(colors, 'area_gait', k, 1));
            hold on;
        end
        plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,1)*coeffi,'-', 'Color',getColor(colors, 'area', 3, 2),'LineWidth', 2); hold on; % y - front green
        plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,2)*coeffi,'-', 'Color',getColor(colors, 'area', 6, 2),'LineWidth', 2); hold on % y - hind purple
        yline(0, '-k', 'LineWidth', 2); % y=0 선 추가 (검은색 점선)
        ylabel('Height (mm)');
        xlim([para.tstartgraph para.tendgraph]);
        title(sprintf('%s (Height)', sprintf(title_format, para.variable1(ii), para.variable2(jj)*normunit)));
        xlabel('Time (s)')
        ylim([-5*1e-1 15*1e-1])
        % if jj == num_var2
        %     ;
        % else
        %     set(gca, 'xticklabel', []);
        % end
    end
end
%% 7. ============= Friction Force & Gait Analysis (REAl, Not interpolated) ====================
stateNames = {'AE','AC','FC','BC','RC','FE','BE','RE'};
% stateColors = [
%     1 0 0;      % AC - red
%     0 1 0;      % AE - green
% 
%     0 0 1;      % FC - blue
%     1 1 0;      % FE - yellow
% 
%     1 0 1;      % RC - magenta
%     0 1 1;      % RE - cyan
% 
%     0.5 0.5 0.5;% BC - gray
%     0.4940 0.1840 0.5560       % BE - purple
% ];

% gait cell array (당신 코드의 gait 결과들로 교체)
gaitDataAll = {aei, aci, fci, bci, rci, fei, bei, rei,};

colors = initializeColors();
% Friction force + gait
for ii = 1:length(para.variable1)
    figure(60 + ii)
    nrow = length(para.variable2);
    ha = tight_subplot(nrow, 1, [0.08 0.04], [0.07 0.07], [0.04 0.02]);
    % [gap between rows/cols], [bottom & top margin], [left & right margin]
    set(gcf,'OuterPosition',[30 30 1820 1000]);
    
    for jj = 1:nrow
        axes(ha(jj));
        
        % ⭐⭐ gait 상태 먼저 patch로 표시 ⭐⭐
        
        ylimForce = [-10*1e-4 50*1e-4];
        [x1 , index1] = min( abs(para.trajectories{ii, jj, 1} - (para.tstartgraph - timegap * 5))) ;
        [x2 , index2] = min( abs(para.trajectories{ii, jj, 1} - (para.tendgraph + timegap * 5))) ;
        
        for k = 1:8
            gait = gaitDataAll{k}{ii,jj};
            for s = index1:index2 
                if gait(s) == 1
                    x1 = para.trajectories{ii, jj, 1}(s);
                    x2 = para.trajectories{ii, jj, 1}(s+1);
                    x_fill = [x1 x2 x2 x1];
                    y_fill = [ylimForce(1) ylimForce(1) ylimForce(2) ylimForce(2)];
                    
                    % patch → fill으로 변경. patch는 코드돌리면 무거움.
                    fill(x_fill, y_fill, getColor(colors, 'area_gait', k, 1), ...
                        'FaceAlpha', 0.2, 'EdgeColor','none'); 
                    hold on
                end
            end
        end
        % ⭐⭐ Force plot ⭐⭐
        plot(para.trajectories{ii, jj, 1}, F1sy{ii, jj}, 'Color',getColor(colors, 'area', 3, 1),'LineWidth', 1.5, 'Marker', '^', 'MarkerSize', 6); hold on; % y - front (초록색 사각형)
        plot(para.trajectories{ii, jj, 1}, F2sy{ii, jj}, 'Color',getColor(colors, 'area', 6, 1),'LineWidth', 1.5, 'Marker', 'd', 'MarkerSize', 6); hold on; % y - hind (보라색 사각형)
        plot(para.trajectories{ii, jj, 1}, F1sx{ii, jj}, 'Color',getColor(colors, 'area', 1, 1),'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 6); hold on;  % x - front (빨간색 원)
        plot(para.trajectories{ii, jj, 1}, F2sx{ii, jj}, 'Color',getColor(colors, 'area', 4, 1),'LineWidth', 1.5, 'Marker', 'sq', 'MarkerSize', 6); hold on;  % x - hind (파란색 원)
        % if jj == nrow
        %     xlabel('Time (s)');
        % else
        %     set(gca, 'xticklabel', []);
        % end
        xlabel('Time (s)')
        ylabel('Force (N)');
        set(gca, 'fontsize', 12);
        xlim([para.tstartgraph para.tendgraph])
        ylim(ylimForce)
        % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii), para.variable2(jj)));
        title(sprintf(title_format, para.variable1(ii),para.variable2(jj)*normunit  ) );
    end
end
%%
% for ii = 1:length(para.variable1)
% 
%     figure(40 + ii)
%     % ax = gca;
%     % set(ax, 'Units', 'pixels');
%     nrow = length(para.variable2);
%     nrow = 4;
% 
%     % 생성: subplot 배열 (세로 간격 0.05, 여백 0.05)
%     ha = tight_subplot(nrow, 1, [0.05 0.04], [0.07 0.07], [0.04 0.02]);  
%     % [gap between rows/cols], [bottom & top margin], [left & right margin]
%     set(gcf,'OuterPosition',[30 30 1820 1000]);
%     for jj = 1:nrow
%         axes(ha(jj));  % jj번째 subplot에 접근
%         % plot(para.trajectories{ii, jj, 1},  vfxs{ii,jj}(:,1), 'r'); hold on; % x - front
%         % plot(para.trajectories{ii, jj, 1},  vfxs{ii,jj}(:,2), 'b'); hold on % x - hind
%         % plot(para.trajectories{ii, jj, 1},  vfys{ii,jj}(:,1), 'g'); hold on; % y - front
%         % plot(para.trajectories{ii, jj, 1},  vfys{ii,jj}(:,2), 'm'); hold on % y - hind
%         % plot(para.trajectories{ii, jj, 1},  ptisx{ii,jj}(:,7), 'm'); hold on % y - hind
% 
%         plot(para.trajectories{ii, jj, 1}, vfxs{ii,jj}(:,1),'-', 'Color',getColor(colors, 'area', 1, 1), 'LineWidth', 2); hold on; % x - front (빨간색 원)
%         plot(para.trajectories{ii, jj, 1}, vfxs{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 4, 1), 'LineWidth', 2); hold on % x - hind (파란색 사각형)
%         % plot(para.trajectories{ii, jj, 1}, vfys{ii,jj}(:,1), 'g', 'LineWidth', 1.5, 'Marker', 'sq', 'MarkerSize', 6); hold on; % y - front (초록색 다이아몬드)
%         % plot(para.trajectories{ii, jj, 1}, vfys{ii,jj}(:,2), 'm', 'LineWidth', 1.5, 'Marker', 'sq', 'MarkerSize', 6); hold on % y - hind (자홍색 삼각형)
%         plot(para.trajectories{ii, jj, 1}, vpins{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 8, 3), 'LineWidth', 2); hold on % pin2 (검은색 실선)
%         ylabel('Velocity (m/s)');
% 
% 
%         ylabel('Position (m)');
% 
%         plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,1),'-', 'Color',getColor(colors, 'area', 1, 1),'LineWidth', 1.5); hold on; % y - front
%         plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,2),'-', 'Color',getColor(colors, 'area', 1, 1),'LineWidth', 1.5); hold on % y - hind
%         if jj == nrow
%             xlabel('Time (s)');
%         else
%             set(gca, 'xticklabel', []);  % 중복되는 xlabel 제거
%         end
%         xlim([timestart timeend])
%         % ylim([-15*1e-4 15*1e-4])
%         title(sprintf(title_format, para.variable1(ii),para.variable2(jj)  ) );
%     end
% end
% %% foot height
% for ii = 1:length(para.variable1)
% 
%     figure(50 + ii)
%     % ax = gca;
%     % set(ax, 'Units', 'pixels');
%     nrow = length(para.variable2);
%     ncol = 1;
% 
%     % 생성: subplot 배열 (세로 간격 0.05, 여백 0.05)
%     ha = tight_subplot(nrow, ncol, [0.05 0.04], [0.07 0.07], [0.04 0.02]);  
%     % [gap between rows/cols], [bottom & top margin], [left & right margin]
%     set(gcf,'OuterPosition',[30 30 1820 1000]);
%     for jj = 1:nrow
%         axes(ha(jj));  % jj번째 subplot에 접근
%         plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,1), 'r'); hold on; % y - front
%         plot(para.trajectories{ii, jj, 1},  fys{ii,jj}(:,2), 'b'); hold on % y - hind
%         ylabel('Height (m)');
%         set(gca, 'fontsize', 14);
% 
%         if jj == nrow
%             xlabel('Time (s)');
%         else
%             set(gca, 'xticklabel', []);  % 중복되는 xlabel 제거
%         end
%         xlim([timestart timeend])
%         % ylim([-15*1e-4 15*1e-4])
%         title(sprintf(title_format, para.variable1(ii),para.variable2(jj)  ) );
%     end
% end
%% Normal force + gait
% for ii = 1:length(para.variable1)
%     figure(70 + ii)
%     nrow = length(para.variable2);
%     ha = tight_subplot(nrow, 1, [0.05 0.04], [0.07 0.07], [0.04 0.02]);
%     set(gcf,'OuterPosition',[30 30 1820 1000]);
% 
%     for jj = 1:nrow
%         axes(ha(jj));
%         % ⭐⭐ gait 상태 먼저 patch로 표시 ⭐⭐
%         hold on
%         ylimForce = [-5*1e-4 50*1e-4];
%         [x1 , index1] = min( abs(para.trajectories{ii, jj, 1} - timestart )*1e+6 ) ;
%         [x2 , index2] = min( abs(para.trajectories{ii, jj, 1} - timeend )*1e+6 ) ;
%         for k = 1:8
%             gait = gaitDataAll{k}{ii,jj};
%             for s = index1:index2 
%                 if gait(s) == 1
%                     x1 = para.trajectories{ii, jj, 1}(s);
%                     x2 = para.trajectories{ii, jj, 1}(s+1);
%                     x_fill = [x1 x2 x2 x1];
%                     y_fill = [ylimForce(1) ylimForce(1) ylimForce(2) ylimForce(2)];
% 
%                     % patch → fill으로 변경. patch는 코드돌리면 무거움.
%                     fill(x_fill, y_fill, stateColors(k,:), ...
%                         'FaceAlpha', 0.4, 'EdgeColor','none');
%                     hold on
%                 end
%             end
%         end
% 
%         % ⭐⭐ Force plot ⭐⭐
%         plot(para.trajectories{ii, jj, 1}, F1sy{ii, jj}, 'r'); hold on;
%         plot(para.trajectories{ii, jj, 1}, F2sy{ii, jj}, 'b'); hold on
% 
%         if jj == nrow
%             xlabel('Time (s)');
%         else
%             set(gca, 'xticklabel', []);
%         end
%         ylabel('Normal force (N)');
%         set(gca, 'fontsize', 12);
%         xlim([timestart timeend])
%         ylim(ylimForce)
%         % title(sprintf('Spring kf %.1f & kh %.1f', para.variable1(ii), para.variable2(jj)));
%         title(sprintf(title_format, para.variable1(ii),para.variable2(jj)  ) );
% 
%     end
% end

%% 1. ============= Initial State Figure ====================

xScale = 1000;  % 위치 스케일
Fscale = 1;     % 힘 벡터 스케일
tt=figure(100);
nrow = length(para.variable1);
ncol = length(para.variable2);
ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1820 1000]);
for n=1:1:length(para.variable1)
    for m=1:1:length(para.variable2)
        axes(ha(m+ncol*(n-1)));  % jj번째 subplot에 접근

        ax = gca;
        % set(ax, 'Units', 'pixels');
        % ax.Position = [60, 60, 1632, 918];  % plot 범위 명확히 설정
        hold on
        set(gca, 'ylim', [-0.01, 0.05]);
        axis equal
        xlabel('x (mm)');
        ylabel('y (mm)');
        set(gca,'fontsize',14)
    
        % 초기 dummy 그래픽 객체 생성
        h_leg1 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
        h_body1 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
        h_body2 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
        h_body3 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
        h_body4 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
        h_leg2 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
        hF1 = quiver(0,0,0,0,'r','linewidth',2);
        hF2 = quiver(0,0,0,0,'r','linewidth',2);
        h_gnd = plot([-1,1],[0,0],'k');
        h_title = title('Initializing...');
    
        kk=1; % Initial state
        % 그래픽 업데이트
        set(h_leg1, 'xdata', xScale * ptisx{n,m}(kk,[1,2]), ...
                    'ydata', xScale * ptisy{n,m}(kk,[1,2]));
        set(h_body1, 'xdata', xScale * ptisx{n,m}(kk,[3,2,4,5]), ...
                     'ydata', xScale * ptisy{n,m}(kk,[3,2,4,5]));
        set(h_body2, 'xdata', xScale * ptisx{n,m}(kk,[5,6,7]), ...
                     'ydata', xScale * ptisy{n,m}(kk,[5,6,7]));
        set(h_body3, 'xdata', xScale * ptisx{n,m}(kk,[7,8,9]), ...
                     'ydata', xScale * ptisy{n,m}(kk,[7,8,9]));
        set(h_body4, 'xdata', xScale * ptisx{n,m}(kk,[9,10,12,11]), ...
                     'ydata', xScale * ptisy{n,m}(kk,[9,10,12,11]));
        set(h_leg2, 'xdata', xScale * ptisx{n,m}(kk,[12,13]), ...
                    'ydata', xScale * ptisy{n,m}(kk,[12,13]));
    
        set(hF1, 'xdata', xScale * ptisx{n,m}(kk,1), ...
                 'ydata', xScale * ptisy{n,m}(kk,1), ...
                 'udata', F1isx{n,m}(kk) * Fscale * xScale, ...
                 'vdata', F1isy{n,m}(kk) * Fscale * xScale);
        set(hF2, 'xdata', xScale * ptisx{n,m}(kk,13), ...
                 'ydata', xScale * ptisy{n,m}(kk,13), ...
                 'udata', F2isx{n,m}(kk) * Fscale * xScale, ...
                 'vdata', F2isy{n,m}(kk) * Fscale * xScale);
    
        % 카메라 위치 및 타이틀
        set(gca, 'xlim', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]));
        set(h_gnd, 'xdata', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]));
        % set(h_title, 'string', sprintf('Spring kf %.1f & kh %.1f', para.variable1(n), para.variable2(m)));
        set(h_title, 'string', sprintf(title_format, para.variable1(n), para.variable2(m)*normunit));
    end
end
% filename=['Robot Initial State LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Spring kf %.1f to %.1f & kh %.1f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Robot Initial State LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1),para.variable1(end) ,para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];

savefig(tt, fullfile(filepath, filename));

%

xScale = 1000;  % 위치 스케일
Fscale = 1;     % 힘 벡터 스케일
tt=figure(110);
nrow = 1;
ncol = 1;
ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
                % [gap between rows/cols], [bottom & top margin], [left & right margin]
set(gcf,'OuterPosition',[30 30 1820 1000]);
n=1;
m=1;
axes(ha(1));  % jj번째 subplot에 접근

ax = gca;
% set(ax, 'Units', 'pixels');
% ax.Position = [60, 60, 1632, 918];  % plot 범위 명확히 설정
hold on
set(gca, 'ylim', [-0.01, 0.05]);
axis equal
xlabel('x (mm)');
ylabel('y (mm)');
set(gca,'fontsize',14)

% 초기 dummy 그래픽 객체 생성
h_leg1 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
h_body1 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
h_body2 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
h_body3 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
h_body4 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
h_leg2 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
hF1 = quiver(0,0,0,0,'r','linewidth',2);
hF2 = quiver(0,0,0,0,'r','linewidth',2);
h_gnd = plot([-1,1],[0,0],'k');
h_title = title('Initializing...');

kk=1; % Initial state
% 그래픽 업데이트
set(h_leg1, 'xdata', xScale * ptisx{n,m}(kk,[1,2]), ...
            'ydata', xScale * ptisy{n,m}(kk,[1,2]));
set(h_body1, 'xdata', xScale * ptisx{n,m}(kk,[3,2,4,5]), ...
             'ydata', xScale * ptisy{n,m}(kk,[3,2,4,5]));
set(h_body2, 'xdata', xScale * ptisx{n,m}(kk,[5,6,7]), ...
             'ydata', xScale * ptisy{n,m}(kk,[5,6,7]));
set(h_body3, 'xdata', xScale * ptisx{n,m}(kk,[7,8,9]), ...
             'ydata', xScale * ptisy{n,m}(kk,[7,8,9]));
set(h_body4, 'xdata', xScale * ptisx{n,m}(kk,[9,10,12,11]), ...
             'ydata', xScale * ptisy{n,m}(kk,[9,10,12,11]));
set(h_leg2, 'xdata', xScale * ptisx{n,m}(kk,[12,13]), ...
            'ydata', xScale * ptisy{n,m}(kk,[12,13]));

set(hF1, 'xdata', xScale * ptisx{n,m}(kk,1), ...
         'ydata', xScale * ptisy{n,m}(kk,1), ...
         'udata', F1isx{n,m}(kk) * Fscale * xScale, ...
         'vdata', F1isy{n,m}(kk) * Fscale * xScale);
set(hF2, 'xdata', xScale * ptisx{n,m}(kk,13), ...
         'ydata', xScale * ptisy{n,m}(kk,13), ...
         'udata', F2isx{n,m}(kk) * Fscale * xScale, ...
         'vdata', F2isy{n,m}(kk) * Fscale * xScale);

% 카메라 위치 및 타이틀
set(gca, 'xlim', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]));
set(h_gnd, 'xdata', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]));
% set(h_title, 'string', sprintf('Spring kf %.1f & kh %.1f', para.variable1(n), para.variable2(m)));
set(h_title, 'string', sprintf(title_format, para.variable1(n), para.variable2(m)*normunit));

% filename=['Robot Initial State Magnifi LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf('Freq %.1f to %.1f & Zeta %.2f to %.1f', para.variable1(1),para.variable1(end),para.variable2(1),para.variable2(end) ) ,'.fig'];
filename=['Robot Initial State Magnifi LegF_',num2str(para.leg.theta_front_deg(1)),'_Legh_',num2str(para.leg.theta_hind_deg(1)),'_',sprintf(title_format, para.variable1(1), para.variable1(end), para.variable2(1)*normunit, para.variable2(end)*normunit ) ,'.fig'];

savefig(tt, fullfile(filepath, filename));

%% ======================================
%% 2. ======================= Entire VIDEO Recording =============================
% Robot 상태 애니메이션 + 영상 저장 (최종 개선판)
xScale = 1000;
yScale = 1000;
Fscale = 1.5;
step = 1;  % ▶ 원하는 경우 10으로 설정 시 속도 빠르게 가능

nrow = length(para.variable1);
ncol = length(para.variable2);

% ---- 원하는 출력 해상도 지정 (예: 4K) ----
VID_W = 1920;   % 가로 픽셀
VID_H = 1000;   % 세로 픽셀
FPS   = 30;

tt = figure(101); clf(tt);
set(tt, 'Color','w', 'Units','pixels', ...
    'Position',[10 10 VID_W VID_H], ...
    'Renderer','opengl', ...              % quiver 포함시 opengl 권장
    'GraphicsSmoothing','on', ...
    'PaperPositionMode','auto');

% [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
% ha = tight_subplot(nrow, ncol, [0.2 0.1], [0.02 0.02], [0.04 0.02]);
% set(tt, 'OuterPosition', [30 30 1881 1026]);
% aa = axes;
% ▶ 비디오 저장 경로 및 설정
videoFile = fullfile(filepath, 'Robot_Motion_Animation.avi');
vid = VideoWriter(videoFile, 'Motion JPEG AVI');
vid.FrameRate = FPS;
vid.Quality = 100;
open(vid);
fixed_ylim=[-0.01, 0.01]*yScale;
tl = tiledlayout(tt, nrow, ncol, 'TileSpacing','loose', 'Padding','loose');

% ▶ 그래픽 object 미리 생성
for n = 1:nrow
    for m = 1:ncol
        ax = nexttile(tl);
        axes_handles{n,m} = ax;
        hold(ax,'on');
        ylim(ax, fixed_ylim);
        xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)');
        set(ax,'FontSize',8);

        h_leg1(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        h_body1(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        h_body2(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        h_body3(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        h_body4(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        h_leg2(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
        hF1(n,m) = quiver(0,0,0,0,'r','linewidth',2);
        hF2(n,m) = quiver(0,0,0,0,'b','linewidth',2);
        h_gnd(n,m) = plot([-1,1], [0,0], 'k');
        
        % h_title(n,m) = title(sprintf('Leg-Ang-f %.1f & Leg-Ang-h %.1f', ...
        %                     para.variable1(n), para.variable2(m)));
        h_title(n,m) = title(sprintf(title_format, ...
                            para.variable1(n), para.variable2(m)*normunit));
        set(hF1(n,m), 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
        set(hF2(n,m), 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
        
        % subplot 위치 및 크기 조정 (약간 확대)
        % current_pos = get(ax, 'Position');
        % xscale_factor = 1.2;
        % yscale_factor = 1.0;
        % new_width = current_pos(3) * xscale_factor;
        % new_height = current_pos(4) * yscale_factor;
        % new_left = current_pos(1) - (new_width - current_pos(3)) / 2;
        % new_bottom = current_pos(2) - (new_height - current_pos(4)) / 2;
        % 
        % % Figure 경계 조정 (간단하게)
        % overlap_reduction = 0.5; % 겹침 감소량
        % if new_left < 0, new_left = 0 + overlap_reduction; end
        % if new_bottom < 0, new_bottom = 0 + overlap_reduction; end
        % if new_left + new_width > 1, new_width = 1 - new_left - overlap_reduction; end
        % if new_bottom + new_height > 1, new_height = 1 - new_bottom - overlap_reduction; end

        % set(ax, 'Position', [new_left, new_bottom, new_width, new_height]);
        % yticks(ax, linspace(fixed_ylim(1), fixed_ylim(2), 3)); % 고정된 ylim에 따라 yticks 설정
    end
end

% ▶ 시간 루프 (모든 subplot 동시 업데이트)

[value,t_index]=min(abs(tint{1,1}(1,:)-1));
kk_list = 1:step*5:t_index;

for kk = kk_list
    for n = 1:nrow
        for m = 1:ncol
            ax = axes_handles{n, m}; % 현재 subplot의 axes 핸들 가져오기
            set(h_leg1(n,m), 'xdata', xScale * ptisx{n,m}(kk,[1,2]), ...
                        'ydata', xScale * ptisy{n,m}(kk,[1,2]));
            set(h_body1(n,m), 'xdata', xScale * ptisx{n,m}(kk,[3,2,4,5]), ...
                         'ydata', xScale * ptisy{n,m}(kk,[3,2,4,5]));
            set(h_body2(n,m), 'xdata', xScale * ptisx{n,m}(kk,[5,6,7]), ...
                         'ydata', xScale * ptisy{n,m}(kk,[5,6,7]));
            set(h_body3(n,m), 'xdata', xScale * ptisx{n,m}(kk,[7,8,9]), ...
                         'ydata', xScale * ptisy{n,m}(kk,[7,8,9]));
            set(h_body4(n,m), 'xdata', xScale * ptisx{n,m}(kk,[9,10,12,11]), ...
                         'ydata', xScale * ptisy{n,m}(kk,[9,10,12,11]));
            set(h_leg2(n,m), 'xdata', xScale * ptisx{n,m}(kk,[12,13]), ...
                        'ydata', xScale * ptisy{n,m}(kk,[12,13]));
        
            set(hF1(n,m), 'xdata', xScale * ptisx{n,m}(kk,1), ...
                     'ydata', xScale * ptisy{n,m}(kk,1), ...
                     'udata', F1isx{n,m}(kk) * Fscale * xScale, ...
                     'vdata', F1isy{n,m}(kk) * Fscale * xScale);
            set(hF2(n,m), 'xdata', xScale * ptisx{n,m}(kk,13), ...
                     'ydata', xScale * ptisy{n,m}(kk,13), ...
                     'udata', F2isx{n,m}(kk) * Fscale * xScale, ...
                     'vdata', F2isy{n,m}(kk) * Fscale * xScale);

            set(h_title(n,m), 'String', ...
                sprintf(title_format2, ...
                        para.variable1(n), para.variable2(m)*normunit, tint{n,m}(kk)*1000));
            % camera & ground 업데이트
            xlim_val = xScale * (cgis{n,m}(kk,1) + [2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]);
            set(h_gnd(n,m), 'xdata', xlim_val);
            set(ax, 'xlim', xlim_val);
            % ylim_val =[-10 10];  
            % yticks(ax, linspace(fixed_ylim(1), fixed_ylim(2), 3)); % 고정된 ylim에 따라 yticks 설정

        end
    end

    drawnow limitrate;
    frame = getframe(tt);
    frameResized = imresize(frame.cdata, [1080, 1920]);  % Full HD 강제 고정
    writeVideo(vid, im2frame(frameResized));
    % writeVideo(vid, frame);
end

close(vid);
fprintf('영상 저장 완료: %s\n', videoFile);


%% Save Data ============================

% 저장하고 싶지 않은 필드 제거
parameter = rmfield(para, 'trajectories');

entire_tout = para.trajectories(:,:,1);
entire_cgxy = cellfun(@(X) X(:,1:2), para.trajectories(:,:,2), 'UniformOutput', false);
% 수정된 structure 저장
% save('robot_data_partial.mat', 'data');
% save(filepath, 'gaitdata','parameter','tout','xout','vpins_max','vymax_steady','vxmax_steady');
namepath=fullfile(filepath, 'robot_data.mat');
save(namepath, 'gaitdata','parameter','entire_tout','entire_cgxy','xpins','ypins','vpins','vxmax_steady','vymax_steady');
%% 3. ======================= Each VIDEO Recording ===============================
% videoBaseName = 'video';  % 저장할 비디오 접두어
% xScale = 1000;  % 위치 스케일
% Fscale = 1;     % 힘 벡터 스케일
% 
% nrow = length(para.variable1);
% ncol = length(para.variable2);
% step = 1;  % 프레임 간격 (최종적으로 속도 조절 가능)
% 
% % 병렬 풀 초기화 (필요시)
% if isempty(gcp('nocreate'))
%     parpool;  % 워커 개수 기본값 사용
% end
% 
% parfor idx = 1:(nrow * ncol)
%     [n, m] = ind2sub([nrow, ncol], idx);
% 
%     fig = figure('Visible','off');
%     set(fig, 'Color', 'w');
%     set(fig, 'Units', 'pixels');
%     set(fig, 'Position', [100, 100, 1920, 1080]);  % Full HD 크기
% 
%     % 축(axes) 생성 및 정확한 크기 지정
%     ax = axes('Parent', fig);
%     set(ax, 'Units', 'pixels');
%     set(ax, 'Position', [50, 50, 1824, 980]);  % 여백 포함, 여전히 짝수
% 
%     ax = gca;
%     % set(ax, 'Units', 'pixels');
%     % ax.Position = [60, 60, 1632, 918];  % plot 범위 명확히 설정
%     hold on
%     set(gca, 'ylim', [-0.01, 0.05]);
%     axis equal
%     xlabel('x (mm)');
%     ylabel('y (mm)');
%     set(gca,'fontsize',14)
% 
%     % 초기 dummy 그래픽 객체 생성
%     h_leg1 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
%     h_body1 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
%     h_body2 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
%     h_leg2 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
%     hF1 = quiver(0,0,0,0,'r','linewidth',2);
%     hF2 = quiver(0,0,0,0,'r','linewidth',2);
%     h_gnd = plot([-1,1],[0,0],'k');
%     h_title = title('Initializing...');
% 
%     set(hF1, 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
%     set(hF2, 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
% 
%     % 비디오 객체 생성
%     vidname = sprintf('%s_kyF%d_kyh%d.mp4', videoBaseName, n, m);
%     vid = VideoWriter(vidname, 'MPEG-4');
%     vid.FrameRate = 30;
%     vid.Quality = 100;
%     open(vid);
% 
%     T_full = size(tint{n,m}, 2);  % 전체 프레임 수
%     for kk = 1:step*10:floor(T_full/5)
%         % 그래픽 업데이트
%         set(h_leg1, 'xdata', xScale * ptisx{n,m}(kk,[1,2]), ...
%                     'ydata', xScale * ptisy{n,m}(kk,[1,2]));
%         set(h_body1, 'xdata', xScale * ptisx{n,m}(kk,[3,2,4,5]), ...
%                      'ydata', xScale * ptisy{n,m}(kk,[3,2,4,5]));
%         set(h_body2, 'xdata', xScale * ptisx{n,m}(kk,[5,6,8,7]), ...
%                      'ydata', xScale * ptisy{n,m}(kk,[5,6,8,7]));
%         set(h_leg2, 'xdata', xScale * ptisx{n,m}(kk,[8,9]), ...
%                     'ydata', xScale * ptisy{n,m}(kk,[8,9]));
% 
%         set(hF1, 'xdata', xScale * ptisx{n,m}(kk,1), ...
%                  'ydata', xScale * ptisy{n,m}(kk,1), ...
%                  'udata', F1isx{n,m}(kk) * Fscale * xScale, ...
%                  'vdata', F1isy{n,m}(kk) * Fscale * xScale);
%         set(hF2, 'xdata', xScale * ptisx{n,m}(kk,9), ...
%                  'ydata', xScale * ptisy{n,m}(kk,9), ...
%                  'udata', F2isx{n,m}(kk) * Fscale * xScale, ...
%                  'vdata', F2isy{n,m}(kk) * Fscale * xScale);
% 
%         % 카메라 위치 및 타이틀
%         set(gca, 'xlim', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.l_hind), 2*(para.body.l_front)]));
%         set(h_gnd, 'xdata', xScale*(cgis{n,m}(kk,1)+[2*(-para.body.l_hind), 2*(para.body.l_front)]));
%         set(h_title, 'string', ['Spring kf ',num2str(para.variable1(n)),' & kh',num2str(para.variable2(m)),',',num2str(floor(tint{n,m}(kk)*1000),3), ' ms']);
% 
%         drawnow limitrate;
%         frame = getframe(fig);
%         fixedFrame = imresize(frame.cdata, [1080, 1920]);  % 강제 고정 크기
%         writeVideo(vid, im2frame(fixedFrame));
%     end
% 
%     close(vid);
%     close(fig);
% end

%% 4. ============================ Only 1 video =============================
% % Animate
% % Write a video file
% videoName = 'high_quality_video_ky1_ky2.mp4';     % Save video to videoName if nonemtpy.
% saveVideo = ~isempty(videoName); % 이 비어 있지 않으면 → 영상 저장 모드
% 
% % Figure 생성 및 크기 설정 (해상도 증가)
% f_animate = figure(103);
% % ('Position', [100, 100, 1920, 1080]); % Full HD, 필요에 따라 조정
% %function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
% % customizePlot(1920, 1080, 50, 50, 0.83, 0.73,    35,    0, 500, 0, 1000,  0,0,0,0,    5, 5,  'on');
% ax = gca;
% set(gcf,'OuterPosition',[30 30 1820 1000]);
% set(ax, 'Units', 'pixels');
% ax.Position= [60,60 1632, 918];
% 
% clf; % 이전 Figure 내용 지우기
% a = axes;
% hold on
% % h = plot(0,0,'.-k','markersize',10);
% h_leg1 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
% h_body1 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
% h_body2 = plot([0,0,0,0], [0,0,0,0], 'k-', 'LineWidth', 2);
% h_leg2 = plot([0,0], [0,0], 'k-', 'LineWidth', 2);
% hF1 = quiver(0,0,0,0,'r','linewidth',1);
% hF2 = quiver(0,0,0,0,'r','linewidth',1);
% set(hF1,'autoscale','off','showarrowhead','on','MaxHeadSize', 2.5, 'LineWidth', 2);
% set(hF2,'autoscale','off','showarrowhead','on','MaxHeadSize', 2.5, 'LineWidth', 2);
% h_gnd = plot([-1,1],[0,0],'k');
% hold off
% set(gca,'ylim',[-0.01,0.05]);
% axis equal
% xlabel('x (mm)')
% ylabel('y (mm)')
% h_title = title('Initializing ...');
% set(gca,'fontsize',14)
% 
% if saveVideo
%     % VideoWriter 객체 생성 및 설정 (화질 향상)
%     vid = VideoWriter(videoName, 'MPEG-4');  % MP4 형식 (압축 효율 좋음) (avi로 하면 용량 매우 큼)
%     vid.FrameRate = 30;                     % 30 프레임 설정
%     vid.Quality = 100;                       % 최대 품질 설정
%     open(vid);                             % 영상 파일 열기
% end
% 
% xScale = 1000; % 스케일 팩터는 필요에 따라 조정
% Fscale = 1;
% n=1; m=1;
% for ii = 1:size(tint{n,m},2)/5
%     tic
%     % 앞다리
%     set(h_leg1, 'xdata', xScale * ptisx{n,m}(ii,[1,2]), ...
%                 'ydata', xScale * ptisy{n,m}(ii,[1,2]));
%     % 앞몸통
%     set(h_body1, 'xdata', xScale * ptisx{n,m}(ii,[3,2,4,5]), ...
%                  'ydata', xScale * ptisy{n,m}(ii,[3,2,4,5]));
%     % 뒷몸통
%     set(h_body2, 'xdata', xScale * ptisx{n,m}(ii,[5,6,8,7]), ...
%                  'ydata', xScale * ptisy{n,m}(ii,[5,6,8,7]));
%     % 뒷다리
%     set(h_leg2, 'xdata', xScale * ptisx{n,m}(ii,[8,9]), ...
%                 'ydata', xScale * ptisy{n,m}(ii,[8,9]));
%     % 힘 화살표
%     set(hF1, 'xdata', xScale * ptisx{n,m}(ii,1), ...
%              'ydata', xScale * ptisy{n,m}(ii,1), ...
%              'udata', F1isx{n,m}(ii) * Fscale * xScale, ...
%              'vdata', F1isy{n,m}(ii) * Fscale * xScale);
%     set(hF2, 'xdata', xScale * ptisx{n,m}(ii,9), ...
%              'ydata', xScale * ptisy{n,m}(ii,9), ...
%              'udata', F2isx{n,m}(ii) * Fscale * xScale, ...
%              'vdata', F2isy{n,m}(ii) * Fscale * xScale);
% 
%     % 카메라 중심 및 바닥선 갱신
%     set(a, 'xlim', xScale*(cgis{n,m}(ii,1)+[2*(-para.body.l_hind), 2*(para.body.l_front)]), ...
%            'ylim', xScale*[-0.01, para.body.l_front + para.body.l_hind]);
%     set(h_gnd, 'xdata', xScale*(cgis{n,m}(ii,1)+[2*(-para.body.l_hind), 2*(para.body.l_front)]));
%     set(h_title, 'string', [num2str(n),',',num2str(m),',',num2str(floor(tint{n,m}(ii)*1000),3), ' ms']);
%     drawnow;
%     % 저장 or pause
%     if saveVideo
%         frame = getframe(f_animate); % 현재 Figure의 프레임 캡처
%         writeVideo(vid, frame);       % 프레임을 VideoWriter 객체에 쓰기
%     else
%         pause(toc - (1/30));
%     end
% end
% 
% if saveVideo
%     close(vid); % VideoWriter 객체 닫기
% end

% %% 2. ======================= Entire VIDEO Recording =============================
% % Robot 상태 애니메이션 + 영상 저장 (최종 개선판)
% xScale = 1000;
% Fscale = 1.5;
% step = 1;  % ▶ 원하는 경우 10으로 설정 시 속도 빠르게 가능
% 
% nrow = length(para.variable1);
% ncol = length(para.variable2);
% 
% tt = figure(101);
% set(tt, 'Color', 'w');
% set(tt, 'Units', 'pixels');
% set(tt, 'Position', [100, 100, 1881 1026]);  % Full HD 영상 사이즈 고정
% 
% ha = tight_subplot(nrow, ncol, [0.09 0.04], [0.06 0.04], [0.04 0.01]);
% set(tt, 'OuterPosition', [30 30 1881 1026]);
% % aa = axes;
% % ▶ 비디오 저장 경로 및 설정
% videoFile = fullfile(filepath, 'Robot_Motion_Animation2.mp4');
% vid = VideoWriter(videoFile, 'MPEG-4');
% vid.FrameRate = 30;
% vid.Quality = 100;
% open(vid);
% 
% % ▶ 그래픽 object 미리 생성
% for n = 1:nrow
%     for m = 1:ncol
%         axes(ha(m + nrow*(n-1)));
%         hold on;
%         set(gca, 'ylim', [-0.01, 0.05]);
%         axis equal;
%         xlabel('x (mm)');
%         ylabel('y (mm)');
%         set(gca,'fontsize',10);
% 
%         h_leg1(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         h_body1(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         h_body2(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         h_body3(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         h_body4(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         h_leg2(n,m) = plot(NaN, NaN, 'k-', 'LineWidth', 2);
%         hF1(n,m) = quiver(0,0,0,0,'r','linewidth',2);
%         hF2(n,m) = quiver(0,0,0,0,'b','linewidth',2);
%         h_gnd(n,m) = plot([-1,1], [0,0], 'k');
%         % h_title(n,m) = title(sprintf('Leg-Ang-f %.1f & Leg-Ang-h %.1f', ...
%         %                     para.variable1(n), para.variable2(m)));
%         h_title(n,m) = title(sprintf(title_format, ...
%                             para.variable1(n), para.variable2(m)));
%         set(hF1(n,m), 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
%         set(hF2(n,m), 'autoscale','off','showarrowhead','on','MaxHeadSize',2.5,'LineWidth', 2);
%     end
% end
% 
% % ▶ 시간 루프 (모든 subplot 동시 업데이트)
% T = size(ptisx{1,1}, 1);
% [value,t_index]=min(abs(tint{1,1}(1,:)-1));
% kk_list = 1:step*5:t_index;
% 
% for kk = kk_list
%     for n = 1:nrow
%         for m = 1:ncol
% 
%             set(h_leg1(n,m), 'xdata', xScale * ptisx{n,m}(kk,[1,2]), ...
%                         'ydata', xScale * ptisy{n,m}(kk,[1,2]));
%             set(h_body1(n,m), 'xdata', xScale * ptisx{n,m}(kk,[3,2,4,5]), ...
%                          'ydata', xScale * ptisy{n,m}(kk,[3,2,4,5]));
%             set(h_body2(n,m), 'xdata', xScale * ptisx{n,m}(kk,[5,6,7]), ...
%                          'ydata', xScale * ptisy{n,m}(kk,[5,6,7]));
%             set(h_body3(n,m), 'xdata', xScale * ptisx{n,m}(kk,[7,8,9]), ...
%                          'ydata', xScale * ptisy{n,m}(kk,[7,8,9]));
%             set(h_body4(n,m), 'xdata', xScale * ptisx{n,m}(kk,[9,10,12,11]), ...
%                          'ydata', xScale * ptisy{n,m}(kk,[9,10,12,11]));
%             set(h_leg2(n,m), 'xdata', xScale * ptisx{n,m}(kk,[12,13]), ...
%                         'ydata', xScale * ptisy{n,m}(kk,[12,13]));
% 
%             set(hF1(n,m), 'xdata', xScale * ptisx{n,m}(kk,1), ...
%                      'ydata', xScale * ptisy{n,m}(kk,1), ...
%                      'udata', F1isx{n,m}(kk) * Fscale * xScale, ...
%                      'vdata', F1isy{n,m}(kk) * Fscale * xScale);
%             set(hF2(n,m), 'xdata', xScale * ptisx{n,m}(kk,13), ...
%                      'ydata', xScale * ptisy{n,m}(kk,13), ...
%                      'udata', F2isx{n,m}(kk) * Fscale * xScale, ...
%                      'vdata', F2isy{n,m}(kk) * Fscale * xScale);
% 
%             set(h_title(n,m), 'String', ...
%                 sprintf(title_format2, ...
%                         para.variable1(n), para.variable2(m), tint{n,m}(kk)*1000));
%             set(h_gnd(n,m), 'xdata', xlim_val);
%             % camera & ground 업데이트
%             xlim_val = xScale * (cgis{n,m}(kk,1) + [2*(-para.body.length3-para.body.length4), 2*(para.body.length1+para.body.length2)]);
%             % xlim_val_floored = floor(xlim_val(1) * 100) / 100;
%             % xlim_val_ceiled = ceil(xlim_val(2) * 100) / 100;
%             % xticks(xlim_val_floored:ceil(diff(xlim_val)/10):xlim_val_ceiled); % 예시: xlim 범위 내에서 대략 5개의 정수 눈금 생성
%             set(h_gnd(n,m), 'xdata', xlim_val);      
% 
%         end
%     end
% 
%     drawnow limitrate;
%     frame = getframe(tt);
%     frameResized = imresize(frame.cdata, [1080, 1920]);  % Full HD 강제 고정
%     writeVideo(vid, im2frame(frameResized));
% end
% 
% close(vid);
% fprintf('영상 저장 완료: %s\n', videoFile);

%% ======================================
%% Time X-Y Each , Entire & Magnified with Voltage

tt=figure(1000);

gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=1:1:length(parameter.variable1)
    for jj=para.vcol_index
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,jj};
        xout = entire_cgxy{ii,jj};
        plot(tout,xout(:,1),'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,2))
        hold on
        plot(tout,xout(:,2),'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 1),'Color',getColor(colors, 'area', ii,1))
        hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, parameter.tend, 0, 0.8,  0, 0, -0.025, 0,    parameter.tend*2, 4, 'off');
xlabel('Time (s)')
ylabel('Trajectory (m)')
filename=['Distance-Time-XY-1 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end), parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

% Time X-Y Each
tt=figure(1001);
gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=1:1:length(parameter.variable1)
    for jj=para.vcol_index
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,jj};
        xout = entire_cgxy{ii,jj};
        plot(tout,xout(:,1)*1e3,'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,2))
        hold on
        plot(tout,xout(:,2)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 1),'Color',getColor(colors, 'area', ii,1))
        hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, parameter.tend, 0, 6,  0, 0, 0, 0,    parameter.tend*2, 3, 'off');
xlabel('Time (s)')
ylabel('Trajectory (mm)')
filename=['Distance(mag)-Time-XY-1 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end), parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

% Time X' & Y' Each, with voltage
tt=figure(1002);
gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=1:1:length(parameter.variable1)
    for jj=para.vcol_index
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,jj};
        plot(tout,vpins{ii,jj}(:,2)*1e3,'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,2))
        hold on
        % plot(tout,xout(:,8)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,3))
        % hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.77,    30,    0, parameter.tend, 0, 1000,  0, 0, -50, 100,    parameter.tend*2, 5, 'off');
xlabel('Time (s)')
ylabel('Velocity (mm/s)')
filename=['Velocity-Time-XY-1 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end), parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

% Voltage X-Y
tt=figure(1003);

gr.mainLinewidth=0.5;
gr.MarkSize=1;
colors = initializeColors();
timegap=2;
for ii=1:1:length(parameter.variable1)
    for jj=para.vcol_index
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,jj};
        xout = entire_cgxy{ii,jj};
        plot(xout(:,1),xout(:,2)*1e3+(timegap)*(ii-1),'-','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,2))
        hold on
        % plot(tout,xout(:,2)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,3))
        % hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, 0.8, 0, 10,  0, 0, 0, 0,    4, 4, 'off');
% customizePlot(1400, 1000, 150, 135, 0.82, 0.75,    30,    0.2, 0.21, 0, 10,  0, 0, 0, 0,    4, 5, 'off');
xlabel('X-Trajectory (m)')
ylabel('Y-Trajectory (mm)')
filename=['Trajectory-Space-XY-1 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end), parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));
% 'vpins_max','vymax_steady','vxmax_steady'

%% ==========================================
%% Time X-Y Each , Entire & Magnified with Damping

% Time X-Y entire
tt=figure(1010);
gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=para.vrow_index
    for jj=length(parameter.variable2):-1:1
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,length(parameter.variable2)-(jj-1)};
        xout = entire_cgxy{ii,length(parameter.variable2)-(jj-1)};
        plot(tout,xout(:,1),'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 3),'Color',getColor(colors, 'area', jj,2))
        hold on
        plot(tout,xout(:,2),'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 1),'Color',getColor(colors, 'area', jj,1))
        hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, parameter.tend, 0, 0.8,  0, 0, -0.025, 0,    parameter.tend*2, 4, 'off');
xlabel('Time (s)')
ylabel('Trajectory (m)')
filename=['Distance-Time-XY-2 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end),parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

% Time X-Y magnified
tt=figure(1011);
gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=para.vrow_index
    for jj=length(parameter.variable2):-1:1
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,length(parameter.variable2)-(jj-1)};
        xout = entire_cgxy{ii,length(parameter.variable2)-(jj-1)};
        plot(tout,xout(:,1)*1e3,'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 3),'Color',getColor(colors, 'area', jj,2))
        hold on
        plot(tout,xout(:,2)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 1),'Color',getColor(colors, 'area', jj,1))
        hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, parameter.tend, 0, 4,  0, 0.05, 0, 0,    parameter.tend*2, 4, 'off');
xlabel('Time (s)')
ylabel('Trajectory (mm)')
filename=['Distance(mag)-Time-XY-2 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end),parameter.variable2(1)*normunit, parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

% Time X' & Y' Each, with damping
tt=figure(1012);
gr.mainLinewidth=0.5;
gr.MarkSize=2;
colors = initializeColors();
for ii=para.vrow_index
    for jj=length(parameter.variable2):-1:1
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,length(parameter.variable2)-(jj-1)};
        plot(tout,vpins{ii,length(parameter.variable2)-(jj-1)}(:,2)*1e3,'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 3),'Color',getColor(colors, 'area', jj,2))
        hold on
        % plot(tout,xout(:,8)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,3))
        % hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.77,    30,    0, parameter.tend, 0, 1000,  0, 0, -50, 100,    parameter.tend*2, 5, 'off');
xlabel('Time (s)')
ylabel('Velocity (mm/s)')
filename=['Velocity-Time-XY-2 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end),parameter.variable2(1)*normunit ,parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));
% Damping X-Y
tt=figure(1013);
gr.mainLinewidth=0.5;
gr.MarkSize=1;
colors = initializeColors();
timegap=2;
for ii=para.vrow_index
    for jj=length(parameter.variable2):-1:1
        % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
        tout = entire_tout{ii,length(parameter.variable2)-(jj-1)};
        xout = entire_cgxy{ii,length(parameter.variable2)-(jj-1)};
        plot(xout(:,1),xout(:,2)*1e3+(timegap)*(jj-1),'-','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', jj, 3),'Color',getColor(colors, 'area', jj,2))
        hold on
        % plot(tout,xout(:,2)*1e3,'s--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 3),'Color',getColor(colors, 'area', ii,3))
        % hold on 
    end
end

% function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
customizePlot(1400, 1000, 150, 135, 0.86, 0.75,    30,    0, 0.8, 0, 10,  0, 0, 0, 0,    4, 4, 'off');
% customizePlot(1400, 1000, 150, 135, 0.82, 0.75,    30,    0.2, 0.21, 0, 10,  0, 0, 0, 0,    4, 5, 'off');
xlabel('X-Trajectory (m)')
ylabel('Y-Trajectory (mm)')
filename=['Trajectory-Space-XY-2 LegF_',num2str(parameter.leg.theta_front_deg(1)),'_Legh_',num2str(parameter.leg.theta_hind_deg(1)),'_',sprintf(title_format, parameter.variable1(1),parameter.variable1(end),parameter.variable2(1)*normunit,parameter.variable2(end)*normunit ) ,'.fig'];
savefig(tt, fullfile(filepath, filename));

%%
% tt=figure(1002);
% nrow = length(para.variable1);
% ncol = length(para.variable2);
% gr.mainLinewidth=0.5;
% gr.MarkSize=10;
% colors = initializeColors();
% for ii=1:1:length(para.variable1)
%     for jj=1:1
%         % axes(ha(jj+nrow*(ii-1)))  % jj번째 subplot에 접근
%         tout = para.trajectories{ii,jj,1};
%         xout = para.trajectories{ii,jj,2};
%         fs = 1 / mean(diff(tout));  % 샘플링 주파수
%         % 1주기 = 1/360초 -> 그에 해당하는 샘플 수
%         period_in_samples = round(fs / 360);  % 예: fs=1000이면 약 2.78 ≈ 3
%         velocity_ma10 = movmean(xout(:,7), period_in_samples*10);  
%         % plot(tout,xout(:,7),'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 1),'Color',getColor(colors, 'area', ii,1))
%         % hold on
%         plot(tout,velocity_ma10,'o--','LineWidth',gr.mainLinewidth,'MarkerSize', gr.MarkSize,'MarkerFaceColor',getColor(colors, 'area', ii, 1),'Color',getColor(colors, 'area', ii,1))
%         hold on
%     end
% end
% 
% % function customizePlot(extwidth, extheight, Xstartpixel, Ystartpixel, intwidthratio, intheightratio, fontsize, xstart, xend, ystart, yend, xsadd, xeadd, ysadd, yeadd, divisionx, divisiony, gridOn)
% customizePlot(1400, 1000, 150, 135, 0.82, 0.75,    30,    0, para.tend-1, 0, 1,  0, 0, 0, 0.1,    4, 4, 'off');
% xlabel('Time (s)')
% ylabel('Velocity (m/s)')
%%
% % 기본 변수
% time = tout;
% velocity = xout(:,7);
% fs = 1 / mean(diff(time));  % 샘플링 주파수
% 
% % 1주기 = 1/360초 -> 그에 해당하는 샘플 수
% period_in_samples = round(fs / 360);  % 예: fs=1000이면 약 2.78 ≈ 3
% 
% % Moving average 적용
% velocity_ma = movmean(velocity, period_in_samples);  
% velocity_ma10 = movmean(velocity, period_in_samples*10);  
% % 시각화
% figure;
% plot(time, velocity, 'b-', 'DisplayName', 'Raw velocity'); hold on;
% plot(time, velocity_ma, 'r-', 'LineWidth', 2, 'DisplayName', 'Moving Average'); hold on
% plot(time, velocity_ma10, 'g-', 'LineWidth', 2, 'DisplayName', 'Moving Average');
% legend();
% xlabel('Time (s)');
% ylabel('Velocity (m/s)');
% title('Velocity with Moving Average (360Hz window)');

%%
% 시간 및 속도 데이터
% time = tout;               % 예: 0 ~ 2초까지
% velocity = xout(:,7);      % 속도 데이터
% 
% % Sampling time (assumes uniform sampling)
% difft = mean(diff(time));
% 
% % --- 1. 주파수 분석 (FFT를 통해 주기 추정) ---
% L = length(velocity);
% Y = fft(velocity - mean(velocity));  % remove DC
% P2 = abs(Y/L);
% P1 = P2(1:floor(L/2)+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = (0:floor(L/2))/L/difft;
% 
% % 피크 주파수 찾기
% [~, idx] = max(P1(2:end));  % DC 제외
% dominantFreq = f(idx+1);    % dominant frequency [Hz]
% T_period = 1/dominantFreq;         % 주기 [s]
% 
% fprintf('Dominant Frequency: %.2f Hz, Period: %.3f s\n', dominantFreq, T_period);
% 
% % --- 2. 주기별 구간으로 분할 ---
% periods = floor(time(end) / T_period);
% max_per_period = zeros(periods,1);
% mean_per_period = zeros(periods,1);
% 
% for kkk = 1:periods
%     t_start = (kkk-1)*T_period;
%     t_end = kkk*T_period;
%     idx_period = (time >= t_start) & (time < t_end);
% 
%     if any(idx_period)
%         vel_segment = velocity(idx_period);
%         max_per_period(kkk) = max(vel_segment);
%         mean_per_period(kkk) = mean(vel_segment);
%     end
% end
% 
% % --- 3. 시각화 ---
% figure;
% subplot(2,1,1);
% plot(1:periods, max_per_period, 'r-o');
% title('Maximum Velocity per Period');
% xlabel('Period Index');
% ylabel('Max Velocity (m/s)');
% 
% subplot(2,1,2);
% plot(1:periods, mean_per_period, 'b-o');
% title('Mean Velocity per Period');
% xlabel('Period Index');
% ylabel('Mean Velocity (m/s)');
