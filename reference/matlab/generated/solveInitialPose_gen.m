function X0 = solveInitialPose_gen(P)
%SOLVEINITIALPOSE_GEN Robust initial pose solver for the 6-link planar robot.
% Returns X0 = [x;y;th1;th2;th3;th4; vx;vy;w1;w2;w3;w4; s1;s2]
%
% IMPORTANT: P is 42x1 (1-based indexing). Mapping aligns with Python:
%   Python P[22] = freq   -> MATLAB P(23)
%   Python P[23:26] = theta1,theta2,theta3 -> MATLAB P(24:26)

    % --- Unpack geometry (MATLAB is 1-based)
    l1=P(10); l2=P(11); l3=P(12); l4=P(13);
    lleg1=P(14); lleg2=P(15);
    al1=P(16); al2=P(17);
    h1=P(18); h2=P(19);

    % --- Desired inter-link angle increments (THIS WAS OFF-BY-ONE BEFORE)
    theta1_init = P(24);
    theta2_init = P(25);
    theta3_init = P(26);

    % --- Initial guess (meters, radians)
    x0_guess = [0; 0.003; 0; theta1_init; theta1_init+theta2_init; theta1_init+theta2_init+theta3_init];

    % --- Bounds (keep angles in a reasonable range to avoid multi-turn solutions)
    lb = [-0.02; 0.0005; -pi; -pi; -pi; -pi];
    ub = [ 0.02; 0.01;    pi;  pi;  pi;  pi];

    options = optimoptions("lsqnonlin", ...
        "Display", "off", ...
        "Algorithm", "levenberg-marquardt", ...
        "MaxIterations", 200, ...
        "FunctionTolerance", 1e-12, ...
        "StepTolerance", 1e-12);

    z = lsqnonlin(@(z) cost_fn(z, l1,l2,l3,l4,lleg1,lleg2,al1,al2,h1,h2,theta1_init,theta2_init,theta3_init), ...
                  x0_guess, lb, ub, options);

    X0 = [z; zeros(8,1)];
end

function F = cost_fn(z, l1,l2,l3,l4,lleg1,lleg2,al1,al2,h1,h2,t1i,t2i,t3i)
    x=z(1); y=z(2); th1=z(3); th2=z(4); th3=z(5); th4=z(6);

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

    % Constraints:
    % - feet on ground (y=0)
    % - foot1 x near 0 (sets reference)
    % - enforce link-angle increments
    F = [
        foot1(2);
        foot2(2);
        foot1(1);
        (th2 - th1) - t1i;
        (th3 - th2) - t2i;
        (th4 - th3) - t3i
    ];
end
