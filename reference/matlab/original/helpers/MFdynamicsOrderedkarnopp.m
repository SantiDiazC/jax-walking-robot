% function xdot = MFdynamicsOrdered(t,X,M,F,v,varargin)
% % Justin Yim - 2018
% 
% if nargin >= 6
%     f_concatenated = varargin{1};
%     concatenated = f_concatenated(t,X);
% else
%     concatenated = [];
% end

function dX = MFdynamicsOrderedkarnopp(t, X, fM, fF, fv, fs1dot, fs2dot)
    % X: 14x1 = [q(6); v(6); s1; s2]
    % q = [x y th1 th2 th3 th4], v = [vx vy w1 w2 w3 w4]

    % 1) 행렬/벡터 평가
    % fM/fF를 14상태용으로 만들었으면 그대로 X를 넘기고,
    % 12상태(X12)로 만들었다면 X(1:12)를 넘기세요. (아래 방식이 안전)
    M = fM(t, X(1:12));    % M은 s에 무관
    F = fF(t, X);          % F는 마찰에 s가 들어가므로 14상태 전달

    % 2) 선형 시스템 풀기: z = [a_gen(6); lambda(6)]
    z = M \ F;
    a_gen = z(1:6);        % ★ 여기서 '앞 6개'만 취함 = [ax ay alpha1 alpha2 alpha3 alpha4]^T

    % 3) 속도(= 위치 미분)
    v = fv(t, X);          % 보통 v = X(7:12)

    % 4) 내부상태 미분
    ds1 = fs1dot(t, X);
    ds2 = fs2dot(t, X);

    % 5) 조립: 길이 14로 맞춤
    dX = [v; a_gen; ds1; ds2];
end

