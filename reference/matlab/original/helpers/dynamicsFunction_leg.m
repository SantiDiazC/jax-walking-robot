function fnh = dynamicsFunction_leg(expr, states, funcName)
% Turns expressions expr in terms of states and time into a function handle
% of time and the state vector;
% INPUTS
% expr: symbolic expressions
% states: cell array of states in order (symbolic variables or characters)
% funcName: 파일 이름으로 사용할 문자열
% OUTPUTS
% fnh: function handle taking (t,X) as input

warning('off','symbolic:sym:sym:DeprecateExpressions')
warning('off','symbolic:mupadmex:CallbackError')
warning('off','symbolic:generate:FunctionNotVerifiedToBeValid')

n_states = length(states);
X_sym = sym('X',[1, n_states]);
Xc = cell(1, n_states);
for ii = 1:n_states; Xc{ii} = X_sym(ii); end;

dynState = subs(expr, states, Xc);
dynString1 = char(dynState);
dynString2 = regexprep(dynString1,'X([0-9]+)','X\($1\)');
dynSym = str2sym(dynString2);

% piecewise 함수 처리를 위해 File 옵션 추가
filename = [funcName, '.m'];
options.File = filename;
options.Vars = {'t', 'X'};
matlabFunction(dynSym, options);
fnh = str2func(funcName); % 생성된 파일의 함수 핸들 반환
end