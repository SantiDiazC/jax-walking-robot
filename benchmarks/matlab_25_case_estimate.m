function matlab_25_case_estimate(output_file, duration_s)
% Time two full-duration cases with the original setup, helpers and ode45.
% No JAX equations, trace regularizer, GUI or trajectory/video postprocessing.
if nargin < 2, duration_s = 1.0; end
settings = [0.01, 1.0; 0.09, 50.0]; % damping ratio, adhesive thickness (um)
rows = cell(1,size(settings,1));
for k=1:size(settings,1)
    fprintf('Benchmark case %d/%d: damping %.3g, thickness %.3g um, horizon %.3g s\n', ...
        k,size(settings,1),settings(k,1),settings(k,2),duration_s);
    rows{k} = run_case(settings(k,1),settings(k,2),duration_s);
    % Save each completed case, so interrupted work is not lost.
    report = summarize(rows(1:k), duration_s);
    fid=fopen(output_file,'w'); assert(fid>=0,'Cannot open benchmark report');
    fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
    fprintf('Case finished: numerical setup %.3f s, ode45 %.3f s, completed=%d\n', ...
        rows{k}.numerical_setup_s,rows{k}.integration_s,rows{k}.completed);
end
end

function row = run_case(zeta, thickness_um, duration_s)
root=fileparts(fileparts(mfilename('fullpath')));
archive=fullfile(root,'reference','matlab','original');
addpath(fullfile(archive,'helpers'));
source=fileread(fullfile(archive,'RobotDynamics_mulbo_legdirstiff_stribeck_karnopp_iCVD_dam_thick.m'));
source=scalar_setting(source,'para.Total.zeta',zeta);
source=scalar_setting(source,'para.Body.had',thickness_um*1e-6);
folder=tempname;mkdir(folder);previous=pwd;cd(folder);
cleanup=onCleanup(@()restore(previous,folder));
shared_timer=tic;
eval(slice(source,'syms t % time','%% III: Simulation'));
shared_setup_s=toc(shared_timer);
ii=1;jj=1;para.tend=duration_s;
setup_timer=tic;tic;
% Includes geometry exports, original lsqnonlin pose, contact and RHS exports.
eval(slice(source,'params1 = ','% Run the simulation'));
numerical_setup_s=toc(setup_timer);
integration_timer=tic;
[tout,xout]=ode45(@(tt,xx)MFdynamicsOrderedkarnopp(tt,xx,fM,fF,fv,fs1dot{ii,jj},fs2dot{ii,jj}),ts,X0,options);
integration_s=toc(integration_timer);
row=struct('damping_ratio',zeta,'adhesive_thickness_um',thickness_um, ...
    'frequency_hz',para.freqs,'shared_symbolic_material_setup_s',shared_setup_s, ...
    'numerical_setup_s',numerical_setup_s,'integration_s',integration_s, ...
    'output_points',length(tout),'final_simulated_time_s',tout(end), ...
    'completed',abs(tout(end)-duration_s)<1e-10 && all(isfinite(xout(:))), ...
    'initial_state',X0.','solver','ode45','relative_tolerance',1e-6,'absolute_tolerance',1e-8);
end

function report=summarize(rows,duration_s)
cases=[rows{:}];complete=all([cases.completed]);
per_case=[cases.numerical_setup_s]+[cases.integration_s];
shared=mean([cases.shared_symbolic_material_setup_s]);
report=struct('matlab_version',version,'duration_per_case_s',duration_s,'cases',cases, ...
    'sampled_cases',length(cases),'target_case_count',25,'all_completed',complete, ...
    'estimated_25_integration_s',25*mean([cases.integration_s]), ...
    'estimated_25_setup_and_integration_s',shared+25*mean(per_case), ...
    'sample_based_low_s',shared+25*min(per_case), ...
    'sample_based_high_s',shared+25*max(per_case), ...
    'scope','Two corner cases of the original 5x5 damping/thickness grid. Original ode45, tolerances, unregularized solve and pose/drop. Estimate only, not all 25 measured. Excludes process startup, GUI, plotting and video. Sample range is not a rigorous bound.');
if ~complete
    report.estimated_25_integration_s=[];report.estimated_25_setup_and_integration_s=[];
    report.sample_based_low_s=[];report.sample_based_high_s=[];
end
end

function value=slice(source,first,last)
a=strfind(source,first);b=strfind(source,last);
assert(numel(a)==1 && numel(b)==1 && b>a,'Original source markers changed');
value=source(a:b-1);
end

function source=scalar_setting(source,name,value)
pattern=['(?m)^',regexptranslate('escape',name),'\s*=\s*[^;\r\n]+;'];
assert(~isempty(regexp(source,pattern,'once')),'Original setting not found');
source=regexprep(source,pattern,sprintf('%s = %.17g;',name,value));
end

function restore(previous,folder)
cd(previous);rmdir(folder,'s');
end
