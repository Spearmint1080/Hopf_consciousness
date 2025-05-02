


%%%% Revised deco model........................ for BOLD signal....

clc
clear all

subj_id= 3;
%[BOLD,EEG]= BOLD_fMRI_revised(subj_id);

%% BOLD 

% fmri= load("empirical_fMRI.mat");
ts_fmri0= fmri.fMRI;
ts_fmri1= struct2cell(ts_fmri0); 

%ts_mri= zeros(668,68,15);

for subj= 1:15
ts_fmri(:,:,subj)= ts_fmri1{1,1,subj}; 
end


first_subj_fmri= ts_fmri(:,:,subj_id);

first_subj_fmri= squeeze(first_subj_fmri);


%dt= 0.01;
N= size(first_subj_fmri,1);
nChannels= 68;

%for i =1:nChannels 
   % signalsrest1(:,i) = first_subj_fmri(:,i) - mean(first_subj_fmri(:,i));
    %signalsrest1(:,i) = first_subj_fmri(:,i)./max(abs(first_subj_fmri(:,i)));
               
%end

filtered_signals1=  first_subj_fmri.';nChannels= 68;
% scaling %

for i =1:nChannels 
    filtered_signals1(i,:) = filtered_signals1(i,:) - mean(filtered_signals1(i,:));
    filtered_signals1(i,:) = filtered_signals1(i,:)/max(abs(filtered_signals1(i,:)));
               
end

mi= filtered_signals1;
BOLD=mi(:,5:100);




%%
p=load ('SCs.mat');
p= p.SC;
pd= struct2cell(p);
SC_all= zeros(68,68,15);
%pd1= struct2array(pd{1,1,1});

for subj= 1:15
    pd1= struct2array(pd{1,1,subj});

    SC_all(:,:,subj)= pd1;
end
A_emp= SC_all(:,:,subj_id);


C= A_emp;

 su= find(triu(ones(68),1));

dd= zeros(68,68);

dd(su)=C(su);
dd= dd+dd.';
A_emp=dd;
%%


%% load data and basic variables


 %Structural connectivity matrix NodesxNodes
A= A_emp;

NSUB=1; %Number of subjects
 BOLD= BOLD(1:68,:);
dt = 0.1; 
fs = 1/dt;
sizedata = size(BOLD);
N = sizedata(2);

nchannels = 68 ;



    subj= 1;
ldata=1;%In this version we run the model only for one subject
 %structural connectivity
Tmax= N ;%Number of timepoints
wG=[0:0.1:10]; %Range of values of g (depends on the data)
ap=0.0005;%a values
idxSub=1;%
%rng('shuffle');
nNodes = length(C);
nSubs = ldata; 
si = 1:ldata; 
nWeights = numel(wG);

%% Details of the model

Cfg.simulID = 'coma1';
Cfg.opt_a.nIters = 200;%Number of iterations for initial optimization
Cfg.opt_a.updateStrength = 0.1;%Update strength for a optimization
Cfg.opt_a.abortCrit = 0.1; %maximally 2.5% error
Cfg.opt_a.gref = 1;%This is reference g, for the initial optimization only. Ignore if not using newOpt


if ~isfield(Cfg, 'simulID'), Cfg.simulID = 'Unknown'; else end;

if ~isfield(Cfg, 'TRsec'), Cfg.TRsec = 2; else end;

if ~isfield(Cfg, 'opt_a'), Cfg.opt_a = []; else end;
if ~isfield(Cfg.opt_a, 'nIters'), Cfg.opt_a.nIters = 100; else end;
if ~isfield(Cfg.opt_a, 'updateStrength'), Cfg.opt_a.updateStrength = 0.1; else end
if ~isfield(Cfg.opt_a, 'abortCrit'), Cfg.opt_a.abortCrit = 0.1; else end
if ~isfield(Cfg, 'plots'), Cfg.plots = []; else end



%% Pre-modelling

%In this section, the calculation of the FC, power spectrum, omega and
%other variables that will be used in the model simulation.

fprintf(1, 'Fitting models for %d different weights\n', nWeights);
FC_simul = zeros(nNodes, nNodes, nWeights);
fitting = zeros(1, nWeights);
meta = zeros(1, nWeights);
ksP = zeros(1, nWeights);
Phases = zeros(nNodes, Tmax, nSubs, nWeights);
bifpar = zeros(nWeights, nNodes);

%--------------------------------------------------------------------------
%CALCULATE FUNCTIONAL CONNECTIVITY MATRIX
%--------------------------------------------------------------------------
r = zeros(nNodes, nNodes, nSubs);
ts = zeros(nNodes, Tmax, nSubs);

ts_all= zeros(1,68,Tmax); 
ts_all(1,:,:)= BOLD(:,:);

%i=1;
%ts(:,:,1) = squeeze(ts_all(subj,:,:));
r(:,:,1) = corrcoef(BOLD.');
FC_emp= r;

%FC_emp=mean(r,3);  %%%%mean of subjects....
C=C/max(max(C))*0.2;%


%--------------------------------------------------------------------------
%COMPUTE POWER SPECTRA FOR
%NARROWLY FILTERED DATA WITH LOW BANDPASS (0.04 to 0.07 Hz)
%WIDELY FILTERED DATA (0.04 Hz to justBelowNyquistFrequency)
%[justBelowNyquistFrequency depends on TR,
%for a TR of 2s this is 0.249 Hz]
%--------------------------------------------------------------------------
TT=Tmax;
Ts = TT*Cfg.TRsec;
freq = (0:TT/2-1)/Ts;
[~, idxMinFreq] = min(abs(freq-0.04));
[~, idxMaxFreq] = min(abs(freq-0.07));
nFreqs = length(freq);

delt = 2;                                   % sampling interval (TR)
fnq = 1/(2*delt);                           % Nyquist frequency
k = 2;                                      % 2nd order butterworth filter

%WIDE BANDPASS
flp = .04;                                  % lowpass frequency of filter
fhi = fnq-0.001;                      % highpass needs to be limited by Nyquist frequency, which in turn depends on TR
Wn = [flp/fnq fhi/fnq];                     % butterworth bandpass non-dimensional frequency
[bfilt_wide, afilt_wide] = butter(k,Wn);    % construct the filter

%NARROW LOW BANDPASS
flp = .04;                                  % lowpass frequency of filter
fhi = .07;                                  % highpass
Wn=[flp/fnq fhi/fnq];                       % butterworth bandpass non-dimensional frequency
[bfilt_narrow,afilt_narrow] = butter(k,Wn); % construct the filter


PowSpect_filt_narrow = zeros(nFreqs, nNodes, nSubs);
PowSpect_filt_wide = zeros(nFreqs, nNodes, nSubs);
for seed=1:nNodes
        %idxSub=1;
        signaldata = squeeze(ts_all(subj,:,:));
        x=detrend(demean(signaldata(seed,:)));
        
        ts_filt_narrow =zscore(filtfilt(bfilt_narrow,afilt_narrow,x));
        pw_filt_narrow = abs(fft(ts_filt_narrow));
        PowSpect_filt_narrow(:,seed,1) = pw_filt_narrow(1:floor(TT/2)).^2/(TT/2);
        
        ts_filt_wide =zscore(filtfilt(bfilt_wide,afilt_wide,x));
        pw_filt_wide = abs(fft(ts_filt_wide));
        PowSpect_filt_wide(:,seed,1) = pw_filt_wide(1:floor(TT/2)).^2/(TT/2);
    
end

Power_Areas_filt_narrow_unsmoothed = mean(PowSpect_filt_narrow,3);
Power_Areas_filt_wide_unsmoothed = mean(PowSpect_filt_wide,3);
Power_Areas_filt_narrow_smoothed = zeros(nFreqs, nNodes);
Power_Areas_filt_wide_smoothed = zeros(nFreqs, nNodes);
vsig = zeros(1, nNodes);
for seed=1:nNodes
    Power_Areas_filt_narrow_smoothed(:,seed)=gaussfilt(freq,Power_Areas_filt_narrow_unsmoothed(:,seed)',0.01);
    Power_Areas_filt_wide_smoothed(:,seed)=gaussfilt(freq,Power_Areas_filt_wide_unsmoothed(:,seed)',0.01);
    %relative power in frequencies of interest (.04 - .07 Hz) with respect
    %to entire power of bandpass-filtered data (.04 - just_below_nyquist)
    vsig(seed) =...
        sum(Power_Areas_filt_wide_smoothed(idxMinFreq:idxMaxFreq,seed))/sum(Power_Areas_filt_wide_smoothed(:,seed));
end
vmax=max(vsig); %consider computing this later where needed
vmin=min(vsig);%consider computing this later where needed

%a-minimization seems to only work if we use the indices for frequency of
%maximal power from the narrowband-smoothed data
[~, idxFreqOfMaxPwr]=max(Power_Areas_filt_narrow_smoothed);
f_diff = freq(idxFreqOfMaxPwr);

%FOR EACH AREA AND TIMEPOINT COMPUTE THE INSTANTANEOUS PHASE IN THE RANGE
%OF .04 TO .09 Hz
PhasesD = zeros(nNodes, Tmax, nSubs);

signaldata=squeeze(ts_all);
for seed=1:nNodes
    x = demean(detrend(signaldata(seed,:)));
    xFilt = filtfilt(bfilt_narrow,afilt_narrow,x);    % zero phase filter the data
    Xanalytic = hilbert(demean(xFilt));
    PhasesD(seed,:,1) = angle(Xanalytic);
end 


%f_diff  previously computed frequency with maximal power (of narrowly filtered data) by area
omega = repmat(2*pi*f_diff',1,2); %angular velocity
omega(:,1) = -omega(:,1);


%% FROM HERE ON SIMULATIONS AND FITTING

dt = 0.1;
sig = 0.04; 
dsig = sqrt(dt)*sig;
mu= 0.0005;


a = repmat(mu.*ones(nNodes,1),1,2);
a1 = a; %Starting values of a set to ~0
trackminm1 = zeros(Cfg.opt_a.nIters, nWeights); %for tracking the minimization (good for debugging)

for trial=1:100 %number of runs of the model

for idx_g = 1:numel(wG)
we = wG(idx_g);

fprintf(1, '-----------------------------------------\n');
fprintf(1, 'g(%d/%d) = %5.3f\n', idx_g, numel(wG), we);
fprintf(1, '-----------------------------------------\n');

xs = zeros(3000/2,nNodes);
wC = we*C; %structural connectivity matrix weighted with current global coupling parameter
sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj 
fprintf(1, 'SIMULATING HOMOGENEOUS MODEL.\n');
a=a1; %use those avalues that have been found to be optimal
bifpar(idx_g,:)=a(:,1)';%store them in an output variable
xs=zeros(Tmax*nSubs,nNodes);
ys=zeros(Tmax*nSubs,nNodes);
z = 0.1*ones(nNodes,2); % --> x = z(:,1), y = z(:,2)
nn=0;
    for t=1:dt:3000 %This part is to initialize the model and to warm the timeseries
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
    end
    
    for t=1:dt:Tmax*Cfg.TRsec*nSubs %check for the specific data
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
        
        if mod(t,2)==0
            nn=nn+1;
            xs(nn,:)=z(:,1)'; 
            ys(nn,:)=z(:,1)'; %%%will be used for hidden layer .....
        end
    end
    
    fprintf(1, 'COMPUTING MODEL FIT.\n');
    FC_simul(:, :, idx_g) = corrcoef(xs(1:nn,:)); 
    cc=corrcoef(squareform(tril(FC_emp,-1)),squareform(tril(FC_simul(:, :, idx_g),-1)));
    fitting(idx_g)=cc(2); %Fitting of the FC
    
    %%%%%%%%%%%%%%%%%%

    tini=(1-1)*Tmax;
    for seed=1:nNodes
        ts_simul = detrend(demean(xs(tini+1:tini+Tmax,seed)'));
        ts_simul_filt_narrow = filtfilt(bfilt_narrow,afilt_narrow,ts_simul);
        Xanalytic = hilbert(ts_simul_filt_narrow);
        Phases(seed,:,1, idx_g) = angle(Xanalytic);
    end
%This part has to be uncomment in case the fitting is also compared with
%the metastability.

%     metastability22 = zeros(1, nSubs);
%     T=1:Tmax;
%     sync = zeros(1, numel(T));
%     for t=T
%         ku=sum(complex(cos(Phases(:,t,idxSub, idx_g)),sin(Phases(:,t,idxSub, idx_g))))/nNodes;
%         sync(t, idx_g)=abs(ku);
%     end
% 
%     metastability22(1)=std(sync(:, idx_g));
%     meta(idx_g)=mean(metastability22);
    pcD(:,1)=patternCons30(PhasesD(:,:),nNodes,Tmax); %empirical FCD
    pcS(:,1)=patternCons30(Phases(:,:,1, idx_g),nNodes,Tmax); % simulated FCD 

    [~, ~, ksP(idx_g)]=kstest2(pcS(:),pcD(:));

    fprintf(1, 'DONE.\n\n');
end

tmnp(trial)=find(fitting==max(fitting));
XG(trial)= max(fitting);
end




%%   EEG..



clc;
clear all
[BOLD,EEG]= BOLD_fMRI_revised(3);
subj_id=3;

p=load ('SCs.mat');
p= p.SC;
pd= struct2cell(p);
SC_all= zeros(68,68,15);
%pd1= struct2array(pd{1,1,1});

for subj= 1:15
    pd1= struct2array(pd{1,1,subj});

    SC_all(:,:,subj)= pd1;
end
A_emp= SC_all(:,:,subj_id);


C= A_emp;

 su= find(triu(ones(68),1));

dd= zeros(68,68);

dd(su)=C(su);
dd= dd+dd.';
A_emp=dd;
%%


%% load data and basic variables


 %Structural connectivity matrix NodesxNodes
A= A_emp;

NSUB=1; %Number of subjects
 BOLD= BOLD(1:68,:);
dt = 0.01; 
fs = 1/dt;
sizedata = size(BOLD);
N = sizedata(2);

nchannels = 68 ;



    subj= 1;
ldata=1;%In this version we run the model only for one subject
 %structural connectivity
Tmax= N ;%Number of timepoints
wG=[0:0.1:10]; %Range of values of g (depends on the data)
ap=0.0005;%a values
idxSub=1;%
%rng('shuffle');
nNodes = length(C);
nSubs = ldata; 
si = 1:ldata; 
nWeights = numel(wG);

%% Details of the model

Cfg.simulID = 'coma1';
Cfg.opt_a.nIters = 200;%Number of iterations for initial optimization
Cfg.opt_a.updateStrength = 0.1;%Update strength for a optimization
Cfg.opt_a.abortCrit = 0.1; %maximally 2.5% error
Cfg.opt_a.gref = 1;%This is reference g, for the initial optimization only. Ignore if not using newOpt


if ~isfield(Cfg, 'simulID'), Cfg.simulID = 'Unknown'; else end;

if ~isfield(Cfg, 'TRsec'), Cfg.TRsec = 2; else end;

if ~isfield(Cfg, 'opt_a'), Cfg.opt_a = []; else end;
if ~isfield(Cfg.opt_a, 'nIters'), Cfg.opt_a.nIters = 100; else end;
if ~isfield(Cfg.opt_a, 'updateStrength'), Cfg.opt_a.updateStrength = 0.1; else end
if ~isfield(Cfg.opt_a, 'abortCrit'), Cfg.opt_a.abortCrit = 0.1; else end
if ~isfield(Cfg, 'plots'), Cfg.plots = []; else end



%% Pre-modelling

%In this section, the calculation of the FC, power spectrum, omega and
%other variables that will be used in the model simulation.

fprintf(1, 'Fitting models for %d different weights\n', nWeights);
FC_simul = zeros(nNodes, nNodes, nWeights);
fitting = zeros(1, nWeights);
meta = zeros(1, nWeights);
ksP = zeros(1, nWeights);
Phases = zeros(nNodes, Tmax, nSubs, nWeights);
bifpar = zeros(nWeights, nNodes);

%--------------------------------------------------------------------------
%CALCULATE FUNCTIONAL CONNECTIVITY MATRIX
%--------------------------------------------------------------------------
r = zeros(nNodes, nNodes, nSubs);
ts = zeros(nNodes, Tmax, nSubs);

ts_all= zeros(1,68,Tmax); 
ts_all(1,:,:)= BOLD(:,:);

%i=1;
%ts(:,:,1) = squeeze(ts_all(subj,:,:));
r(:,:,1) = corrcoef(BOLD.');
FC_emp= r;

%FC_emp=mean(r,3);  %%%%mean of subjects....
C=C/max(max(C))*0.2;%


%--------------------------------------------------------------------------
%COMPUTE POWER SPECTRA FOR
%NARROWLY FILTERED DATA WITH LOW BANDPASS (0.04 to 0.07 Hz)
%WIDELY FILTERED DATA (0.04 Hz to justBelowNyquistFrequency)
%[justBelowNyquistFrequency depends on TR,
%for a TR of 2s this is 0.249 Hz]
%--------------------------------------------------------------------------
TT=Tmax;
Ts = TT*Cfg.TRsec;
freq = (0:TT/2-1)/Ts;
[~, idxMinFreq] = min(abs(freq-0.04));
[~, idxMaxFreq] = min(abs(freq-0.07));
nFreqs = length(freq);

delt = 2;                                   % sampling interval (TR)
fnq = 1/(2*delt);                           % Nyquist frequency
k = 2;                                      % 2nd order butterworth filter

%WIDE BANDPASS
flp = .04;                                  % lowpass frequency of filter
fhi = fnq-0.001;                      % highpass needs to be limited by Nyquist frequency, which in turn depends on TR
Wn = [flp/fnq fhi/fnq];                     % butterworth bandpass non-dimensional frequency
[bfilt_wide, afilt_wide] = butter(k,Wn);    % construct the filter

%NARROW LOW BANDPASS
flp = .04;                                  % lowpass frequency of filter
fhi = .07;                                  % highpass
Wn=[flp/fnq fhi/fnq];                       % butterworth bandpass non-dimensional frequency
[bfilt_narrow,afilt_narrow] = butter(k,Wn); % construct the filter


PowSpect_filt_narrow = zeros(nFreqs, nNodes, nSubs);
PowSpect_filt_wide = zeros(nFreqs, nNodes, nSubs);
for seed=1:nNodes
        %idxSub=1;
        signaldata = squeeze(ts_all(subj,:,:));
        x=detrend(demean(signaldata(seed,:)));
        
        ts_filt_narrow =zscore(filtfilt(bfilt_narrow,afilt_narrow,x));
        pw_filt_narrow = abs(fft(ts_filt_narrow));
        PowSpect_filt_narrow(:,seed,1) = pw_filt_narrow(1:floor(TT/2)).^2/(TT/2);
        
        ts_filt_wide =zscore(filtfilt(bfilt_wide,afilt_wide,x));
        pw_filt_wide = abs(fft(ts_filt_wide));
        PowSpect_filt_wide(:,seed,1) = pw_filt_wide(1:floor(TT/2)).^2/(TT/2);
    
end

Power_Areas_filt_narrow_unsmoothed = mean(PowSpect_filt_narrow,3);
Power_Areas_filt_wide_unsmoothed = mean(PowSpect_filt_wide,3);
Power_Areas_filt_narrow_smoothed = zeros(nFreqs, nNodes);
Power_Areas_filt_wide_smoothed = zeros(nFreqs, nNodes);
vsig = zeros(1, nNodes);
for seed=1:nNodes
    Power_Areas_filt_narrow_smoothed(:,seed)=gaussfilt(freq,Power_Areas_filt_narrow_unsmoothed(:,seed)',0.01);
    Power_Areas_filt_wide_smoothed(:,seed)=gaussfilt(freq,Power_Areas_filt_wide_unsmoothed(:,seed)',0.01);
    %relative power in frequencies of interest (.04 - .07 Hz) with respect
    %to entire power of bandpass-filtered data (.04 - just_below_nyquist)
    vsig(seed) =...
        sum(Power_Areas_filt_wide_smoothed(idxMinFreq:idxMaxFreq,seed))/sum(Power_Areas_filt_wide_smoothed(:,seed));
end
vmax=max(vsig); %consider computing this later where needed
vmin=min(vsig);%consider computing this later where needed

%a-minimization seems to only work if we use the indices for frequency of
%maximal power from the narrowband-smoothed data
[~, idxFreqOfMaxPwr]=max(Power_Areas_filt_narrow_smoothed);
f_diff = freq(idxFreqOfMaxPwr);

%FOR EACH AREA AND TIMEPOINT COMPUTE THE INSTANTANEOUS PHASE IN THE RANGE
%OF .04 TO .09 Hz
PhasesD = zeros(nNodes, Tmax, nSubs);

signaldata=squeeze(ts_all);
for seed=1:nNodes
    x = demean(detrend(signaldata(seed,:)));
    xFilt = filtfilt(bfilt_narrow,afilt_narrow,x);    % zero phase filter the data
    Xanalytic = hilbert(demean(xFilt));
    PhasesD(seed,:,1) = angle(Xanalytic);
end 


%f_diff  previously computed frequency with maximal power (of narrowly filtered data) by area
omega = repmat(2*pi*f_diff',1,2); %angular velocity
omega(:,1) = -omega(:,1);


%% FROM HERE ON SIMULATIONS AND FITTING

dt = 0.1;
sig = 0.04; 
dsig = sqrt(dt)*sig;
mu= 0.0005;


a = repmat(mu.*ones(nNodes,1),1,2);
a1 = a; %Starting values of a set to ~0
trackminm1 = zeros(Cfg.opt_a.nIters, nWeights); %for tracking the minimization (good for debugging)

for trial=1:100 %number of runs of the model

for idx_g = 1:numel(wG)
we = wG(idx_g);

fprintf(1, '-----------------------------------------\n');
fprintf(1, 'g(%d/%d) = %5.3f\n', idx_g, numel(wG), we);
fprintf(1, '-----------------------------------------\n');

xs = zeros(3000/2,nNodes);
wC = we*C; %structural connectivity matrix weighted with current global coupling parameter
sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj 
fprintf(1, 'SIMULATING HOMOGENEOUS MODEL.\n');
a=a1; %use those avalues that have been found to be optimal
bifpar(idx_g,:)=a(:,1)';%store them in an output variable
xs=zeros(Tmax*nSubs,nNodes);
ys=zeros(Tmax*nSubs,nNodes);
z = 0.1*ones(nNodes,2); % --> x = z(:,1), y = z(:,2)
nn=0;
    for t=1:dt:3000 %This part is to initialize the model and to warm the timeseries
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
    end
    
    for t=1:dt:Tmax*Cfg.TRsec*nSubs %check for the specific data
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
        
        if mod(t,2)==0
            nn=nn+1;
            xs(nn,:)=z(:,1)'; 
            ys(nn,:)=z(:,1)'; %%%will be used for hidden layer .....
        end
    end
    
    fprintf(1, 'COMPUTING MODEL FIT.\n');
    FC_simul(:, :, idx_g) = corrcoef(xs(1:nn,:)); 
    cc=corrcoef(squareform(tril(FC_emp,-1)),squareform(tril(FC_simul(:, :, idx_g),-1)));
    fitting(idx_g)=cc(2); %Fitting of the FC
    
    %%%%%%%%%%%%%%%%%%

    tini=(1-1)*Tmax;
    for seed=1:nNodes
        ts_simul = detrend(demean(xs(tini+1:tini+Tmax,seed)'));
        ts_simul_filt_narrow = filtfilt(bfilt_narrow,afilt_narrow,ts_simul);
        Xanalytic = hilbert(ts_simul_filt_narrow);
        Phases(seed,:,1, idx_g) = angle(Xanalytic);
    end
%This part has to be uncomment in case the fitting is also compared with
%the metastability.

%     metastability22 = zeros(1, nSubs);
%     T=1:Tmax;
%     sync = zeros(1, numel(T));
%     for t=T
%         ku=sum(complex(cos(Phases(:,t,idxSub, idx_g)),sin(Phases(:,t,idxSub, idx_g))))/nNodes;
%         sync(t, idx_g)=abs(ku);
%     end
% 
%     metastability22(1)=std(sync(:, idx_g));
%     meta(idx_g)=mean(metastability22);
    pcD(:,1)=patternCons30(PhasesD(:,:),nNodes,Tmax); %empirical FCD
    pcS(:,1)=patternCons30(Phases(:,:,1, idx_g),nNodes,Tmax); % simulated FCD 

    [~, ~, ksP(idx_g)]=kstest2(pcS(:),pcD(:));

    fprintf(1, 'DONE.\n\n');
end

tmnp(trial)=find(fitting==max(fitting));
XG(trial)= max(fitting);
end


