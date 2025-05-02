%% Whole-brain modelling in the homogeneous case

%This model based on Stuard-Landau oscillators, the local dynamics of single brain
%regions using the normal form of a Hopf bifurcation.
%The dynamics of the N brain regions were coupled through the connectivity matrix,
%which was given by the connectome of healthy subjects (C).
%coupling among areas is given by the SC-> by the g scaling parameter.
%For the homogeneous case, in which we seta=0 for all nodes. 
%This choice was based on previous studies which suggest that the best fit
%to the empirical data arises at the brink of the Hopf bifurcation 
%where a~1. So, here the only free parameter is the g.
%The fitting of the model is captures using the correlation of the FC and 
%the Kolmogorov-Smirnov distance of the FCD.


%This version of the model is prepared for individualize subjects (the SC is a template).

%Ane López-González 14/07/2021

%% load data and basic variables
clear;
clc;

load empirical_paris.mat
BOLD= TS{1};

con= MAP{1};
SS4= con;
SS4(eye(160)>0)=0;
C= SS4; %Structural connectivity matrix NodesxNodes
A=C;
NSUB=1; %Number of subjects
 BOLD= BOLD(1:160,:);
dt = 0.1; 
fs = 1/dt;
sizedata = size(BOLD);
N = sizedata(2);

nchannels = 160 ;



    subj= 1;
ldata=1;%In this version we run the model only for one subject
 %structural connectivity
Tmax= N ;%Number of timepoints
wG=[4:0.1:10]; %Range of values of g (depends on the data)
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

ts_all= zeros(1,160,Tmax); 
ts_all(1,:,:)= BOLD(:,:);

%i=1;
%ts(:,:,1) = squeeze(ts_all(subj,:,:));
r(:,:,1) = corrcoef(BOLD.');
FC_emp= r;

%FC_emp=mean(r,3);  %%%%mean of subjects....
C=C/max(max(C));%


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

dt = 0.01;
sig = 0.04; 
dsig = sqrt(dt)*sig;
mu= 0.005;


a = repmat(mu.*ones(nNodes,1),1,2);
a1 = a; %Starting values of a set to ~0
trackminm1 = zeros(Cfg.opt_a.nIters, nWeights); %for tracking the minimization (good for debugging)

for trial=1 %number of runs of the model

for idx_g = 44 %1:101
we = wG(idx_g);

fprintf(1, '-----------------------------------------\n');
fprintf(1, 'g(%d/%d) = %5.3f\n', idx_g, numel(wG), we);
fprintf(1, '-----------------------------------------\n');

xs = zeros(3000/2,nNodes);
wC = we*C; %structural connectivity matrix weighted with current global coupling parameter
wC_mask = wC~=0;
sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj 
fprintf(1, 'SIMULATING HOMOGENEOUS MODEL.\n');
a=a1; %use those avalues that have been found to be optimal
bifpar(idx_g,:)=a(:,1)';%store them in an output variable
xs=zeros(Tmax*nSubs,nNodes);
ys=zeros(Tmax*nSubs,nNodes);

gamma = 0.01; %?? what to set
z = 0.1*ones(nNodes,2); % --> x = z(:,1), y = z(:,2)
W= 0.01*ones(size(wC));
nn=0;
 
    for t=1:dt:3000 %This part is to initialize the model and to warm the timeseries
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
    end

    for t=1:dt:40 %Tmax*Cfg.TRsec*nSubs %check for the specific data
         % for sum Cij*xj 
        
        % phase_z = angle(z(:,1)+1i*z(:,2));
        % phase_diff = phase - phase';
        % wC_complex = wC*exp(1i*phase_diff);
        % wC_real = real(wC_complex);
        % wC_imag = imag(wC_complex);
        % sumC_real = repmat(sum(wC_real,2),1,2);
        % sumC_imag = repmat(sum(wC_imag,2),1,2);
        sumC = repmat(sum(wC,2),1,2); 
        suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
        % suma_real = wC_real*z(:,1) - sumC_real.*z(:,1); % sum(Cij*xi) - sum(Cij)*xj
        % suma_imag = wC_imag*z(:,2) - sumC_real.*z(:,2); % sum(Cij*xi) - sum(Cij)*xj
        zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
        z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(nNodes,2);
        % z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + [suma_real,suma_imag]) + dsig*randn(nNodes,2);


        % dW_max = max(max(dt*(-wC + gamma*z*z')))
        % if isnan(W)
        %     fprintf("Time is %d\n",t)
        %     break
        % end
        dW = (dt*(-0.01*W+.005*z*z'))*wC_mask;
        % W=(W+abs(dW))/max(max(W));
         %else
        %W = clip((W+dW),0.001,0.005);
        % end
        %Wmax =  max(max(W))
        W= W+dW;
        wCD = (wC + W);
        Wc= wCD/max(max(wCD));
        %wC= 0.001*wCD;

        
        
        
        

    end
    fprintf(1, 'COMPUTING MODEL FIT.\n');
    FC_simul(:, :, idx_g) = corrcoef(xs(1:nn,:)); 
    cc=corrcoef(squareform(tril(FC_emp,-1)),squareform(tril(FC_simul(:, :, idx_g),-1)))
    fitting(idx_g)=cc(2); %Fitting of the FC
    end
    
    % fprintf(1, 'COMPUTING MODEL FIT.\n');
    % FC_simul(:, :, idx_g) = corrcoef(xs(1:nn,:)); 
    % cc=corrcoef(squareform(tril(FC_emp,-1)),squareform(tril(FC_simul(:, :, idx_g),-1)))
    % fitting(idx_g)=cc(2); %Fitting of the FC
    
    %%%%%%%%%%%%%%%%%%

    % tini=(1-1)*Tmax;
    % for seed=1:nNodes
    %     ts_simul = detrend(demean(xs(tini+1:tini+Tmax,seed)'));
    %     ts_simul_filt_narrow = filtfilt(bfilt_narrow,afilt_narrow,ts_simul);
    %     Xanalytic = hilbert(ts_simul_filt_narrow);
    %     Phases(seed,:,1, idx_g) = angle(Xanalytic);
    % end
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
    % pcD(:,1)=patternCons30(PhasesD(:,:),nNodes,Tmax); %empirical FCD
    % pcS(:,1)=patternCons30(Phases(:,:,1, idx_g),nNodes,Tmax); % simulated FCD 
    % 
    % [~, ~, ksP(idx_g)]=kstest2(pcS(:),pcD(:));

    fprintf(1, 'DONE.\n\n');
end
% end

%%%%%%after applying the hidden layer...



% for nn= 1:2

%     load empirical_paris.mat
% BOLD= TS{1};
% 
% con= MAP{1};
% SS4= con;
% SS4(eye(160)>0)=0;
% C= SS4; %Structural connectivity matrix NodesxNodes
% A=C;
% 
% kk= find(A(nn,:)>0);
% %positionneighbour= zeros(50,50);
% %positionneighbour(kk,nn)= kk; 
% %omega = 0.01 + 0.08*2*pi*rand(n,1);
% hy= numel(kk)
%     Dim= zeros(hy+1,hy+1);
%     Dim(1,1)= 0;  %%%As there is no self connection anywhere....
%     Dim(1,2:hy+1)= A(nn,kk);
%     Dim(2:hy+1,1)= A(kk,nn);
%     Dim(2:hy+1,2:hy+1)= A(kk,kk);
%     A= Dim;
%     %G = graph(A,'upper');jj= nn+2; figure(jj);plot(G)
% 
% oscneigh(nn)= hy+1 ;
% 
% ze= xs+1i.*ys;
% %ze= ze.';
% 
%  NN= [ze(:,nn),ze(:,kk)];
%         % save channelel_68   %%%take other name
% chnl= 1; %number of channel
% 
% Kbs = hy+1;    %input dimension
% dip = NN.'; %input array
% 
%  %%% learning parameter %%%
% etaWosr = 0.001;
% etaWosi = 0.001;
% 
% chnl= 1;
% % Yd = s_store;   % desired signal
%    Yd= signaldata'; 
%    Yd= Yd(:,nn);
%    %for i = 1:chnl
%     %Yd(:,i) = Yd(:,i) - mean(Yd(:,i));
%     %Yd(:,i) = Yd(:,i)/max(abs(Yd(:,i)));
%    %end
% 
% 
% %% network arceteure
% hln= 50; % hidden layer neuron
% W1bsr = rand(hln,Kbs);
% W1bsi = rand(hln,Kbs);
% W1bs = W1bsr + 1i*W1bsi;
% 
% 
% ohl=chnl; % o/p dimension
% W2bsr=rand(ohl,hln);
% W2bsi=rand(ohl,hln);
% W2bs=W2bsr+1i*W2bsi;
% ahbs = 0.5;
% aobs=0.5;
% etaW1bs = 0.001;
% etaW2bs = 0.001;
% 
%  nepochsbs = 300000; %no of epochs
%  RMSE=zeros(chnl,nepochsbs);
%  error1 = zeros(1,nepochsbs);
% for nep1 = 1:nepochsbs
%     nep1    
% % Foreard propagation %%%
%         nhbs = W1bs*dip;   % 
%         nhbsr = real(nhbs); nhbsi = imag(nhbs);
%         xhbsr = 2*sigmf(nhbsr,[ahbs,0]) - 1;
%         xhbsi = 2*sigmf(nhbsi,[ahbs,0]) - 1;
%         xhbs = xhbsr + 1i*xhbsi;      
% 
%        nobs = W2bs*xhbs;
% 
%         nobsr = real(nobs);
%         nobsi = imag(nobs);
%         ybsr=nobsr;
%         ybsi=nobsi;
%         ybsr = 2*sigmf(nobsr,[aobs,0]) - 1;
%         ybsi = 2*sigmf(nobsi,[aobs,0]) - 1;
%         ybs = ybsr + 1i*ybsi;   
% ybs = ybsr ;  
% ybs=ybs';
% % error1(nep1) = error1(nep1) + norm(Yd' - ybs);   
% for ii=1:chnl
%     RMSE(ii,nep1)=sqrt(mean(((Yd(:,ii) - ybs(:,ii)).^2)));
%     end
% 
% % backpropagnation
% 
% dW2bsr = (-1)*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsr';
% 
% dW2bsi = ((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsi';                
% 
% 
% 
% dW1bsr= (-1)*((W2bsr'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*real(dip)' ...
%               + ((W2bsi'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*imag(dip)';         
% 
%           dW1bsi = ((W2bsr'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*imag(dip)' ...
%          + ((W2bsi'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*real(dip)';
%   % Update
%         W2bsr = W2bsr - etaW2bs*dW2bsr;
%         W2bsi = W2bsi - etaW2bs*dW2bsi;
%         W2bs = W2bsr + 1i*W2bsi;
%         W1bsr = W1bsr - etaW1bs*dW1bsr;
%         W1bsi = W1bsi - etaW1bs*dW1bsi;
%         W1bs = W1bsr + 1i*W1bsi;
% end
%      op3=ybs;
% 
%     %%ploting
%  ch_name=["ROI 1","ROI 2","ROI 3","ROI 4","ROI 5","ROI 6","ROI 7","ROI 8","ROI 9","ROI 10","ROI 11","ROI 12",...
%      "ROI 13","ROI 14","ROI 15","ROI 16","ROI 17","ROI 18","ROI 19","ROI 20","ROI 21","ROI 22","ROI 23","ROI 24","ROI 25","ROI 26","ROI 27","ROI 28","ROI 29","ROI 30","ROI 31"...
%      "ROI 32","ROI 33","ROI 34","ROI 35","ROI 36","ROI 37","ROI 38","ROI 39","ROI 40","ROI 41","ROI 42","ROI 43","ROI 44","ROI 45","ROI 46","ROI 47","ROI 48","ROI 49",...
%      "ROI 50","ROI 51","ROI 52","ROI 53","ROI 54","ROI 55","ROI 56","ROI 57","ROI 58","ROI 59","ROI 60","ROI 61","ROI 62","ROI 63","ROI 64","ROI 65","ROI 66","ROI 67", "ROI 68",...
%     "ROI 69","ROI 70","ROI 71","ROI 72","ROI 73","ROI 74","ROI 75","ROI 76","ROI 77","ROI 78","ROI 79","ROI 80","ROI 81", "ROI 82","ROI 83","ROI 84","ROI 85","ROI 86","ROI 87",...
%     "ROI 88","ROI 89","ROI 90", "ROI 91", "ROI 92", "ROI 93", "ROI 94", "ROI 95", "ROI 96", "ROI 97", "ROI 98", "ROI 99", "ROI 100",...
%    "ROI 101","ROI 102","ROI 103","ROI 104","ROI 105","ROI 106","ROI 107","ROI 108","ROI 109","ROI 110","ROI 111","ROI 112",...
%      "ROI 113","ROI 114","ROI 115","ROI 116","ROI 117","ROI 118","ROI 119","ROI 120","ROI 121","ROI 122","ROI 123","ROI 124","ROI 125","ROI 126","ROI 127","ROI 128","ROI 129","ROI 130","ROI 131"...
%      "ROI 132","ROI 133","ROI 134","ROI 135","ROI 136","ROI 137","ROI 138","ROI 139","ROI 140","ROI 141","ROI 142","ROI 143","ROI 144","ROI 145","ROI 146","ROI 147","ROI 148","ROI 149",...
%      "ROI 150","ROI 151","ROI 152","ROI 153","ROI 154","ROI 155","ROI 156","ROI 157","ROI 158","ROI 159","ROI 160"];
% % % 
% % % "ROI 81", "ROI 82","ROI 83","ROI 84","ROI 85","ROI 86","ROI 87",...
%     %"ROI 88","ROI 89","ROI 90", "ROI 91", "ROI 92", "ROI 93", "ROI 94", "ROI 95", "ROI 96", "ROI 97", "ROI 98", "ROI 99", "ROI 100",...
%    %"ROI 101","ROI 102","ROI 103","ROI 104","ROI 105","ROI 106","ROI 107","ROI 108","ROI 109","ROI 110","ROI 111","ROI 112",...
%     % "ROI 113","ROI 114","ROI 115","ROI 116","ROI 117","ROI 118","ROI 119","ROI 120","ROI 121","ROI 122","ROI 123","ROI 124","ROI 125","ROI 126","ROI 127","ROI 128","ROI 129","ROI 130","ROI 131"...
%      %"ROI 132","ROI 133","ROI 134","ROI 135","ROI 136","ROI 137","ROI 138","ROI 139","ROI 140","ROI 141","ROI 142","ROI 143","ROI 144","ROI 145","ROI 146","ROI 147","ROI 148","ROI 149",...
%      %"ROI 150","ROI 151","ROI 152","ROI 153","ROI 154","ROI 155","ROI 156","ROI 157","ROI 158","ROI 159","ROI 160" 
% % % 
% % 
% Ypp_f = zeros(661,chnl);
%  Ydd_f = zeros(661,chnl);
%  %sim_output= zeros(N,nn);
% 
%  sim_output(:,nn)= op3;
% % % 
% %% main ploting
% dn= 2;
% tt = (0:N-1)*dn;
% 
%   figure(nn)
%  subplot(2,2,[1,2]);
%  plot(tt,Yd(:,1),'linewidth',1.4)
%  hold on 
%  plot(tt,op3(:,1),'linewidth',1.4)
%  legend('empirical','output')
% 
% 
%  xlabel('Time (sec)');
%  ylabel('Magnitude response')
%  title((ch_name(:,nn)))
% % % 
% % % 
% % % 
%  subplot(2,2,3);
%  plot(RMSE(1,:),'linewidth',1.4)
%  xlabel('epoch')
%  ylabel('rmse error')
% 
%  xlim([0 nepochsbs]);
% 
%  title((ch_name(:,nn)))
%  xlabel('epoch')
% ylabel('rmse error')
% % % 
% % % 
% clear('A');
% 
% end
% 
% 
% 
% 

