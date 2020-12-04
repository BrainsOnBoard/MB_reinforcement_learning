function out = mb_td(seed,gamma,ntrial,discount,epskm,stimulus,varargin)
%
% As MB_MV_TD2, but with shibire/dTrpA1-like interventions to MBONs/DANs
%
% Inputs:
%     seed - an integer, N, that selects a prime number to seed the random
%            number generator
%    gamma - KC->DAN synaptic weight
%   ntrial - number of trials (i.e. training episodes)
% discount - discount factor in the range [0,1]
%    epskm - learning rate
% stimulus - Choose the stimulus for generating cues: 'ofstad', 'one2one'
% or 'sparse'
% varargin - See below for optional arguments (line 51)

% Outputs:
%  out - struct containing numerous fields (see bottom of script) 
%
% Example run: q = mb_td(1,1,100,0.9,10^-1.8,'memsave',false, 'stimulus', 'ofstad');
%   

%%% Set defaults for optional parameters
fpath = '~/git/MB_reinforcement_learning/td_learning/';
policy = 'on';
progressflag = false;
memsave = false;
d1flag = false;
action_flag = 'allocentric'; % allocentric (turn towards a goal heading); egocentric (turn relative to current heading)
one2one = true;
intervene_id = 0;
% Environment size
pap.ntile = 8;
pap.stile = 25; % mm
pap.speed = 12.5; % mm/s
pap.ttrial = 300; % s
nx = 20; 
ny = 20;
nx2 = ceil(nx/2);
ny2 = ceil(ny/2);
nx14 = ceil(nx/4);
ny14 = ceil(ny/4);
nx34 = ceil(nx*3/4);
ny34 = ceil(ny*3/4);
nx116 = ceil(nx/16);
ny116 = ceil(ny/16);
nk = nx*ny; % # of KCs

% Compute a scaling factor for time, so that an equivalent trial duration
% is used in this model when compared to the trial duration in Ofstad et
% al. The basis for the scaling factor is that, if a fly can walk the width
% of the arena N times in Ofstad et al, given its walking speed, the model
% should be able to walk the width of the simulated arena N times, given a
% walking speed of 1 tile per time step.
% N = pap.ntile * pap.stile / (pap.speed * pap.ttrial) = nx * 1 / (1 * nt)
tscale = 1 / (pap.stile / pap.speed) / pap.ntile * min(nx,ny);
nt = ceil(pap.ttrial * tscale);

%%% Update default parameters with custom options
if nargin>7
  j = 1;
  while j<=numel(varargin)
    if strcmp(varargin{j},'intervene_id')
      % For simulating genetic interventions (e.g. shibire/dTrpA1)
      intervene_id = varargin{j+1}; % Which cell type
      intrvn_type = varargin{j+2}; % 1: multiplicative; 0: additive
      if isscalar(varargin{j+3}) 
        intrvn_strength = ones(nt,1)*varargin{j+3}; % Strength of intervention
      else
        intrvn_strength = varargin{j+3}; % Strength of intervention
      end;
      j = j + 4;
    elseif strcmp(varargin{j},'no')
      % Specify the number of cues
      no = varargin{j+1};
      j = j+2;
    elseif strcmp(varargin{j},'nk')
      % Specify the number of KCs
      nk = varargin{j+1};
      j = j+2;    
%       one2one = false;
    elseif strcmp(varargin{j},'policy')
      % Specify whether actions are ON or OFF policy
      policy = varargin{j+1};
      j = j+2;    
    elseif strcmp(varargin{j},'oned')
      % Test run in 1D: 
      d1flag = varargin{j+1};
      j = j+2;
    elseif strcmp(varargin{j},'actionflag')
      % Test run in 1D:
      action_flag = varargin{j+1};
      j = j+2;
    elseif strcmp(varargin{j},'memsave')
      % Test run in 1D: 
      memsave = varargin{j+1};
      j = j+2;
    elseif strcmp(varargin{j},'progress')
      % Test run in 1D:
      progressflag = varargin{j+1};
      j = j+2;
    else
      error('???MB_MV_A: Optional arguments not recognised.');
    end;
  end;
end;

if d1flag, ny = 1; nt = nx; end;

%%% Init random # stream
seeds = 10000:65000;
seeds = seeds(isprime(seeds));
rng(seeds(seed));          

%%% Network setup
sparseness = 0.15; % KC sparseness
mr = 10; % Total population firing rate.

% Softmax temperature
T = 0.25;
beta = 1 / T;
% Eligibility trace
tau_el = 15 * tscale; if d1flag, tau_el = 1; end;
dec_el = 1 - 1/tau_el;

%%% Generate KC responses to cues
%s = zeros(nk, ny, nx);
if strcmp(stimulus, 'ofstad') % Ofstad Arena Visual Input (load pre-generated KC activations for each location)
    struc = load([fpath 'ofstad_stimulus.mat']);
    s = struc.s/25;
    nk = struc.nk;
    
elseif strcmp(stimulus, 'one2one') % One to one mapping
  ind = 1;
  for j=1:ny 
    for k=1:nx
      s(ind,j,k) = mr;
      ind = ind + 1;
    end
  end
  % Add correlations between KCs
  for j=1:nk
    s(j,:,:) = mygfilter(squeeze(s(j,:,:)),[1 1],[20 20],'replicate');
  end;
  
elseif strcmp(stimulus, 'sparse') % Sparse mapping
  for j=1:ny
    for k=1:nx
      flag = true;
      while flag
        %       s((j-1)*nx+k,j,k) = 1;
        s(:,j,k) = double(rand(nk,1) < sparseness);
        flag = ~any(s(:,j,k));
      end
      s(:,j,k) = s(:,j,k) / sum(s(:,j,k)) * mr;
    end
  end
  % Add correlations between KCs
  for j=1:nk
    s(j,:,:) = mygfilter(squeeze(s(j,:,:)),[1 1],[20 20],'replicate');
  end;

else
  error('???MB_MV_A: Stimulus arguments not recognised. Use: "ofstad", "one2one" or "sparse"');
end;

if d1flag, s(:,1,1) = 0; end; % FOR 1D

%%% Initialise synaptic weights
if memsave
  wkmap = 0.1*rand(1,nk);
  wkmav = 0.1*rand(1,nk);
  wkgo = 0.1*rand(4,nk);
  wknogo = 0.1*rand(4,nk);
else
  wkmap = zeros(1,nk,nt,ntrial); % KC -> M+
  wkmav = zeros(1,nk,nt,ntrial); % KC -> M-
  wkmap(:,:,1,1) = 0.1*rand(1,nk);
  wkmav(:,:,1,1) = 0.1*rand(1,nk);
  wkgo = zeros(4,nk,nt,ntrial); % KC -> GO actor
  wknogo = zeros(4,nk,nt,ntrial); % KC -> NOGO actor
  wkgo(:,:,1,1) = 0.1*rand(4,nk);
  wknogo(:,:,1,1) = 0.1*rand(4,nk);
end;
wkdap = gamma * ones(1,nk); % KC -> D+
wkdav = gamma * ones(1,nk); % KC -> D-
wmapdap = 1; % M+ -> D+
wmavdap = 1; % M- -> D+
wmapdav = 1; % M+ -> D-
wmavdav = 1; % M- -> D-

%%% Reward location
% rloc = {(ny14-(ny116-1)):(ny14+(ny116-1)); (nx34-(nx116-1)):(nx34+(nx116-1))};
rloc = {(ny34-(ny116-1)):(ny34+(ny116-1)); (nx14-(nx116-1)):(nx14+(nx116-1))};
rlocall = zeros(2,length(rloc{1})*length(rloc{2}));
for j=1:length(rloc{1})
  for k=1:length(rloc{2})
    rlocall(1,(j-1)*length(rloc{2}) + k) = rloc{1}(j);
    rlocall(2,(j-1)*length(rloc{2}) + k) = rloc{2}(k);
  end;
end;
amp = 10; % Total volume of available reward
r = zeros(ny,nx);
r(rloc{1},rloc{2}) = amp;
hunger = 0.01; % Negative reinforcement for being hungry (in range [0,1])
radius = min(nx2, ny2) - 1;
[xp yp] = meshgrid((-nx2+1):(nx2-1),(-ny2+1):(ny2-1));
perim = -10; % The negative reinforcement for being outside the perim
r = r - hunger*max(r(:));
flag_escape = 0;
if d1flag, r=zeros(1,nx); r(end) = amp; end;
out.r = r;

%%% Allocate memory
dap = zeros(nt,ntrial);
dav = zeros(nt,ntrial);
map = zeros(nt,ntrial); % Appetitive value neurons
mav = zeros(nt,ntrial); % Aversive value neurons
go = zeros(nt,4,ntrial); % GO actor neurons
nogo = zeros(nt,4,ntrial); % NOGO actor neurons
decision = zeros(nt,ntrial);
el = zeros(nk,1);
probs = zeros(4,1);
xx = nan(nt,ntrial);
yy = nan(nt,ntrial);
trnt = zeros(ntrial,1); % #time steps per trial
th = zeros(nt,ntrial);
th(1,:) = randi(4,1,ntrial);

% Init position
startpos = ones(4,2) * ceil(nx14); 
if d1flag, startpos = [1 1]; end;
for j=1:ntrial
  yy(1,j) = startpos(mod(j-1,size(startpos,1))+1,1);
  xx(1,j) = startpos(mod(j-1,size(startpos,1))+1,2);
end;
rew = zeros(nt,ntrial);

%%% Run simulation
for tr=1:ntrial
 
  %%% Copy synaptic weights from previous trial 
  if tr>1
    if ~memsave
      wkmap(1,:,1,tr) = wkmap(1,:,j-1,tr-1);
      wkmav(1,:,1,tr) = wkmav(1,:,j-1,tr-1);
      wkgo(:,:,1,tr) = wkgo(:,:,j-1,tr-1);
      wknogo(:,:,1,tr) = wknogo(:,:,j-1,tr-1);
    end;
  end;   
  
%   if tr==ntrial
%     r(:) = min(r(:));
%   end;

  % Stimulus at current time
  if strcmp(stimulus, 'ofstad')
    ss = s(:, yy(1,tr), xx(1,tr), th(1,tr));
%     ss = s(:, yy(1,tr), xx(1,tr));
  else
    ss = s(:, yy(1,tr), xx(1,tr));
  end;
  
  %%% Initial time step
  if memsave
    % Save final version of weights before test trial
    if tr==ntrial
      fwkmap = wkmap;
      fwkmav = wkmav;
      fwkgo = wkgo;
      fwknogo = wknogo;
    end;
    % Compute MBON firing rates (state and action values)
    map(1,tr) = wkmap * ss;
    mav(1,tr) = wkmav * ss;
    go(1,:,tr) = wkgo * ss; 
    nogo(1,:,tr) = wknogo * ss; %nogo(1,3,tr) = inf;
  else
    % Save final version of weights before test trial
    if tr==ntrial
      fwkmap = wkmap(1,:,1,tr);
      fwkmav = wkmav(1,:,1,tr);
      fwkgo = wkgo(:,:,1,tr);
      fwknogo = wknogo(:,:,1,tr);
    end;
    % Compute MBON firing rates (state and action values)
    map(1,tr) = wkmap(:,:,1,tr) * ss;
    mav(1,tr) = wkmav(:,:,1,tr) * ss;
    go(1,:,tr) = wkgo(:,:,1,tr) * ss;
    nogo(1,:,tr) = wknogo(:,:,1,tr) * ss;
  end;
    
  % Choose which direction to move (1-down, 2-left, 3-up, 4-right)
  if ~d1flag
    if strcmp(policy,'on')
      act_val = go(1,:,tr)-nogo(1,:,tr); % Action values
      act_val = act_val - max(act_val);
      probs = exp(act_val*beta) / sum(exp(act_val*beta));
      probs = probs / sum(probs); % Ensure probs is normalised to 1 (to avoid rounding errors)
      randchoice = rand;
      flag = 1; k = 1;
      while flag
        if k>4 % FOR DEBUGGING
          %         error('???MB_MV_TD2: no choice made.');
          keyboard;
          %         fprintf('K ABOVE NO\n');
          %         fprintf('sumprobs = %.5f     randchoice = %f\n',sum(probs),randchoice);
          %         fprintf('no = %d      seed = %d\n',no,seed);
          %         for zz=1:nk
          %           fprintf('%f\n',s(zz,17));
          %         end;
        end;
        if randchoice<sum(probs(1:k))
          decision(1,tr) = k;
          flag = 0;
        end;
        k = k + 1;
      end;
    elseif strcmp(policy,'off')
      dec = find((go(1,:,tr) - nogo(1,:,tr))==max((go(1,:,tr) - nogo(1,:,tr))));
      decision(1,tr) = dec(randi(length(dec)));
    end;
  else
    decision(1,tr) = 4; 
  end;
  
  %%% Update orientation
  if strcmp(action_flag,'allocentric')
    th(2,tr) = decision(1,tr);
  elseif strcmp(action_flag,'egocentric')
    th(2,tr) = mod(th(1, tr) + (decision(1,tr)-1) - 1,4) + 1;
  end;
  
  %%% Update location
  if strcmp(action_flag,'allocentric')
    if decision(1,tr) == 1 % Head towards 0 (right)
      xx(2,tr) = min(nx,max(1,xx(1,tr) + 1));
      yy(2,tr) = yy(1,tr);
    elseif decision(1,tr) == 2 % Head towards 90 (up)
      xx(2,tr) = xx(1,tr);
      yy(2,tr) = min(ny,max(1,yy(1,tr) + 1));
    elseif decision(1,tr) == 3 % Head towards 180 (left)
      xx(2,tr) = min(nx,max(1,xx(1,tr) - 1));
      yy(2,tr) = yy(1,tr);
    elseif decision(1,tr) == 4 % Head towards 270 (down)
      xx(2,tr) = xx(1,tr);
      yy(2,tr) = min(ny,max(1,yy(1,tr) - 1));
    end;
  elseif strcmp(action_flag,'egocentric')
    xx(2,tr) = round(min(nx,max(1,xx(1,tr) + cos((th(2,tr)-1)*pi/2))));
    yy(2,tr) = round(min(ny,max(1,yy(1,tr) + sin((th(2,tr)-1)*pi/2))));
  end;

  el(:) = 0;
  el = el + ss / tau_el;
  
  endtrialflag = false;
  j = 2;
  
  %%% Subsequent timesteps
  while (j<=nt) && ~endtrialflag % Loop over time until target is reached
      
    if tr<ntrial
      if ~d1flag
        if any(sum(repmat([yy(j,tr);xx(j,tr)],[1 size(rlocall,2)])==rlocall,1)==2)
          endtrialflag = true;
        end;
      else
        if xx(j,tr)==nx
          endtrialflag = true;
        end;
      end;
    end;
    
    % Stimulus at current time
    if strcmp(stimulus, 'ofstad')
      try
        ss = s(:, yy(j,tr), xx(j,tr), th(j,tr));
      catch m
        keyboard;
      end;
%       ss = s(:, yy(j,tr), xx(j,tr));
    else
      ss = s(:, yy(j,tr), xx(j,tr));
    end;
    
    % Compute MBON firing rates (state and action values)
    if memsave
      map(j,tr) = wkmap * ss;
      mav(j,tr) = wkmav * ss;
      go(j,:,tr) = wkgo * ss; 
      nogo(j,:,tr) = wknogo * ss; %nogo(j,3,tr) = inf;
    else
      map(j,tr) = wkmap(:,:,j-1,tr) * ss;
      mav(j,tr) = wkmav(:,:,j-1,tr) * ss;
      go(j,:,tr) = wkgo(:,:,j-1,tr) * ss; 
      nogo(j,:,tr) = wknogo(:,:,j-1,tr) * ss; 
    end;
    
    rew(j,tr) = r(yy(j,tr),xx(j,tr));
%     if r(yy(j,tr),xx(j,tr))>0
%       endtrialflag = true;
%     end;
    
    % Compute DAN firing rates (add strong punishment if failed to reach target)
    dap(j,tr) = max(0,wkdap * ss - wmapdap * map(j-1,tr) + wmavdap * mav(j-1,tr) + discount*(wmapdap * map(j,tr) - wmavdap * mav(j,tr)) + r(yy(j,tr),xx(j,tr)) + perim*flag_escape);
    dav(j,tr) = max(0,wkdav * ss - wmavdav * mav(j-1,tr) + wmapdav * map(j-1,tr) + discount*(wmavdav * mav(j,tr) - wmapdav * map(j,tr)) - r(yy(j,tr),xx(j,tr)) - perim*flag_escape);    
    
    % Update KC->MBON weights
    if memsave
      wkmap = max(0,wkmap + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav = max(0,wkmav + epskm * el' .* (dav(j,tr) - dap(j,tr)));
      wkgo(decision(j-1,tr),:) = max(0,wkgo(decision(j-1,tr),:) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision(j-1,tr),:) = max(0,wknogo(decision(j-1,tr),:) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
    else
      wkmap(:,:,j,tr) = max(0,wkmap(:,:,j-1,tr) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav(:,:,j,tr) = max(0,wkmav(:,:,j-1,tr) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
      wkgo(:,:,j,tr) = wkgo(:,:,j-1,tr);
      wknogo(:,:,j,tr) = wknogo(:,:,j-1,tr);
      wkgo(decision(j-1,tr),:,j,tr) = max(0,wkgo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision(j-1,tr),:,j,tr) = max(0,wknogo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
    end;
    
    if j<nt
      if ~d1flag
        % Choose which direction to move (1-down, 2-left, 3-up, 4-right)
        if strcmp(policy,'on')
          act_val = go(j,:,tr)-nogo(j,:,tr); % Action values
          act_val = act_val - max(act_val);
          probs = exp(act_val*beta) / sum(exp(act_val*beta));
          probs = probs / sum(probs); % Ensure probs is normalised to 1 (to avoid rounding errors)
          randchoice = rand;
          flag = 1; k = 1;
          while flag
            if k>4 % FOR DEBUGGING
              keyboard;
              %             error('???MB_MV_TD2: Line 402 - no choice made.');
              
              %             keyboard;
              %         fprintf('K ABOVE NO\n');
              %         fprintf('sumprobs = %.5f     randchoice = %f\n',sum(probs),randchoice);
              %         fprintf('no = %d      seed = %d\n',no,seed);
              %         for zz=1:nk
              %           fprintf('%f\n',s(zz,17));
              %         end;
            end;
            if randchoice<sum(probs(1:k))
              decision(j,tr) = k;
              flag = 0;
            end;
            k = k + 1;
          end;
        elseif strcmp(policy,'off')
          dec = find((go(j,:,tr) - nogo(j,:,tr))==max((go(j,:,tr) - nogo(j,:,tr))));
          decision(j,tr) = dec(randi(length(dec)));
        end;
      else
        decision(j,tr) = 4; 
      end;
      
      %%% Update orientation
      if strcmp(action_flag,'allocentric')
        th(j+1,tr) = decision(j,tr);
      elseif strcmp(action_flag,'egocentric')
        th(j+1,tr) = mod(th(j, tr) + (decision(j,tr)-1) - 1,4) + 1;
      end;
      
      %%% Update location
      if strcmp(action_flag,'allocentric')
        if decision(j,tr) == 1 % Head towards 0 (right)
          xx(j+1,tr) = min(nx,max(1,xx(j,tr) + 1));
          yy(j+1,tr) = yy(j,tr);
        elseif decision(j,tr) == 2 % Head towards 90 (up)
          xx(j+1,tr) = xx(j,tr);
          yy(j+1,tr) = min(ny,max(1,yy(j,tr) + 1));
        elseif decision(j,tr) == 3 % Head towards 180 (left)
          xx(j+1,tr) = min(nx,max(1,xx(j,tr) - 1));
          yy(j+1,tr) = yy(j,tr);
        elseif decision(j,tr) == 4 % Head towards 270 (down)
          xx(j+1,tr) = xx(j,tr);
          yy(j+1,tr) = min(ny,max(1,yy(j,tr) - 1));
        end;
      elseif strcmp(action_flag,'egocentric')
        xx(j+1,tr) = round(min(nx,max(1,xx(j,tr) + cos((th(j+1,tr)-1)*pi/2))));
        yy(j+1,tr) = round(min(ny,max(1,yy(j,tr) + sin((th(j+1,tr)-1)*pi/2))));
      end;                  
    
      % If the position update led the fly to be outside the perim, bring
      % it back to its previous location
      if ((nx2 - xx(j+1,tr))^2 + (ny2 - yy(j+1,tr))^2) > radius^2
          xx(j+1,tr) = xx(j,tr);
          yy(j+1,tr) = yy(j,tr);
          flag_escape = 1;
      else
          flag_escape = 0;
      end;
    end;         
    
    % Update eligibility trace
    el = el * dec_el + ss / tau_el;
    
    % Report progress
    if progressflag
      if ((tr+1)/ntrial*10)>ceil(tr/ntrial*10) && j==2
        fprintf('%d percent complete\n',tr/ntrial*100);
      end;
    end;
    % Record number of time steps in trial
    trnt(tr) = j;
    
    % Update time
    j = j + 1;
  end;  
end;

%%% Creater output struct
out.map = map;
out.mav = mav;
out.dap = dap;
out.dav = dav;
out.go = go;
out.nogo = nogo;
out.wkmap = wkmap;
out.wkmav = wkmav;
out.wkgo = wkgo;
out.wknogo = wknogo;
out.decision = decision;
out.r = r;
out.s = s;
out.xx = xx;
out.yy = yy;
out.th = th;
out.nx = nx;
out.ny = ny;
out.nt = nt;
out.ntrial = ntrial;
out.trnt = trnt;
out.nk = nk;
out.rew = rew;
out.fwkmap = fwkmap;
out.fwkmav = fwkmav;
out.fwkgo = fwkgo;
out.fwknogo = fwknogo;

% Plotting
% % Rewards obtained as function of time and trials: quick way to see how fast learning took place
% sr=zeros(q.nt,q.ntrial);for j=1:q.ntrial for k=1:q.trnt(j) sr(k,j)=q.r(q.yy(k,j),q.xx(k,j)); end;end;imagesc(sr);

% % All visited locations for each trial, colour coded by trial
% jjet=zeros(q.ntrial,3); jjet(:,3) = 0:1/(q.ntrial-1):1; jjet(:,1) = 1:-1/(q.ntrial-1):0; for j=1:size(q.wkmap,4) plot(q.xx(:,j)+0.0*randn(q.nt,1),q.yy(:,j)+0.0*randn(q.nt,1),'.','color',jjet(ceil(j/size(q.wkmap,4)*64),:)); hold on; set(gca, 'xlim', [0.5, 20.5], 'ylim', [0.5, 20.5]); pause(0.1); end; hold off;

% % Learning the value function map (1D)
% j=1;ch=1; for kk=1:5:size(q.wkmap,4) plot(reshape(q.wkmap(ch,:,j,kk)-q.wkmav(ch,:,j,kk),q.ny,q.nx),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;

% %  DAN-RPE (1D)
% for kk=1:20:size(q.wkmap,4) plot(q.dap(:,kk)-q.dav(:,kk),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;

% %  Sliding average track history
% a=zeros(q.ny,q.nx,q.ntrial);for k=1:q.ntrial for j=1:(sum(q.decision(:,k)>0)+1) a(q.yy(j,k),q.xx(j,k),k)=1;end;end; aa=mylwfitends(a,3,20); myplaymov(0,aa(:,:,1:10:end),0.1,1-gray,[0 1]);
% a=zeros(q.ny,q.nx,q.ntrial);for k=1:q.ntrial for j=1:(sum(q.decision(:,k)>0)+1) a(q.yy(j,k),q.xx(j,k),k)=1;end;end; aa=mylwfitends(a,3,20); myplaymov(0,aa(:,:,1:10:end),0.1,1-gray,[0 1]);

% % Learning the value function map
% v=zeros(q.ny,q.nx,q.ntrial); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.ntrial val=(q.wkmap(1,:,1,j)-q.wkmav(1,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 

% % Final value function map 
% ss=reshape(q.s,q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v=reshape(val,q.ny,q.nx);imagesc(v);colormap(1-gray);
% v=zeros(q.ny,q.nx,4); for j=1:4 ss=reshape(q.s(:,:,:,j),q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v(:,:,j)=reshape(val,q.ny,q.nx); end; imagesc(mean(v,3));colormap(1-gray);

% % Learning the action value map
% act=1;v=zeros(q.ny,q.nx,q.ntrial); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.ntrial val=(q.wkgo(act,:,1,j)-q.wknogo(act,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 

% % Final state-value map PLUS action-value vectors TODO (CHANGE INDICES TO MAKE
% IT WORK ONCE YOU'VE UPDATED THE DECISION MAKING)
% ss=reshape(q.s,q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v=reshape(val,q.ny,q.nx);imagesc(v,[0 5]);colormap(1-gray); ss=reshape(q.s,q.nk,q.ny*q.nx);vy=(q.fwkgo(3,:)-q.fwknogo(3,:)-(q.fwkgo(1,:)-q.fwknogo(1,:)))*ss; vy=reshape(vy,q.ny,q.nx);vx=(q.fwkgo(4,:)-q.fwknogo(4,:)-(q.fwkgo(2,:)-q.fwknogo(2,:)))*ss; vx=reshape(vx,q.ny,q.nx); hold on;quiver(vx,vy,'r','autoscalefactor',1);hold off;

% % Smoothed x
% xx=zeros(size(xx));for j=1:q.ntrial xx(1:q.trnt(j),j)=mylwfitends(q.xx(1:q.trnt(j),j),1,2); end;
% % Smoothed y
% yy=zeros(size(xx));for j=1:q.ntrial yy(1:q.trnt(j),j)=mylwfitends(q.yy(1:q.trnt(j),j),1,2); end;
% % Find times and trials for which xx and yy are in the desired range
% xin=xx>4 & xx<12; yin=yy>3 & yy<5;
% [t tr]=find(yin&xin);
% t(tr>30) = [];
% tr(tr>30) = [];
% tr(t==size(xx,1))=[];
% t(t==size(xx,1))=[];
% % Compute state values
% v=zeros(q.ny,q.nx,size(q.wkmap,4)); ss=reshape(q.s,q.nk,q.ny*q.nx); for j=1:size(q.wkmap,4) val=(q.wkmap(1,:,1,j)-q.wkmav(1,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;imagesc(v(:,:,end));
% % Init arrays
% a=nan(q.ny,q.nx,50);
% vv=nan(q.ny,q.nx,50);
% ind=ones(q.ny,q.nx);
% % Compute angles
% for j=1:length(t)
%   vx = xx(t(j)+1,tr(j)) - xx(t(j),tr(j));
%   vy = yy(t(j)+1,tr(j)) - yy(t(j),tr(j));
%   xind = round(xx(t(j),tr(j))); yind = round(yy(t(j),tr(j)));
%   a(yind,xind,ind(yind,xind)) = atan2(vy,vx);
%   vv(yind,xind,ind(yind,xind)) = v(yind,xind,tr(j));
%   ind(yind,xind) = ind(yind,xind) + 1;
% end;
% asd = nanstd(a,[],3);
% am = nanmean(a,3);
% vvm = nanmean(vv,3);
