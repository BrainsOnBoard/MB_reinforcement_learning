function out = mb_td(seed, gamma, trntrain, discount, epskm, stimulus, varargin)
%
% As MB_MV_TD2, but with shibire/dTrpA1-like interventions to MBONs/DANs
%
% Inputs:
%     seed - an integer, N, that selects a prime number to seed the random
%            number generator
%    gamma - KC -> DAN synaptic weight
%   trntrain - number of learning walks (i.e. training episodes)
%    ntest - number of test trials
% discount - discount factor in the range [0, 1]
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
nx = 51; 
ny = 51;
centre = ceil(nx/2); % the centre of the arena
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
% N = pap.ntile * pap.stile / (pap.speed * pap.ttrial) = nx * 1 / (1 * tsntrain)
tscale = 1 / (pap.stile / pap.speed) / pap.ntile * min(nx,ny);

%%% Update default parameters with custom options
if nargin>7
  j = 1;
  while j<=numel(varargin)
    if strcmp(varargin{j},'intervene_id')
      % For simulating genetic interventions (e.g. shibire/dTrpA1)
      intervene_id = varargin{j+1}; % Which cell type
      intrvn_type = varargin{j+2}; % 1: multiplicative; 0: additive
      if isscalar(varargin{j+3}) 
        intrvn_strength = ones(ntraints,1)*varargin{j+3}; % Strength of intervention
      else
        intrvn_strength = varargin{j+3}; % Strength of intervention
      end
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
    end
  end
end

if d1flag, ny = 1; tsntrain = nx; end

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
tau_el = 15 * tscale; if d1flag, tau_el = 1; end
dec_el = 1 - 1/tau_el;

%%% Generate KC responses to cues
s = zeros(nk, ny, nx);

if strcmp(stimulus, 'ofstad') % Ofstad Arena Visual Input (load pre-generated KC activations for each location)
    struc = load('ofstad_stimulus.mat'); % TODO I think this file is a fixed arena size?
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
    s(j,:,:) = mygfilter(squeeze(s(j,:,:)),[1 1],[nx ny],'replicate');
  end
  
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
    s(j,:,:) = mygfilter(squeeze(s(j,:,:)),[1 1],[nx ny],'replicate');
  end

else
  error('???MB_MV_A: Stimulus arguments not recognised. Use: "ofstad", "one2one" or "sparse"');
end

if d1flag, s(:,1,1) = 0; end % FOR 1D

%%% Generate Learning Routes
circles = generate_circles(trntrain, nx);
actions = generate_actions(circles);
[c,~] = cellfun(@size, circles); % Find the lengths of the circles
tsntrain = max(c);  % Set the number of training time steps to the largest circle

%%% Initialise synaptic weights
if memsave
  wkmap = 0.1*rand(1,nk);
  wkmav = 0.1*rand(1,nk);
  wkgo = 0.1*rand(4,nk);
  wknogo = 0.1*rand(4,nk);
else
  wkmap = nan(1,nk,tsntrain,trntrain); % KC -> M+
  wkmav = nan(1,nk,tsntrain,trntrain); % KC -> M-
  wkmap(:,:,1,1) = 0.1*rand(1,nk);
  wkmav(:,:,1,1) = 0.1*rand(1,nk);
  wkgo = nan(4,nk,tsntrain,trntrain); % KC -> GO actor
  wknogo = nan(4,nk,tsntrain,trntrain); % KC -> NOGO actor
  wkgo(:,:,1,1) = 0.1*rand(4,nk);
  wknogo(:,:,1,1) = 0.1*rand(4,nk);
end
wkdap = gamma * ones(1,nk); % KC -> D+
wkdav = gamma * ones(1,nk); % KC -> D-
wmapdap = 1; % M+ -> D+
wmavdap = 1; % M- -> D+
wmapdav = 1; % M+ -> D-
wmavdav = 1; % M- -> D-

%%% Allocate memory for training structures
dap = nan(tsntrain,trntrain);
dav = nan(tsntrain,trntrain);
map = nan(tsntrain,trntrain); % Appetitive value neurons
mav = nan(tsntrain,trntrain); % Aversive value neurons
go = nan(tsntrain,4,trntrain); % GO actor neurons
nogo = nan(tsntrain,4,trntrain); % NOGO actor neurons
el = zeros(nk,1);

%%% Reward location
% rloc = {(ny14-(ny116-1)):(ny14+(ny116-1)); (nx34-(nx116-1)):(nx34+(nx116-1))};
% rloc = {(ny34-(ny116-1)):(ny34+(ny116-1)); (nx14-(nx116-1)):(nx14+(nx116-1))};
% rlocall = zeros(2,length(rloc{1})*length(rloc{2}));
% for j=1:length(rloc{1})
%   for k=1:length(rloc{2})
%     rlocall(1,(j-1)*length(rloc{2}) + k) = rloc{1}(j);
%     rlocall(2,(j-1)*length(rloc{2}) + k) = rloc{2}(k);
%   end;
% end;
% amp = 10; % Total volume of available reward
% r = zeros(ny,nx);
% r(rloc{1},rloc{2}) = amp;
% hunger = 0.01; % Negative reinforcement for being hungry (in range [0,1])
% radius = min(nx2, ny2) - 1;
% [xp yp] = meshgrid((-nx2+1):(nx2-1),(-ny2+1):(ny2-1));
% perim = 0; % The negative reinforcement for being outside the perim
% r = r - hunger*max(r(:));
% flag_escape = 0;
% if d1flag, r=zeros(1,nx); r(end) = amp; end;
% out.r = r;

%%% Path Integration Reward Schedule
pic = 1; % A constant
rloc = [centre, centre]; % Nest location
rew = zeros(tsntrain,trntrain); % Reward vector
pid = zeros(tsntrain,trntrain); % Euclidean distance to nest history
pid(1,:) = 0; % Euclidean distance to nest for initial time steps, 0 because the agent starts at its nest
pir = zeros(tsntrain,trntrain);

perim = 0;
flag_escape = 0;
radius = min(nx2, ny2) - 1;


%%% Run training phase
for tr=1:trntrain % from 1 to number of learning routes
  M = length(actions{tr});
  xx = circles{tr}(1,1);
  yy = circles{tr}(1,2);
  
  %%% Copy synaptic weights from previous trial 
  if tr>1
    if ~memsave
      wkmap(1,:,1,tr) = wkmap(1,:,j-1,tr-1);
      wkmav(1,:,1,tr) = wkmav(1,:,j-1,tr-1);
      wkgo(:,:,1,tr) = wkgo(:,:,j-1,tr-1);
      wknogo(:,:,1,tr) = wknogo(:,:,j-1,tr-1);
    end
  end

  % Stimulus at current time
  if strcmp(stimulus, 'ofstad')
    ss = s(:, yy, xx, th(1,tr));
  else
    ss = s(:, yy, xx);
  end

  %%% Initial time step
  if memsave
    % Save final version of weights before test trial
    if tr==trntrain
      fwkmap = wkmap;
      fwkmav = wkmav;
      fwkgo = wkgo;
      fwknogo = wknogo;
    end
    % Compute MBON firing rates (state and action values)
    map(1,tr) = wkmap * ss;
    mav(1,tr) = wkmav * ss;
    go(1,:,tr) = wkgo * ss;
    nogo(1,:,tr) = wknogo * ss;
  else
    % Save final version of weights before test trial
    if tr==trntrain
      fwkmap = wkmap(1,:,1,tr);
      fwkmav = wkmav(1,:,1,tr);
      fwkgo = wkgo(:,:,1,tr);
      fwknogo = wknogo(:,:,1,tr);
    end
    % Compute MBON firing rates (state and action values)
    map(1,tr) = wkmap(:,:,1,tr) * ss;
    mav(1,tr) = wkmav(:,:,1,tr) * ss;
    go(1,:,tr) = wkgo(:,:,1,tr) * ss;
    nogo(1,:,tr) = wknogo(:,:,1,tr) * ss;
  end

  %%% Enforce decisions
  decision = actions{tr}(1);
  
  %%% Update orientation
  if strcmp(action_flag,'allocentric')
    th = decision;
  elseif strcmp(action_flag,'egocentric')
    th = mod(th + (decision-1) - 1,4) + 1;
  end

  %%% Update location
  if strcmp(action_flag,'allocentric')
    if decision == 1 % Head towards 0 (right)
      xx = min(nx,max(1,xx + 1));
    elseif decision == 2 % Head towards 90 (up)
      yy = min(ny,max(1,yy + 1));
    elseif decision == 3 % Head towards 180 (left)
      xx = min(nx,max(1,xx - 1));
    elseif decision == 4 % Head towards 270 (down)
      yy = min(ny,max(1,yy - 1));
    end
  elseif strcmp(action_flag,'egocentric')
    xx = round(min(nx,max(1,xx + cos((th-1)*pi/2))));
    yy = round(min(ny,max(1,yy + sin((th-1)*pi/2))));
  end

  el(:) = 0;
  el = el + ss / tau_el;

  %%% Subsequent timesteps
  for j=2:M

    % Stimulus at current time
    if strcmp(stimulus, 'ofstad')
      try
        ss = s(:, yy, xx, th(j,tr));
      catch m
        keyboard;
      end
    else
      ss = s(:, yy, xx);
    end

    % Compute MBON firing rates (state and action values)
    if memsave
      map(j,tr) = wkmap * ss;
      mav(j,tr) = wkmav * ss;
      go(j,:,tr) = wkgo * ss;
      nogo(j,:,tr) = wknogo * ss;
    else
      map(j,tr) = wkmap(:,:,j-1,tr) * ss;
      mav(j,tr) = wkmav(:,:,j-1,tr) * ss;
      go(j,:,tr) = wkgo(:,:,j-1,tr) * ss;
      nogo(j,:,tr) = wknogo(:,:,j-1,tr) * ss;
    end
    
    % Add additional rewards based on location (current not used)
    %rew(j,tr) = r(yy,xx);

    % Compute reward
    pid(j,tr) = sqrt((rloc(1)-xx)^2 + (rloc(2)-yy)^2);
    pir(j,tr) = - pic * (pid(j,tr) - pid(j-1,tr));
    
    % Compute DAN firing rates (add strong punishment if failed to reach target)
    dap(j,tr) = max(0,wkdap * ss - wmapdap * map(j-1,tr) + wmavdap * mav(j-1,tr) + discount*(wmapdap * map(j,tr) - wmavdap * mav(j,tr)) + pir(j,tr) + perim*flag_escape);
    dav(j,tr) = max(0,wkdav * ss - wmavdav * mav(j-1,tr) + wmapdav * map(j-1,tr) + discount*(wmavdav * mav(j,tr) - wmapdav * map(j,tr)) - pir(j,tr) - perim*flag_escape);    

    % Update KC->MBON weights
    if memsave
      wkmap = max(0,wkmap + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav = max(0,wkmav + epskm * el' .* (dav(j,tr) - dap(j,tr)));
      wkgo(decision,:) = max(0,wkgo(decision,:) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision,:) = max(0,wknogo(decision,:) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
    else
      wkmap(:,:,j,tr) = max(0,wkmap(:,:,j-1,tr) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav(:,:,j,tr) = max(0,wkmav(:,:,j-1,tr) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
      wkgo(:,:,j,tr) = wkgo(:,:,j-1,tr);
      wknogo(:,:,j,tr) = wknogo(:,:,j-1,tr);
      wkgo(decision,:,j,tr) = max(0,wkgo(decision,:,j-1,tr) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision,:,j,tr) = max(0,wknogo(decision,:,j-1,tr) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
    end

    %%% Enforce decision
    decision = actions{tr}(j);
    
    %%% Update orientation
    if strcmp(action_flag,'allocentric')
      th = decision;
    elseif strcmp(action_flag,'egocentric')
      th = mod(th(j, tr) + (decision-1) - 1,4) + 1;
    end

    %%% Update location
    if strcmp(action_flag,'allocentric')
      if decision == 1 % Head towards 0 (right)
        xx = min(nx,max(1,xx + 1));
      elseif decision == 2 % Head towards 90 (up)
        yy = min(ny,max(1,yy + 1));
      elseif decision == 3 % Head towards 180 (left)
        xx = min(nx,max(1,xx - 1));
      elseif decision == 4 % Head towards 270 (down)
        yy = min(ny,max(1,yy - 1));
      end
    elseif strcmp(action_flag,'egocentric')
      xx = round(min(nx,max(1,xx + cos((th-1)*pi/2))));
      yy = round(min(ny,max(1,yy + sin((th-1)*pi/2))));
    end      

    % Update eligibility trace
    el = el * dec_el + ss / tau_el;

    % Report progress
    if progressflag
      if ((tr+1)/trntrain*10)>ceil(tr/trntrain*10) && j==2
        fprintf('%d percent complete\n',tr/trntrain*100);
      end
    end

  end
end
  
%%% Run testing phase

% Set parametres and allocate memory for testing phase
tsntest = ceil(pap.ttrial * tscale);
trntest = 5;
ntestarray = 4;
startloc = round(nx/(ntestarray+1):nx/(ntestarray+1):nx);
tx = nan(tsntest,trntest*ntestarray^2);
ty = nan(tsntest,trntest*ntestarray^2);
decision = zeros(tsntest,trntest); % Decisions during training phase
trnt = zeros(trntest,1); % Time steps per trial
th = zeros(tsntest,trntest);
th(1,:) = randi(4,1,trntest);

rew = zeros(tsntest,trntrain);

% Set trial index
tr = 1;

for j=1:trntest
  for k=1:ntestarray
    for l=1:ntestarray
 
      % Set initial location
      tx(1,tr) = startloc(l);
      ty(1,tr) = startloc(k);
      
      % Run simulation
      t = 1;
      endtrialflag = false;
      
      while t<tsntest && ~endtrialflag
        
        % Stimulus at current time
        if strcmp(stimulus, 'ofstad')
          try
            ss = s(:, ty(t,tr), tx(t,tr), th(t,tr));
          catch m
            keyboard;
          end
        else
          ss = s(:, ty(t,tr), tx(t,tr));
        end
        
        % Compute go and nogo firing rates
        go = fwkgo * ss;
        nogo = fwknogo * ss;
        
        % Make decision
        % Choose which direction to move (1-down, 2-left, 3-up, 4-right)
        if ~d1flag
          if strcmp(policy,'on')
            act_val = go - nogo; % Action values
            act_val = act_val - max(act_val);
            probs = exp(act_val*beta) / sum(exp(act_val*beta));
            probs = probs / sum(probs); % Ensure probs is normalised to 1 (to avoid rounding errors)
            randchoice = rand;
            flag = 1; 
            k = 1;
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
              end
              if randchoice<sum(probs(1:k))
                decision(t,tr) = k;
                flag = 0;
              end
              k = k + 1;
            end
          elseif strcmp(policy,'off')
            dec = find((go - nogo)==max((go - nogo)));
            decision(t,tr) = dec(randi(length(dec)));
          end
        else
          decision(t,tr) = 4; 
        end
        
        % Update location and orientation
        %%% Update orientation
        if strcmp(action_flag, 'allocentric')
          th(t+1, tr) = decision(t, tr);
        elseif strcmp(action_flag, 'egocentric')
          th(t+1, tr) = mod(th(t, tr) + (decision(t, tr) - 1) - 1, 4) + 1;
        end

        %%% Update location
        if strcmp(action_flag, 'allocentric')
          if decision(t, tr) == 1 % Head towards 0 (right)
            tx(t+1, tr) = min(nx, max(1, tx(t, tr) + 1));
            ty(t+1, tr) = ty(t, tr);
          elseif decision(t, tr) == 2 % Head towards 90 (up)
            tx(t+1, tr) = tx(t, tr);
            ty(t+1, tr) = min(ny, max(1, ty(t, tr) + 1));
          elseif decision(t, tr) == 3 % Head towards 180 (left)
            tx(t+1, tr) = min(nx, max(1, tx(t, tr) - 1));
            ty(t+1, tr) = ty(t, tr);
          elseif decision(t, tr) == 4 % Head towards 270 (down)
            tx(t+1, tr) = tx(t, tr);
            ty(t+1, tr) = min(ny, max(1, ty(t, tr) - 1));
          end
        elseif strcmp(action_flag, 'egocentric')
          tx(t+1, tr) = round(min(nx, max(1, tx(t, tr) + cos((th(t+1, tr) - 1) * pi/2))));
          ty(t+1, tr) = round(min(ny, max(1, ty(t, tr) + sin((th(t+1, tr) - 1) * pi/2))));
        end 
        
        % TODO replace fly inside of perimetre if it escapes
        % If the position update led the fly to be outside the perim, bring
        % it back to its previous location
%         if ((nx2 - xx(j+1,tr))^2 + (ny2 - yy(j+1,tr))^2) > radius^2
%             xx(j+1,tr) = xx(j,tr);
%             yy(j+1,tr) = yy(j,tr);
%             flag_escape = 1;
%         else
%             flag_escape = 0;
%         end  
        
        % Check if target is reached and set endtrialflag
        if tx(t+1, tr) == centre && ty(t+1, tr) == centre
          endtrialflag = true;
        end
        
        t = t + 1;
        
      end
      tr = tr + 1;
    end
  end
end

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
out.s = s;
out.tx = xx;
out.ty = yy;
out.th = th;
out.nx = nx;
out.ny = ny;
out.tsntrain = tsntrain;
out.trntrain = trntrain;
out.tsntest = tsntest;
out.trntest = trntest;
out.trnt = trnt;
out.nk = nk;
out.rew = rew;
out.fwkmap = fwkmap;
out.fwkmav = fwkmav;
out.fwkgo = fwkgo;
out.fwknogo = fwknogo;
out.pid = pid;
out.pir = pir;

% Plotting
% % Rewards obtained as function of time and trials: quick way to see how fast learning took place
% sr=zeros(q.tsntrain,q.trntrain);for j=1:q.trntrain for k=1:q.trnt(j) sr(k,j)=q.r(q.yy(k,j),q.xx(k,j)); end;end;imagesc(sr);

% % All visited locations for each trial, colour coded by trial
% jjet=zeros(q.trntrain,3); jjet(:,3) = 0:1/(q.trntrain-1):1; jjet(:,1) = 1:-1/(q.trntrain-1):0; for j=1:size(q.wkmap,4) plot(q.xx(:,j)+0.0*randn(q.tsntrain,1),q.yy(:,j)+0.0*randn(q.tsntrain,1),'.','color',jjet(ceil(j/size(q.wkmap,4)*64),:)); hold on; set(gca, 'xlim', [0.5, 20.5], 'ylim', [0.5, 20.5]); pause(0.1); end; hold off;

% % Learning the value function map (1D)
% j=1;ch=1; for kk=1:5:size(q.wkmap,4) plot(reshape(q.wkmap(ch,:,j,kk)-q.wkmav(ch,:,j,kk),q.ny,q.nx),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;

% %  DAN-RPE (1D)
% for kk=1:20:size(q.wkmap,4) plot(q.dap(:,kk)-q.dav(:,kk),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;

% %  Sliding average track history
% a=zeros(q.ny,q.nx,q.trntrain);for k=1:q.trntrain for j=1:(sum(q.decision(:,k)>0)+1) a(q.yy(j,k),q.xx(j,k),k)=1;end;end; aa=mylwfitends(a,3,20); myplaymov(0,aa(:,:,1:10:end),0.1,1-gray,[0 1]);
% a=zeros(q.ny,q.nx,q.trntrain);for k=1:q.trntrain for j=1:(sum(q.decision(:,k)>0)+1) a(q.yy(j,k),q.xx(j,k),k)=1;end;end; aa=mylwfitends(a,3,20); myplaymov(0,aa(:,:,1:10:end),0.1,1-gray,[0 1]);

% % Learning the value function map
% v=zeros(q.ny,q.nx,q.trntrain); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.trntrain val=(q.wkmap(1,:,1,j)-q.wkmav(1,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 

% % Final value function map 
% ss=reshape(q.s,q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v=reshape(val,q.ny,q.nx);imagesc(v);colormap(1-gray);
% v=zeros(q.ny,q.nx,4); for j=1:4 ss=reshape(q.s(:,:,:,j),q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v(:,:,j)=reshape(val,q.ny,q.nx); end; imagesc(mean(v,3));colormap(1-gray);

% % Learning the action value map
% act=1;v=zeros(q.ny,q.nx,q.trntrain); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.trntrain val=(q.wkgo(act,:,1,j)-q.wknogo(act,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 

% % Final state-value map PLUS action-value vectors TODO (CHANGE INDICES TO MAKE
% IT WORK ONCE YOU'VE UPDATED THE DECISION MAKING)
% ss=reshape(q.s,q.nk,q.ny*q.nx); val=(q.fwkmap-q.fwkmav)*ss; v=reshape(val,q.ny,q.nx);imagesc(v,[0 5]);colormap(1-gray); ss=reshape(q.s,q.nk,q.ny*q.nx);vy=(q.fwkgo(3,:)-q.fwknogo(3,:)-(q.fwkgo(1,:)-q.fwknogo(1,:)))*ss; vy=reshape(vy,q.ny,q.nx);vx=(q.fwkgo(4,:)-q.fwknogo(4,:)-(q.fwkgo(2,:)-q.fwknogo(2,:)))*ss; vx=reshape(vx,q.ny,q.nx); hold on;quiver(vx,vy,'r','autoscalefactor',1);hold off;

% % Smoothed x
% xx=zeros(size(xx));for j=1:q.trntrain xx(1:q.trnt(j),j)=mylwfitends(q.xx(1:q.trnt(j),j),1,2); end;
% % Smoothed y
% yy=zeros(size(xx));for j=1:q.trntrain yy(1:q.trnt(j),j)=mylwfitends(q.yy(1:q.trnt(j),j),1,2); end;
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