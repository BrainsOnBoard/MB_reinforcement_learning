function out = mb_mv_td3(seed,gamma,ntrial,nt,discount,epskm,phi,varargin)
%
% As MB_MV_TD2, but with shibire/dTrpA1-like interventions to MBONs/DANs
%
% Inputs:
%    gamma - KC->DAN synaptic weight
%     seed - an integer, N, that selects a prime number to seed the random
%            number generator
%       nt - # trials
%  rs_flag - reward schedule ID
%    epskm - learning rate
%
% Outputs:
%  out - struct containing numerous fields (see bottom of script) 

%%% Set defaults for optional parameters
policy = 'on';
progressflag = false;
memsave = true;
d1flag = false;
one2one = true;
intervene_id = 0;
% Environment size
pap.ntile = 8;
pap.stile = 25; % mm
pap.speed = 12.5; % mm/s
pap.ttrial = 300; % s
nx = 20; 
% %%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%
% nx=50;
% %q
%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%
ny = 20; 
nx14 = ceil(nx/4);
ny14 = ceil(ny/4);
nx34 = ceil(nx*3/4);
ny34 = ceil(ny*3/4);
nx116 = ceil(nx/16);
ny116 = ceil(ny/16);
% nk = nx*ny;
% np = 50;
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
if nargin>6
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
nk = nx*ny;

%%% Init random # stream
seeds = 10000:65000;
seeds = seeds(isprime(seeds));
rng(seeds(seed));          

%%% Network setup
sparseness = 0.15; % KC sparseness
mr = 10; % Total population firing rate.

% Softmax temperature
T = 1;
beta = 1 / T;
% Eligibility trace
tau_el = 15 * tscale; if d1flag, tau_el = 1; end;
dec_el = 1 - 1/tau_el;

%%% Initialise synaptic weights
if memsave
  wkmap = 0.1*rand(1,nk);
  wkmav = 0.1*rand(1,nk);
  wkgo = 0.1*rand(4,nk);
  wknogo = 0.1*rand(4,nk);
else
  wkmap = zeros(1,nk,nt,ntrial); % KC -> M+
  wkmav = zeros(1,nk,nt,ntrial); % KC -> M-
  wkmap(:,:,:,1) = 0.1*rand(nk,nt);
  wkmav(:,:,:,1) = 0.1*rand(nk,nt);
  wkgo = zeros(4,nk,nt,ntrial); % KC -> GO actor
  wknogo = zeros(4,nk,nt,ntrial); % KC -> NOGO actor
  wkgo(:,:,:,1) = 0.1*rand(4,nk,nt);
  wknogo(:,:,:,1) = 0.1*rand(4,nk,nt);
end;
wkdap = gamma * ones(1,nk); % KC -> D+
wkdav = gamma * ones(1,nk); % KC -> D-
wmapdap = 1; % M+ -> D+
wmavdap = 1; % M- -> D+
wmapdav = 1; % M+ -> D-
wmavdav = 1; % M- -> D-

% Reward location
% rloc = {9:12; 9:12}; if d1flag, rloc = {1 ;(nx-1):nx}; end; % FOR 1D
% rloc = {14:17; 14:17}; if d1flag, rloc = {1 ;(nx-1):nx}; end; % FOR 1D
rloc = {(ny14-(ny116-1)):(ny14+(ny116-1)); (nx34-(nx116-1)):(nx34+(nx116-1))};
rlocall = zeros(2,length(rloc{1})*length(rloc{2}));
for j=1:length(rloc{1})
  for k=1:length(rloc{2})
    rlocall(1,(j-1)*length(rloc{2}) + k) = rloc{1}(j);
    rlocall(2,(j-1)*length(rloc{2}) + k) = rloc{2}(k);
  end;
end;
% amp = 10; % Total volume of available reward
amp = 10; % Total volume of available reward
% sd = 1; % SD in spatial distribution of reward
% r = amp * mygaussiann([ny nx],[sd sd],'centre',rloc(1,:)); % rewards
r = zeros(ny,nx);
r(rloc{1},rloc{2}) = amp;
hunger = 0.01; % Negative reinforcement for being hungry (in range [0,1])
x2 = ceil(nx/2); y2 = ceil(ny/2);
radius = inf;min(x2,y2) + 10;
[xp yp] = meshgrid((-x2+1):(x2-1),(-y2+1):(y2-1));
perimamp = -1;
perim = (xp.^2 + yp.^2) / radius^2 * perimamp;
perim((xp.^2 + yp.^2)<radius^2) = 0;
perim = -2;
r = r - hunger*max(r(:));
% r(1:15,9)=-10;
if d1flag, r=zeros(1,nx); r(end) = amp; end;
out.r = r;

% rr = mygfilter(r,[15 15],[50 50],'replicate');
% [gx gy] = gradient(rr);
% o = atan2(gy,gx);
% do = (1 - mydiffmatrix(2*pi,o(:))/pi).^10;

%%% Generate KC responses to cues
s = zeros(nk,ny,nx);
if one2one
  ind = 1;
  for j=1:ny 
    for k=1:nx
      s(ind,j,k) = mr;
      ind = ind + 1;
    end
  end
else
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
end
if d1flag, s(:,1,1) = 0; end; % FOR 1D
% s = reshape(s,nk,nx*ny);
% ss = zeros(size(s));
% for j=1:(ny*nx)
%   ss(:,j) = sum(s .* repmat(do(j,:),[nk,1]),2);
%   ss(:,j) = ss(:,j) / sum(ss(:,j)) * 10;
% end;
% s = reshape(ss,nk,ny,nx);
for j=1:nk
  s(j,:,:) = mygfilter(squeeze(s(j,:,:)),[1 1],[20 20],'replicate');
end;
% %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%
% s = zeros(nk,ny,nx);
% for j=1:ny
%   for k=1:nx    
%     flag = true;
%     while flag
% %       s((j-1)*nx+k,j,k) = 1;
%       s(:,j,k) = double(rand(nk,1) < sparseness);
%       flag = ~any(s(:,j,k));
%     end;
%     s(:,j,k) = s(:,j,k) / sum(s(:,j,k)) * mr;
%   end;
% end;
% if d1flag, s(:,1,1) = 0; end; % FOR 1D
% 
% % s(:,1,2:10) = repmat(s(:,1,2),[1,1,9]);
% ss = zeros(size(s));
% ss(:,1,21) = s(:,1,21); s(:,1,21) = 0;
% ss = ss * 4;
% t=0:(nx-1);
% ye = exp(-t/2); ye = ye / sum(ye); ye = repmat(ye,[nk,1]); fye = fft(ye,[],2);
% % ye = zeros(1,nx); ye(1:15)=1; ye = ye / sum(ye); ye = repmat(ye,[nk,1]); fye = fft(ye,[],2);
% ss(:,1,:) = ifft(fft(squeeze(ss(:,1,:)),[],2) .* fye,[],2);
% % ss(ss(:,1,21)>0,1,21:35) = ss(ss(:,1,21)>0,1,21:35) + 2;
% ss(ss(:,1,21)>0,1,21:25) = ss(ss(:,1,21)>0,1,21:25) + 2;
% s = ss;s + ss;
% ss = s / max(s(:)) * 0.25;
% %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%

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
%   %%%%%%%%%%%%%%%%%
%   %%%%%%%%%%%%%%%%%
%   %%%%%%%%%%%%%%%%%
%   s = double(rand(size(ss))<ss) * 4;
%   %%%%%%%%%%%%%%%%%
%   %%%%%%%%%%%%%%%%%
%   %%%%%%%%%%%%%%%%%
  %%% Init trial
  if tr>1
    % Copy synaptic weights from previous trial
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
  
  % Compute MBON firing rates (state and action values)
  if memsave
    map(1,tr) = wkmap * s(:,yy(1,tr),xx(1,tr));
    mav(1,tr) = wkmav * s(:,yy(1,tr),xx(1,tr));
    go(1,:,tr) = wkgo * s(:,yy(1,tr),xx(1,tr));
    nogo(1,:,tr) = wknogo * s(:,yy(1,tr),xx(1,tr));
  else
    map(1,tr) = wkmap(:,:,1,tr) * s(:,yy(1,tr),xx(1,tr));
    mav(1,tr) = wkmav(:,:,1,tr) * s(:,yy(1,tr),xx(1,tr));
    go(1,:,tr) = wkgo(:,:,1,tr) * s(:,yy(1,tr),xx(1,tr));
    nogo(1,:,tr) = wknogo(:,:,1,tr) * s(:,yy(1,tr),xx(1,tr));
  end;
  
  % For "genetic" interventions
  if any(intervene_id==1)
    map(1,tr) = max(0,intrvn_type * map(1,tr) * intrvn_strength(1) + (1-intrvn_type) * (map(1,tr) + intrvn_strength(1)));
  end;
  if any(intervene_id==2)
    mav(1,tr) = max(0,intrvn_type * mav(1,tr) * intrvn_strength(1) + (1-intrvn_type) * (mav(1,tr) + intrvn_strength(1)));
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
  
  % Update position
  if mod(decision(1,tr),2)
    xx(2,tr) = xx(1,tr);
    yy(2,tr) = min(ny,max(1,yy(1,tr) + decision(1,tr) - 2));
  else
    xx(2,tr) = min(nx,max(1,xx(1,tr) + decision(1,tr) - 3));
    yy(2,tr) = yy(1,tr);
  end;
  el(:) = 0;
  el = el + s(:,yy(1,tr),xx(1,tr)) / tau_el;
  
  endtrialflag = false;
  j = 2;
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

    % Compute MBON firing rates (state and action values)
    if memsave
      map(j,tr) = wkmap * s(:,yy(j,tr),xx(j,tr));
      mav(j,tr) = wkmav * s(:,yy(j,tr),xx(j,tr));
      go(j,:,tr) = wkgo * s(:,yy(j,tr),xx(j,tr));
      nogo(j,:,tr) = wknogo * s(:,yy(j,tr),xx(j,tr));
    else
      map(j,tr) = wkmap(:,:,j-1,tr) * s(:,yy(j,tr),xx(j,tr));
      mav(j,tr) = wkmav(:,:,j-1,tr) * s(:,yy(j,tr),xx(j,tr));
      go(j,:,tr) = wkgo(:,:,j-1,tr) * s(:,yy(j,tr),xx(j,tr));
      nogo(j,:,tr) = wknogo(:,:,j-1,tr) * s(:,yy(j,tr),xx(j,tr));
    end;
    
    % For "genetic" interventions
    if any(intervene_id==1)
      map(j,tr) = max(0,intrvn_type * map(j,tr) * intrvn_strength(j) + (1-intrvn_type) * (map(j,tr) + intrvn_strength(j)));
    end;
    if any(intervene_id==2)
      mav(j,tr) = max(0,intrvn_type * mav(j,tr) * intrvn_strength(j) + (1-intrvn_type) * (mav(j,tr) + intrvn_strength(j)));
    end;
    
    rew(j,tr) = r(yy(j,tr),xx(j,tr));
%     if r(yy(j,tr),xx(j,tr))>0
%       endtrialflag = true;
%     end;
%     if sum(rew(1:j,tr))>9.2
%       endtrialflag = true;
%     end;
    
    % Compute DAN firing rates (add strong punishment if failed to reach target)
    dap(j,tr) = max(0,wkdap * s(:,yy(j,tr),xx(j,tr)) - wmapdap * map(j-1,tr) + wmavdap * mav(j-1,tr) + discount*(wmapdap * map(j,tr) - wmavdap * mav(j,tr)) + r(yy(j,tr),xx(j,tr)) + perim*double(((x2-xx(j,tr))^2 + (y2 - yy(j,tr))^2)>radius^2));
    dav(j,tr) = max(0,wkdav * s(:,yy(j,tr),xx(j,tr)) - wmavdav * mav(j-1,tr) + wmapdav * map(j-1,tr) + discount*(wmavdav * mav(j,tr) - wmapdav * map(j,tr)) - r(yy(j,tr),xx(j,tr)) - perim*double(((x2-xx(j,tr))^2 + (y2 - yy(j,tr))^2)>radius^2));
    
    % For "genetic" interventions
    if any(intervene_id==3)
      dap(j,tr) = max(0,intrvn_type * dap(j,tr) * intrvn_strength(j) + (1-intrvn_type) * (dap(j,tr) + intrvn_strength(j)));
    end;
    if any(intervene_id==4)
      dav(j,tr) = max(0,intrvn_type * dav(j,tr) * intrvn_strength(j) + (1-intrvn_type) * (dav(j,tr) + intrvn_strength(j)));
    end;
    
    % Update KC->MBON weights
    if memsave
      wkmap = max(0,wkmap + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav = max(0,wkmav + epskm * el' .* (dav(j,tr) - dap(j,tr)));
      wkgo(decision(j-1,tr),:) = max(0,wkgo(decision(j-1,tr),:) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision(j-1,tr),:) = max(0,wknogo(decision(j-1,tr),:) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
    else
      wkmap(:,:,j,tr) = max(0,wkmap(:,:,j-1,tr)*(1-phi) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wkmav(:,:,j,tr) = max(0,wkmav(:,:,j-1,tr)*(1-phi) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
%       wkmap(:,:,j,tr) = max(0,wkmap(:,:,j-1,tr) + epskm * el' .* (wkdav*s(:,yy(j,tr),xx(j,tr)) - dav(j,tr)));
%       wkmav(:,:,j,tr) = max(0,wkmav(:,:,j-1,tr) + epskm * el' .* (wkdap*s(:,yy(j,tr),xx(j,tr)) - dap(j,tr)));
      wkgo(:,:,j,tr) = wkgo(:,:,j-1,tr);
      wknogo(:,:,j,tr) = wknogo(:,:,j-1,tr);
      wkgo(decision(j-1,tr),:,j,tr) = max(0,wkgo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (dap(j,tr) - dav(j,tr)));
      wknogo(decision(j-1,tr),:,j,tr) = max(0,wknogo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (dav(j,tr) - dap(j,tr)));
%       wkgo(decision(j-1,tr),:,j,tr) = max(0,wkgo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (wkdav*s(:,yy(j,tr),xx(j,tr)) - dav(j,tr)));
%       wknogo(decision(j-1,tr),:,j,tr) = max(0,wknogo(decision(j-1,tr),:,j-1,tr) + epskm * el' .* (wkdap*s(:,yy(j,tr),xx(j,tr)) - dap(j,tr)));
    end;
    
    % If current position is outside the perimeter, move back to previous
    % position
    if ((x2-xx(j,tr))^2 + (y2 - yy(j,tr))^2)>radius^2
      xx(j,tr) = xx(j-1,tr);
      yy(j,tr) = yy(j-1,tr);
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
      
      % Update position
      if mod(decision(j,tr),2)
        xx(j+1,tr) = xx(j,tr);
        yy(j+1,tr) = min(ny,max(1,yy(j,tr) + decision(j,tr) - 2));
      else
        xx(j+1,tr) = min(nx,max(1,xx(j,tr) + decision(j,tr) - 3));
        yy(j+1,tr) = yy(j,tr);
      end;                
    end;        
    
    % Update eligibility trace
    el = el * dec_el + s(:,yy(j,tr),xx(j,tr)) / tau_el;
%     plot(el); set(gca,'ylim',[0 0.1]); pause(0.01);
    
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
%     if j>nt
%       endtrialflag = true;
%       j = j - 1;
%     end;
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
% out.r = r;
out.s = s;
% %%%%%%%%%%%%%%
% %%%%%%%%%%%%%%
% %%%%%%%%%%%%%%
% out.ss = ss;
% %%%%%%%%%%%%%%
% %%%%%%%%%%%%%%
% %%%%%%%%%%%%%%
out.xx = xx;
out.yy = yy;
out.nx = nx;
out.ny = ny;
out.nt = nt;
out.ntrial = ntrial;
out.trnt = trnt;
out.nk = nk;
out.rew = rew;

% Plotting
% % Rewards obtained as function of time and trials: quick way to see how
% fast learning took place
% sr=zeros(q.nt,q.ntrial);for j=1:q.ntrial for k=1:q.nt sr(k,j)=q.r(q.yy(k,j),q.xx(k,j)); end;end;imagesc(sr);
% % Movement along single trial route
% k=1;imagesc(q.r>(0.5*max(q.r(:))));colormap(1-gray);hold on;n=size(q.xx,1); for j=1:n plot(q.xx(j,k),q.yy(j,k),'o','color',jjet(mod(j-1,64)+1,:)); hold on;pause(0.01); set(gca,'xlim',[0 20],'ylim',[0 20]);end; hold off;
% % All visited locations for each trial, colour coded by trial
% for j=1:size(q.wkmap,4) plot(q.xx(:,j)+0.5*randn(q.nt,1),q.yy(:,j)+0.5*randn(q.nt,1),'.','color',jjet(ceil(j/size(q.wkmap,4)*64),:)); hold on; pause(0.1); end; hold off;
% % Learning the value function map (2D)
% for kk=1:size(q.wkmap,4) imagesc(reshape((q.wkmap(1,:,1,kk)-q.wkmav(1,:,1,kk))*reshape(q.s,[q.nk,q.nx*q.ny]),q.ny,q.nx));axis xy; pause(0.01); end;
% j=1;ch=1; for kk=1:size(q.wkmap,4) imagesc(reshape(q.wkmap(ch,:,j,kk)-q.wkmav(ch,:,j,kk),q.ny,q.nx),[-1 1]);axis xy; pause(0.1); end;
% % Learning the value function map (1D)
% j=1;ch=1; for kk=1:5:size(q.wkmap,4) plot(reshape(q.wkmap(ch,:,j,kk)-q.wkmav(ch,:,j,kk),q.ny,q.nx),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;
% %  DAN-RPE (1D)
% for kk=1:20:size(q.wkmap,4) plot(q.dap(:,kk)-q.dav(:,kk),'color',[kk/size(q.wkmap,4) 0 (size(q.wkmap,4)-kk)/size(q.wkmap,4)]); pause(0.1); hold on; end;hold off;
% %  Sliding average track history
% a=zeros(q.ny,q.nx,q.ntrial);for k=1:q.ntrial for j=1:(sum(q.decision(:,k)>0)+1) a(q.yy(j,k),q.xx(j,k),k)=1;end;end; aa=mylwfitends(a,3,20); myplaymov(0,aa(:,:,1:10:end),0.1,1-gray,[0 1]);
% % Learning the value function map
% v=zeros(q.ny,q.nx,q.ntrial); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.ntrial val=(q.wkmap(1,:,1,j)-q.wkmav(1,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 
% % Learning the action value map
% act=1;v=zeros(q.ny,q.nx,q.ntrial); ss=reshape(q.s,q.ny*q.nx,q.nk)'; for j=1:q.ntrial val=(q.wkgo(act,:,1,j)-q.wknogo(act,:,1,j))*ss; v(:,:,j)=reshape(val,q.ny,q.nx);end;myplaymov(0,v,0.1,1-gray); 
% % Final state-value map PLUS action-value vectors 
% ss=reshape(q.s,q.nk,q.ny*q.nx); val=(q.wkmap-q.wkmav)*ss; v=reshape(val,q.ny,q.nx);imagesc(v);colormap(1-gray); ss=reshape(q.s,q.nk,q.ny*q.nx);vy=(q.wkgo(3,:)-q.wknogo(3,:)-(q.wkgo(1,:)-q.wknogo(1,:)))*ss; vy=reshape(vy,q.ny,q.nx);vx=(q.wkgo(4,:)-q.wknogo(4,:)-(q.wkgo(2,:)-q.wknogo(2,:)))*ss; vx=reshape(vx,q.ny,q.nx); hold on;quiver(vx,vy,'r');hold off;

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