function out = mb_vs_conditioning(seed,rewards,epskm,varargin)
%
% Model as in mb_vs, for the specific case of simulating Drosophila
% conditioning experiments: 1) train CS+, 2) train CS-, 3) test
% intervene_id - choose a cell type to block:
%          1 - map
%          2 - mav
%          3 - dap
%          4 - dav


%%%% Set defaults for optional parameters
flag_plasticity_rule = 1;
intervene_id = 0;

% Model parameters
nt = 30;
nk = 100;
no = 2;
sparseness = 0.1;
lambda = 12;
gamma = 1;
nu = 0.1;

if nargin>4 
  j = 1;
  while j<=numel(varargin)    
    if strcmp(varargin{j},'plasticity_rule')
      % 1 - delta_w+ ~ (lambda - d-)
      % 0 - delta_w+ ~ (w_k*k - d-) 
      flag_plasticity_rule = varargin{j+1}; j = j + 2;
    elseif strcmp(varargin{j},'intervene_id')
      intervene_id = varargin{j+1}; % Which cell type
      intrvn_type = varargin{j+2}; % 1: multiplicative; 0: additive
      if isscalar(varargin{j+3})
        intrvn_strength = ones(nt,1)*varargin{j+3}; % Strength of intervention
      else
        intrvn_strength = varargin{j+3}; % Strength of intervention
      end;
      j = j + 4;
    end;
  end;
end;

% Seed random number generator
seeds = 10000:65000;
seeds = seeds(isprime(seeds));
rng(seeds(seed));

%%%% Reward schedules..............(ID,SD,NT,NO)
r = rewards;

% Initialise synaptic weights
wkmap = zeros(1,nk,nt); % KC->M+
wkmav = zeros(1,nk,nt); % KC->M-
wkmap(:,:,1) = 0.1*rand(1,nk); 
wkmav(:,:,1) = 0.1*rand(1,nk);
wkdap = gamma*ones(1,nk); % KC->D+
wkdav = gamma*ones(1,nk); % KC->D-
wmdap = 1; % M+/- -> D+
wmdav = 1; % M+/- -> D-

% Softmax temperature
T = 0.2;
beta = 1 / T;

% Generate KC response to cues
s = zeros(nk,no);
for j=1:no
  s(:,j) = double(rand(nk,1)<sparseness);
  s(:,j) = s(:,j) / sum(s(:,j)) * 10;  
end;

% Allocate memory
dap = zeros(nt,1);
dav = zeros(nt,1);
map = zeros(nt,2);
mav = zeros(nt,2);
nogo = zeros(nt,2);
decision = zeros(nt,1);
mdiff = zeros(no,1);

%%%
%%% Train CS+
%%%
for j=1:(nt/3)
	stim = 1;
  % Compute MBON firing rates
	map(j,stim) = wkmap(:,:,j) * s(:,stim);
	mav(j,stim) = wkmav(:,:,j) * s(:,stim);
	
  % Apply intervention (if any)
	if any(intervene_id==1)
		map(j,stim) = max(0,intrvn_type * map(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (map(j,stim) + intrvn_strength(j)));
	end;
	if any(intervene_id==2)
		mav(j,stim) = max(0,intrvn_type * mav(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (mav(j,stim) + intrvn_strength(j)));
	end;
  
  % Compute reward prediction
	mdiff(stim) = map(j,stim) - mav(j,stim);
  
  % Force decision
  decision(j) = stim;
  
  dap(j) = max(0,wkdap * s(:,decision(j)) + wmdap * mav(j,decision(j)) + max(0,r(j,decision(j))));
  dav(j) = max(0,wkdav * s(:,decision(j)) + wmdav * map(j,decision(j)) - min(0,r(j,decision(j))));  
%   dap(j) = max(0,wkdap * s(:,decision(j)) + 2*wmdap * mav(j,decision(j)) + r(j,decision(j)));
%   dav(j) = max(0,wkdav * s(:,decision(j)) + 2*wmdav * map(j,decision(j)) - r(j,decision(j)));
    
  % Apply intervention (if any)
  if any(intervene_id==3)
    dap(j) = max(0,intrvn_type * dap(j) * intrvn_strength(j) + (1-intrvn_type) * (dap(j) + intrvn_strength(j)));
  end;
  if any(intervene_id==4)
    dav(j) = max(0,intrvn_type * dav(j) * intrvn_strength(j) + (1-intrvn_type) * (dav(j) + intrvn_strength(j)));
  end;

  % Update KC->MBON synaptic weights (except on last trial)
	if flag_plasticity_rule>0
		wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' * (lambda - dav(j)));
    wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' * (lambda - dap(j)));
	else
    wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' .* (wkdav*s(:,decision(j)) - dav(j)));
    wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' .* (wkdap*s(:,decision(j)) - dap(j)));
	end;
end;

%%%
%%% Train CS-
%%%
for j=(nt/3 + 1):(2*nt/3)
	stim = 2;
  % Compute MBON firing rates
	map(j,stim) = wkmap(:,:,j) * s(:,stim);
	mav(j,stim) = wkmav(:,:,j) * s(:,stim);
	
  % Apply intervention (if any)
	if any(intervene_id==1)
		map(j,stim) = max(0,intrvn_type * map(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (map(j,stim) + intrvn_strength(j)));
	end;
	if any(intervene_id==2)
		mav(j,stim) = max(0,intrvn_type * mav(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (mav(j,stim) + intrvn_strength(j)));
	end;
	
  % Compute reward prediction
	mdiff(stim) = map(j,stim) - mav(j,stim);
  
  % Force decision
  decision(j) = stim;
 
  dap(j) = max(0,wkdap * s(:,decision(j)) + wmdap * mav(j,decision(j)) + max(0,r(j,decision(j))));
  dav(j) = max(0,wkdav * s(:,decision(j)) + wmdav * map(j,decision(j)) - min(0,r(j,decision(j))));
    
  % Apply intervention (if any)
  if any(intervene_id==3)
    dap(j) = max(0,intrvn_type * dap(j) * intrvn_strength(j) + (1-intrvn_type) * (dap(j) + intrvn_strength(j)));
  end;
  if any(intervene_id==4)
    dav(j) = max(0,intrvn_type * dav(j) * intrvn_strength(j) + (1-intrvn_type) * (dav(j) + intrvn_strength(j)));
  end;

  % Update KC->MBON synaptic weights (except on last trial)
	if flag_plasticity_rule>0
		wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' * (lambda - dav(j)));
		wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' * (lambda - dap(j)));
	else
    wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' .* (wkdav*s(:,decision(j)) - dav(j)));
    wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' .* (wkdap*s(:,decision(j)) - dap(j)));
	end;
end;

%%%
%%% Test
%%%
for j=(2*nt/3 + 1):nt
	for stim=1:no
    % Compute MBON firing rates
		map(j,stim) = wkmap(:,:,j) * s(:,stim);
		mav(j,stim) = wkmav(:,:,j) * s(:,stim);
		
    % Apply intervention (if any)
		if any(intervene_id==1)
			map(j,stim) = max(0,intrvn_type * map(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (map(j,stim) + intrvn_strength(j)));
		end;
		if any(intervene_id==2)
			mav(j,stim) = max(0,intrvn_type * mav(j,stim) * intrvn_strength(j) + (1-intrvn_type) * (mav(j,stim) + intrvn_strength(j)));
		end;
		
    % Compute reward prediction
		mdiff(stim) = map(j,stim) - mav(j,stim);
	end;
  
  % Make decision
  decision(j) = double(rand < exp(mdiff(2)*beta)/(exp(mdiff(1)*beta)+exp(mdiff(2)*beta))) + 1;
  
  dap(j) = max(0,wkdap * s(:,decision(j)) + wmdap * mav(j,decision(j)) + max(0,r(j,decision(j))));
  dav(j) = max(0,wkdav * s(:,decision(j)) + wmdav * map(j,decision(j)) - min(0,r(j,decision(j))));
    
  % Apply intervention (if any)
  if any(intervene_id==3)
    dap(j) = max(0,intrvn_type * dap(j) * intrvn_strength(j) + (1-intrvn_type) * (dap(j) + intrvn_strength(j)));
  end;
  if any(intervene_id==4)
    dav(j) = max(0,intrvn_type * dav(j) * intrvn_strength(j) + (1-intrvn_type) * (dav(j) + intrvn_strength(j)));
  end;

  % Update KC->MBON synaptic weights (except on last trial)
  if j<nt
    if flag_plasticity_rule>0
      wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' * (lambda - dav(j)));
      wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' * (lambda - dap(j)));
    else
      wkmap(:,:,j+1) = max(0,wkmap(:,:,j) + epskm * s(:,decision(j))' .* (wkdav*s(:,decision(j)) - dav(j)));
      wkmav(:,:,j+1) = max(0,wkmav(:,:,j) + epskm * s(:,decision(j))' .* (wkdap*s(:,decision(j)) - dap(j)));
    end;
  end;
end;

% Create output struct
out.map = map;
out.mav = mav;
out.go = map;
out.nogo = mav;
out.dap = dap;
out.dav = dav;
out.wkmap = wkmap;
out.wkmav = wkmav;
out.decision = decision;
out.r = r;
out.s = s;