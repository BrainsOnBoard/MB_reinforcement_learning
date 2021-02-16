function [circles, varargout] = generate_circles(N, nx)
%
% A function to generate random learning routes of increasing radius size
%
% Inputs:
%    N - number of circles to generate
%   nx - length of the arena
%
% Outputs:
%  circles - a cell array of circle coordinates

centre = ceil(nx/2); % the centre of the arena
r = 2:(nx/4-2)/3:nx/4; % Circle radii

r_ind = randi(length(r),N,1); % generate random indices
r = r(r_ind); % to randomly rearrange the radius order
theta = linspace(0, 2*pi*((N-1)/N), N); % equally distant angles based on N
circles = cell(N,1); % empty cell array to store circles of varying coordinate sizes
noise = rand(N,1) * 2*pi/N; % Generate noise for theta

for i = 1 : N
  r_i = r(i);
  theta_i = theta(i) + noise(i); % which is applied to theta
  x_i = zeros(10000, 1);
  y_i = zeros(10000, 1);
  j = 1;
  for alpha = 0 : 2*pi/9999 : 2*pi
    x_i(j) = r_i * cos(theta_i - pi + alpha);
    y_i(j) = r_i * sin(theta_i - pi + alpha);
    j = j + 1;
  end
  x_i = x_i + r_i * cos(theta_i) + centre + 0.5;
  y_i = y_i + r_i * sin(theta_i) + centre + 0.5;
  xy = floor([x_i, y_i]); % connect two column vectors to make array
  xy = unique(xy, 'rows', 'stable'); % filter out duplicates
  xy = [xy; centre, centre]; % Add nest location to start of route
  % Check there are no diagonals on circle path
  dxy = diff(xy,1,1);
  while any(abs(dxy(:,1)) & abs(dxy(:,2)))
    ind = find(dxy(:,1) & dxy(:,2),1,'first'); % Where does diagonal begin
    xy = [xy(1:ind,:); xy(ind+1,1), xy(ind,2); xy(ind+1:end,:)]; % Add new coordinate
    dxy = diff(xy,1,1);
  end
  % Check for step sizes >1 
  if any(abs(dxy(:))>1)
    error('??? generate_circles: Step size >1');
  end
  circles{i, 1} = xy; % add circle coordinates to cell array
end

if nargout>1
  varargout{1} = noise;
end
% plot circles
% for k=1:length(c) plot(c{k}(:,1),c{k}(:,2)); hold on; end; hold off; set(gca,'xlim',[1 51],'ylim',[1 51]);