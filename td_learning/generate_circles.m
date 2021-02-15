function circles = generate_circles(N, length)
%
% A function to generate random learning routes of increasing radius size
%
% Inputs:
%        N - number of circles to generate
%   length - length of the arena
%
% Outputs:
%  circles - a cell array of circle coordinates

  centre = ceil(length/2); % the centre of the arena
  r = [2, 4, 6, 8, 10, 12]; % TODO generate these based on N
  r_ind = randperm(6); % generate random indices
  r = r(r_ind); % to randomly rearrange the radius order
  theta = linspace(0, 2*pi*((N-1)/N), N); % equally distant angles based on N
  circles = cell(N,1); % empty cell array to store circles of varying coordinate sizes

  for i = 1 : N
    r_i = r(i);
    noise = rand * (2*pi)/N; % generate noise
    theta_i = theta(i) + noise; % which is applied to theta
    x_i = zeros(5000, 1);
    y_i = zeros(5000, 1);
    j = 1;
    for alpha = 0 : 2*pi/4999 : 2*pi
      x_i(j) = r_i * cos(theta_i - pi + alpha);
      y_i(j) = r_i * sin(theta_i - pi + alpha);
      j = j + 1;
    end
    x_i = x_i + r_i * cos(theta_i) + centre;
    y_i = y_i + r_i * sin(theta_i) + centre;
    xy = floor([x_i, y_i]); % connect two column vectors to make array
    xy = unique(xy, 'rows', 'stable'); % filter out duplicates
    circles{i, 1} = xy; % add circle coordinates to cell array
  end
end

% plot circles
% grayImage = 128 * ones(rows, columns, 'uint8'); for j = 1 : N circ = circles{j, 1}; x = circles{j}(:,1); y = circles{j}(:,2); M = length(circ); for k = 1 : M grayImage(y(k), x(k)) = 255; end; end; imshow(grayImage); axis('on', 'image'); fprintf('Done!\n');