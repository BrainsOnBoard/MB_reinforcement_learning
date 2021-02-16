function all_actions = generate_actions(circles)
%
% A function to determine the actions necessary to trace the random learning routes
%
% Inputs:
%  circles - a cell array of circle coordinates
%
% Outputs:
%  actions - a cell array of the actions necessary to trace each learning
%            route
  num_circles = length(circles(:, 1));
  all_actions = cell(num_circles, 1); % empty cell array to store circle actions of varying sizes
  for cir = 1 : num_circles
    N = length(circles{cir, 1}); % number of circle's coordinates
    actions = zeros(N-1, 1); % vector to store actions
    x = circles{cir}(:,1); 
    y = circles{cir}(:,2);
    for c = 1 : N-1  
      dx = x(c+1) - x(c);
      dy = y(c+1) - y(c);
      if dx == 1 && dy == 0 % if x increases by 1
        actions(c, 1) = 1; % agent moves right
      elseif dx == -1 && dy == 0 % if x decreases by 1
        actions(c, 1) = 3; % agent moves left
      elseif dy == 1 && dx == 0 % if y increases by 1
        actions(c, 1) = 2; % agent moves up
      elseif dy == -1 && dx == 0 % if y decreases by 1
        actions(c, 1) = 4; % agent moves down
      else
        fprintf('Circle %d\n',c);
        error('???: generate_actions -- diagonal move found but not permitted. Problem with generate_circles.');
      end
    end
    all_actions{cir, 1} = actions; % add circle actions to cell array
  end
end