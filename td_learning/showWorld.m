function h=showWorld(X,Y,Z)
% Example use:
%       >> z=load('ofstad_etal_arena.mat');
%       >> h=showWorld(z.X, z.Y, z.Z);

clf
h=fill3(X',Y',Z','k');
axis equal