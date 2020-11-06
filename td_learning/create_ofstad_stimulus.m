z = load('ofstad_etal_arena.mat');
nx = 20; 
ny = 20;
scale_f = z.d / (nx + 1);
im = getviewfast(0, 0, 0, 0/180*pi, z.X, z.Y, z.Z);
im_scale = 1/(sqrt(numel(im)/400));
im_r = imresize(im, im_scale, 'bilinear');
nk = numel(im_r);
s = zeros(nk, ny, nx, 4);
th = [0, 90, 180, 270];
num_th = length(th);
for j=1:ny
  for k=1:nx
      for th_ind=1:num_th
        im = getviewfast(k*scale_f-(z.d/2), j*scale_f-(z.d/2), 0, th(th_ind)/180*pi, z.X, z.Y, z.Z);
        im_r = imresize(im, im_scale, 'bilinear');
        s(:, j, k, th_ind) = im_r(:);
      end  
  end
end
save('ofstad_stimulus.mat', 's', 'nk');