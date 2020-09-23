function im=getviewfast(x,y,z,th,X,Y,Z,im_size,el_max,orig_im_size,pitch)
% 
% Inputs:
%       x, y, z = coordinates
%       th = orientation in radians (so use: x/180*pi)
%       X, Y, Z = coordinate space from showWorld
%
% Example use:
%       >> z=load('ofstad_etal_arena.mat');
%       >> im = getviewfast(0.05, 0, 0, 0/180*pi, z.X, z.Y, z.Z);
%       >> imshow(im);

    if nargin < 10 || isempty(orig_im_size)
        orig_im_size = [170 900];
    end
    if nargin < 9 || isempty(el_max)
        el_max = 5*pi/12;
    else
        el_max = pi*el_max/180;
    end

    if nargin < 11
        pitch = el_max/2;
    else
        pitch = pitch*(pi/180);
    end

    if nargin < 8
        im_size = [];
    else
        switch numel(im_size)
            case 0 % do nothing
            case 1
                if im_size==1
                    im_size = [];
                else
                    im_size = orig_im_size./im_size;
                end
            case 2
                if all(im_size==orig_im_size)
                    im_size = [];
                end
            otherwise
                error('invalid im_size')
        end
    end

    is2d = size(X,2)>1;
    if is2d
        nshapes = size(X,1);
    else
        nind = [0;find(isnan(X))];
        nshapes = length(nind)-1;
        
        X = X';
        Y = Y';
        Z = Z';
    end

    im = false(orig_im_size);

    [az1,el] = cart2sph(X-x,Y-y,Z-z);
    halfim = (orig_im_size(2)-1)/2;
    az = mod(round((1-orig_im_size(2))*(pi+az1)/(2*pi)),orig_im_size(2));
    el = round((orig_im_size(1)-1)*(0.5+(pitch-el)/el_max));
%     if ~is2d
%         cazmid_all = 1+mod((orig_im_size(2)-1)* ...
%             (th-atan2((Y+circshift(Y,-1))/2-y,(X+circshift(X,-1))/2-x)+pi)/(2*pi), ...
%             orig_im_size(2)-1);
%     else
        az = az(:)';
        el = el(:)';
%     end

    for i = 1:nshapes
    %     fprintf('i: %d\n',i);
        if is2d
            cind = sub2ind(size(X),i*ones(1,size(X,2)),1:size(X,2));
%             cazmid = cazmid_all(cind);
        else
            cind = nind(i)+1:nind(i+1)-1;
%             cX = X(cind)-x;
%             cY = Y(cind)-y;
%             cazmid = mod(orig_im_size(2)* ...
%                 (th-atan2((cY+circshift(cY,-1))/2,(cX+circshift(cX,-1))/2)+pi)/(2*pi), ...
%                 orig_im_size(2));
        end
        
        cel = el(cind);
        his = cel > orig_im_size(1)-1;
        los = cel < 0;
%         if all(hiorlo)
%             continue;
%         end
        
        hi2hi = his & his([2:end,1]);
        lo2lo = los & los([2:end,1]);
        elouts = ~hi2hi & ~lo2lo;
        
        if all(~elouts)
            continue;
        end
        
        cel = cel(elouts);
        cind = cind(elouts);
        caz = round(az(cind));
        
        cel = max(0,min(cel,orig_im_size(1)-1));
        
        azlo = min(caz);
        ello = min(cel);
        cimwd = max(caz)-azlo+1;
        cimht = max(cel)-ello+1;
        if cimwd==1 || cimht==1
            continue;
        end
        
        caz2 = caz([2:end,1]);
        
        cazsign = sign(caz2-caz);

        gthlf1 = caz > halfim;
        gthlf2 = caz2 > halfim;
        cX = X(cind);
        cazsel = gthlf2~=gthlf1 & (cX([2:end,1]) < 0 | cX < 0);
        
        ello = min(cel);
        cimht = max(cel)-ello+1;
        if cimwd==1 || cimht==1
            continue;
        end
        if any(cazsel)
            y1 = Y(cind);
            y2 = y1([2:end,1]);
            cazsign(cazsel) = sign(y2(cazsel)-y1(cazsel));
            
            azlo = 0;
            cim = false(cimht,orig_im_size(2));
        else
            azlo = min(caz);
            cim = false(cimht,max(caz)-azlo+1);
        end
        cimwd = size(cim,2);
        
        celsign = sign(cel([2:end,1])-cel);
        
        %% Bresenham's line algorithm
        del = cel([2:end,1])-cel;
        derr = abs(del./(caz([2:end,1])-caz));
        caz = 1-azlo+[caz,caz(1)];
        cel = 1-ello+[cel,cel(1)];
        for j = 1:length(caz)-1
            if del(j)==0 % horizontal line or single pixel
                if cazsign(j)==0
                    cim(cel(j),caz(j)) = true;
                else
                    cim(cel(j),fillvals(caz(j),caz(j+1),cazsign(j),orig_im_size(2))) = true;
                end
            elseif isinf(derr(j)) % vertical line
                cim(fillvals(cel(j),cel(j+1),celsign(j),orig_im_size(1)),caz(j)) = true;
            else
                err = 0;
                y = cel(j);
                for x = fillvals(caz(j),caz(j+1),cazsign(j),orig_im_size(2))
                    cim(y,x) = true;
                    err = err+derr(j);
                    while err >= 0.5
                        cim(y,x) = true;
                        y = max(1,min(cimht,y+sign(del(j))));
                        err = err-1;
                    end
                end
            end
            
%             if ~all(size(cim)==[cimht,cimwd])
%                 keyboard
%             end

%             figure(10);clf
%             imshow(cim)
%             keyboard
        end
        
        if sum(any(cim)) <= 1 || sum(any(cim,2)) <= 1 % vertical/horizontal line only
            continue;
        end
        
        %% fill in triangle
        cimfill = false(size(cim));
        
        dcim = diff([false(1,cimwd);cim])==1;
        only1edge = sum(dcim)==1;
        
        cimfill(:,only1edge) = cim(:,only1edge);
        cimfill(:,~only1edge) = cim(:,~only1edge) | mod(cumsum(dcim(:,~only1edge)),2)==1;

%         figure(1);clf
%         subplot(2,1,1)
%         imshow(cimfill)
%         subplot(2,1,2)
%         imagesc(diff([false(1,orig_im_size(2));cim]))
%         colorbar
%         keyboard
        
        %% add to master image
        eli = ello+(1:cimht);
        azi = azlo+(1:cimwd);
        im(eli,azi) = im(eli,azi) | cimfill;
        
%         figure(1);clf
%         subplot(3,1,1)
%         imshow(cim)
%         subplot(3,1,2)
%         imshow(cimfill)
%         subplot(3,1,3)
%         imshow(im)
%         keyboard
    end
    
    % rotate horizontally
    im = circshift(~im,[0 round(mod(th*(orig_im_size(2)-1)/(2*pi),orig_im_size(2)))]);

    % averaging if needed
    horizel = round(size(im,1)*max(0,el_max/2 - pitch)/el_max);
    if horizel>0 || ~isempty(im_size)
        im = im2double(im);

%         horizrows = (size(im,1)-horizel):size(im,1);
%         im(horizrows,:) = max(0,im(horizrows,:)-0.5);

        if ~isempty(im_size)
            im = imresize(im,im_size,'bilinear');
%             im=blockproc(im,av_area,@(x)mean2(x.data));
        end
    end
end

function vals=fillvals(from,to,sgn,lim)
    if sgn==0
        vals = from;
    else
        truesgn = sign(to-from);
        if truesgn==sgn
            vals = from:sgn:to;
        elseif sgn==-1
            vals = [1:from, to:lim];
        else
            vals = [1:to, from:lim];
        end
    end
end
