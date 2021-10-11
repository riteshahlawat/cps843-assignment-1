% STEP 1
a = 0.45;
T = maketform('affine', [1 0 0; a 1 0; 0 0 1]);
A = imread('ball.jpg');

h1 = figure; 
imshow(A); 
title('Original Image');

orange = [255 127 0]';
R = makeresampler({'cubic','nearest'},'fill');
B = imtransform(A,T,R,'FillValues',orange); 

h2 = figure; 
imshow(B);
title('Sheared Image');

% STEP 2

[U,V] = meshgrid(0:64:320,0:64:256);
[X,Y] = tformfwd(T,U,V);
gray = 0.65 * [1 1 1];

figure(h1);
hold on;
line(U, V, 'Color',gray);
line(U',V','Color',gray);

figure(h2);
hold on;
line(X, Y, 'Color',gray);
line(X',Y','Color',gray);

gray = 0.65 * [1 1 1];
for u = 0:64:320
    for v = 0:64:256
        theta = (0 : 32)' * (2 * pi / 32);
        uc = u + 20*cos(theta);
        vc = v + 20*sin(theta);
        [xc,yc] = tformfwd(T,uc,vc);
        figure(h1); line(uc,vc,'Color',gray);
        figure(h2); line(xc,yc,'Color',gray);
    end
end

% STEP 3

R = makeresampler({'cubic','nearest'},'fill');

Bf = imtransform(A,T,R,'XData',[-49 500],'YData',[-49 400],...
                 'FillValues',orange);

figure, imshow(Bf);
title('Pad Method = ''fill''');

R = makeresampler({'cubic','nearest'},'replicate');
Br = imtransform(A,T,R,'XData',[-49 500],'YData', [-49 400]);

figure, imshow(Br);
title('Pad Method = ''replicate''');

R = makeresampler({'cubic','nearest'}, 'bound');
Bb = imtransform(A,T,R,'XData',[-49 500],'YData',[-49 400],...
                 'FillValues',orange);
figure, imshow(Bb);
title('Pad Method = ''bound''');

% STEP 4

Thalf = maketform('affine',[1 0; a 1; 0 0]/2);

R = makeresampler({'cubic','nearest'},'circular');
Bc = imtransform(A,Thalf,R,'XData',[-49 500],'YData',[-49 400],...
                 'FillValues',orange);
figure, imshow(Bc);
title('Pad Method = ''circular''');

R = makeresampler({'cubic','nearest'},'symmetric');
Bs = imtransform(A,Thalf,R,'XData',[-49 500],'YData',[-49 400],...
                 'FillValues',orange);
figure, imshow(Bs);
title('Pad Method = ''symmetric''');

