function plotspheres(p, grids, n, quiver, visible)

if exist('visible','var')
    if visible == 0
        % Create a new invisible figure window
        figure('visible','off');
    else
        % Create a new figure window
        figure('Renderer','OpenGL')
    end
else
    % Create a new figure window
    figure('Renderer','OpenGL')
end

% Generate unit sphere, consisting of n-by-n faces (i.e. the resolution)
if exist('n', 'var')
    [x,y,z] = sphere(n);
else
    [x,y,z] = sphere(10);
end

% Iterate through particle data
hold on
for i=1:p.np
    spheresurf = surf(x*p.radius(i)+p.x(i,1), ...
                      y*p.radius(i)+p.x(i,2), ...
                      z*p.radius(i)+p.x(i,3));
    set(spheresurf,'EdgeColor','none', ...
                   'FaceColor',[0.96 0.64 0.38], ...  % RGB
                   'FaceLighting','phong', ... 
                   'AmbientStrength',0.3, ... 
                   'DiffuseStrength',0.8, ... 
                   'SpecularStrength',0.9, ... 
                   'SpecularExponent',25, ... 
                   'BackFaceLighting','lit'); 
end
camlight left;
hidden off

% Optional quiver3 plot (velocity vectors)
if exist('quiver','var')
    if quiver == 1
        quiver3(p.x(:,1),    p.x(:,2)  , p.x(:,3), ...
                p.vel(:,1), p.vel(:,2), p.vel(:,3), 3);
    end
end

% Draw walls from 8 vertices and connect these to 6 faces
vertices = [grids.origo(1) grids.origo(2) grids.origo(3); ... % 1
            grids.origo(1) grids.L(2)     grids.origo(3); ... % 2
            grids.L(1)     grids.L(2)     grids.origo(3); ... % 3
            grids.L(1)     grids.origo(2) grids.origo(3); ... % 4
            grids.origo(1) grids.origo(2) grids.L(3);     ... % 5
            grids.origo(1) grids.L(2)     grids.L(3);     ... % 6
            grids.L(1)     grids.L(2)     grids.L(3);     ... % 7
            grids.L(1)     grids.origo(2) grids.L(3)];        % 8
            
faces = [1 2 3 4; ... %  (observing along pos. y axis direction)
         2 6 7 3; ... % 
         4 3 7 8; ... % 
         1 5 8 4; ... % 
         1 2 6 5; ... % 
         5 6 7 8];    % 

patch('Faces', faces, 'Vertices', vertices, ...
      'FaceColor','none', 'EdgeColor','black','LineWidth',2);

% View specifications
%daspect([1 1 1])

view([grids.L(1), -2*grids.L(2), grids.L(3)])
grid on
axis equal
maxr = max(p.radius);
axis([grids.origo(1)-maxr grids.L(1)+maxr ... 
      grids.origo(2)-maxr grids.L(2)+maxr ...
      grids.origo(3)-maxr grids.L(3)+maxr]);

light('Position', [grids.L(1), -grids.L(2), grids.L(3)]);
material shiny; %shiny dull metal
camlight(45,45);
lighting gouraud; %flat gouraud phone none

% Remove hidden lines from mesh plot (hidden on)
hidden off

% Labels
title([num2str(p.np) ' particles']);
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');

% Add delaunay triangulation
%dt = DelaunayTri(p.x(:,1), p.x(:,2), p.x(:,3));
%tetramesh(dt,'FaceAlpha',0.02);

end