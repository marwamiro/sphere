 function initsetup(plotfigure)
% initsetup() creates a model setup of particles in 3D with cubic
% packing. Specify PSD and desired number of particles, and the 
% function will determine the model dimensions, and fill the space 
% with a number of particles from a specified particle size distribution.
close all;

% Simulation project name
simulation_name = '1e3-init-visc';

% Plot particle assembly in MATLAB after initialization
if exist('plotfigure','var')
    if plotfigure == 1
        visualize = 1; 
    else
        visualize = 0;
    end
else
    visualize = 0;
end
 
% Physical, constant parameters
params.g = [0.0, 0.0, -9.80665]; %standard gravity, by definition 9.80665 m/s^2
%params.g = 0.98;

% No. of dimensions
grids.nd = 3;
grids.origo = [0 0 0]; %Coordinate system origo

% Number of particles
p.np = 1e3;

% Create grid
form = 4; % For a cubic grid, use 3.
          % For a higher grid, use 4 or more (for cubic end config. use 5).
          % For a flatter grid, use 1 or 2.
grids.num(1) = ceil(nthroot(p.np, form)); % Particles in x direction
grids.num(2) = grids.num(1);  % Particles in y direction
grids.num(3) = ceil(p.np/(grids.num(1)*grids.num(2))); % Particles in z direction

disp(['Grid dimensions: x=' num2str(grids.num(1)) ...
      ', y=' num2str(grids.num(2)) ...
      ', z=' num2str(grids.num(3))])

%grids.num = ceil(nthroot(p.np,grids.nd)) * ones(grids.nd, 1); %Square/cubic
p.np = grids.num(1)*grids.num(2)*grids.num(3);

% Particle size distribution
p.psd = 'logn'; %PSD: logn or uni

p.m = 440e-6;       %Mean size
p.v = p.m*0.00002;  %Variance

p.radius = zeros(p.np, 1);
p.x      = zeros(p.np, grids.nd);

p.bonds  = ones(p.np, 4) * p.np; % No bonds when the values equal the no. of particles

if strcmp(p.psd,'logn') %Log-normal PSD. Note: Keep variance small.
    mu = log((p.m^2)/sqrt(p.v+p.m^2));
    sigma = sqrt(log(p.v/(p.m^2)+1));
    p.radius = lognrnd(mu,sigma,1,p.np); %Array of particle radii 
elseif strcmp(p.psd,'uni') %Uniform PSD between rmin and rmax
    rmin = p.m - p.v*1e5;
    rmax = p.m + p.v*1e5;
    %rmin = 0.1*dd; rmax = 0.4*dd;
    p.radius = (rmax-rmin)*rand(p.np,1)+rmin; %Array of particle radii
end

% Display PSD
if visualize == 1
    figure;
    hist(p.radius);
    title(['PSD: ' num2str(p.np) ' particles, m = ' num2str(p.m) ' m']); 
end

% Friction angles
ang_s = 30; % Angle of static shear resistance
ang_d = 25; % Angle of dynamic shear resistance
ang_r = 35; % Angle of rolling resistance

% Other particle parameters
p.vel    = zeros(p.np, grids.nd); % Velocity vector
p.angvel = zeros(p.np, grids.nd); % Angular velocity
p.fixvel = zeros(p.np, 1); 	  % 0: horizontal particle velocity free, 1: hotiz. particle velocity fixed
p.xsum   = zeros(p.np, 1);	  % Total displacement along x-axis
p.force  = zeros(p.np, grids.nd); % Force vector
p.torque = zeros(p.np, grids.nd); % Torque vector

params.global = 1; % 1: Physical parameters global, 0: individual per particle
p.rho     = 3600*ones(p.np,1);    % Density
p.k_n     = 4e5*ones(p.np,1);     % Normal stiffness [N/m]
p.k_s	  = p.k_n(:);		  % Shear stiffness [N/m]
p.k_r	  = p.k_s(:).*(10);	  % Rolling stiffness
params.shearmodel = 1;	 	  % Contact model. 1=frictional, viscous, 2=frictional, elastic
p.gamma_s = p.k_n./1e3;  	  % Shear viscosity [Ns/m]
p.gamma_r = p.gamma_s(:);	  % Rolling viscosity [?]
p.mu_s    = tand(ang_s)*ones(p.np,1);     % Inter-particle shear contact friction coefficient [0;1[
p.mu_r    = tand(ang_r)*ones(p.np,1);     % Inter-particle rolling contact friction coefficient [0;1[
p.C       = 0*zeros(p.np,1);      % Inter-particle cohesion
p.E       = 10e9*ones(p.np,1);    % Young's modulus
p.K       = 38e9*ones(p.np,1);    % Bulk modulus
p.nu      = 0.1*2.0*sqrt(min(4/3*pi.*p.radius(:).*p.radius(:).*p.radius(:).*p.rho(1))*p.k_n(1))*ones(p.np,1);      % Poisson's ratio (critical damping: 2*sqrt(m*k_n)). Normal component elastic if nu=0
p.es_dot  = zeros(p.np,1);        % Current shear energy dissipation
p.es      = zeros(p.np,1);        % Total shear energy dissipation
p.p       = zeros(p.np,1);        % Pressure on particle

% Parameters related to capillary bonds
% Disable capillary cohesion by setting kappa to zero.
enableCapillaryCohesion = 0;
theta = 0.0;             % Wettability (0: perfect)
if (enableCapillaryCohesion == 1) 
    params.kappa = 2*pi*p.gamma_s(1) * cos(theta); % Prefactor
    params.V_b = 1e-12;  % Liquid volume at bond
else   
    params.kappa = 0;    % Zero capillary force
    params.V_b = 0;      % Zero liquid volume at bond
end
params.db = (1.0 + theta/2.0) * params.V_b^(1.0/3.0); % Debonding distance

% Time parameters
%time.dt         = 0.2*min(sqrt((p.rho(:).*p.radius(:).^2)./p.K(:))); % Computational delta t
%time.dt         = time.dt*1e1; % Speedup factor
time.dt         = 0.17*sqrt(min(4/3*pi.*p.radius(:).*p.radius(:).*p.radius(:).*p.rho(1))/p.k_n(1)); % Computational time step (O'Sullivan et al. 2003)
time.current    = 0.0;
time.total      = 1.500+2.0*time.dt; % Total simulation time [s]
time.file_dt    = 0.0010; % Interval between output#.bin generation [s]
time.step_count = 0;

% Calculate particle coordinates
%  Grid unit length. Maximum particle diameter determines grid size
GU = 2*max(p.radius)*1.40; % Forty percent margin
grids.L = [GU*grids.num(1) GU*grids.num(2) GU*grids.num(3)];

% Particle coordinates by filling grid.
x = linspace(GU/2, grids.L(1)-GU, grids.num(1));
y = linspace(GU/2, grids.L(2)-GU, grids.num(2));
z = linspace(GU/2, grids.L(3)-GU, grids.num(3));

[X Y Z] = meshgrid(x,y,z);
X=X(:); Y=Y(:); Z=Z(:);

p.x = [X Y Z]; 

%Particle positions randomly modified by + 10 percent of a grid unit
p.x = p.x + (rand(p.np, grids.nd)*0.49*GU);

% Walls with friction
%   Note that the world boundaries already act as fricionless boundaries
% Upper wall
wno = 1;
walls.n(wno,:)     = [0.0, 0.0, -1.0];   % Normal along negative z-axis
walls.x(wno)       = grids.L(3);  % Positioned at upper boundary
walls.m(wno)       = p.rho(1)*p.np*pi*max(p.radius)*max(p.radius)*max(p.radius); % Wall mass
walls.vel(wno)     = 0.0;   % Wall velocity
walls.force(wno)   = 0.0;   % Wall force
walls.devs(wno)    = 0.0;  % Deviatoric stress
walls.nw = wno;

% Define behavior of x- and y boundaries.
% Uncomment only one!
%params.periodic = 0; % Behave as frictional walls
params.periodic = 1; % Behave as periodic boundaries
%params.periodic = 2; % x: periodic, y: frictional walls

% Write output binary
fwritebin('../input/', [simulation_name '.bin'], p, grids, time, params, walls);
disp('Writing of binary file complete.');

% Plot particles in bubble plot
if visualize == 1
    disp('Waiting for visualization.');
    plotspheres(p, grids, 5, 0, 1);
end

disp(['Project "' simulation_name '" is now ready for processing with SPHERE.']);
disp(['Call "' pwd '/sphere ' simulation_name '" from the system terminal ']);
disp(['to initiate, and check progress here in MATLAB using "status(''' simulation_name ''')"']);

end
