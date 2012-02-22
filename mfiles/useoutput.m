%function useoutput(fn, plotfigure)
% Use particle data from target binary from output directory,
% create a fitting grid, and zero time variables.
% Final configuration is saved in a new filename in input/ directory.

% Plot particle assembly in MATLAB after initialization
visualize = 0;  % No
%visualize = 1;  % Yes

% Input binary file
directory = '../input/';
inputbin  = '5e4-cons_10kPa.bin';

% New configuration name
simulation_name = '5e4-shear_10kPa';

% Read data
[p, grids, time, params, walls] = freadbin(directory, inputbin);
%[p, grids, time, params, walls] = freadbinold(directory, inputbin);

% Change temporal parameters (comment out to use previous values"
time.current    = 0.0;         % New starting time
time.step_count = 0;           % First output file number
time.total      = 0.25+2*time.dt; % Total simulation time [s]
time.file_dt    = 0.01; %Interval between output#.bin generation [s]
time.dt         = 0.17*sqrt(min(4/3*pi.*p.radius(:).*p.radius(:).*p.radius(:).*p.rho(1))/p.k_n(1)); % Computational time step (O'Sullivan et al. 2003)

% Particle stiffness
p.k_n = 4e5*ones(p.np,1); % Normal stiffness [N/m]
p.gamma_s = p.k_n./1e3;	  % Shear viscosity [Ns/m]
p.gamma_r = p.gamma_s(:); % Rolling viscosity

% Friction angles
ang_s = 25; % Angle of shear resistance
ang_r = 35; % Angle of rolling resistance
p.mu_s = tand(ang_s)*ones(p.np,1); % Inter-particle shear contact friction coefficient [0;1[
p.mu_r = tand(ang_r)*ones(p.np,1); % Inter-particle rolling contact friction coefficient [0;1[

% Compute new grid, scaled to fit max.- & min. particle positions
GU = 2*max(p.radius); % Cell size
x_min = min(p.x(:,1));% - p.radius(:));
x_max = max(p.x(:,1));% + p.radius(:));%*3.0;
%y_min = min(p.x(:,2) - p.radius(:));
%y_max = max(p.x(:,2) + p.radius(:));
z_min = min(p.x(:,3) - p.radius(:));
z_max = max(p.x(:,3) + p.radius(:));
z_adjust = 1.2; % Specify overheightening of world to allow for shear dilatancy
%grids.num(1) = ceil((x_max-x_min)/GU);
%grids.num(2) = ceil((y_max-y_min)/GU);
grids.num(3) = ceil((z_max-z_min)*z_adjust/GU);
%grids.L = [(x_max-x_min) (y_max-y_min) (z_max-z_min)*z_adjust];
grids.L(3) = (z_max-z_min)*z_adjust;

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

% Move mobile upper wall to top of domain
walls.x(1) = max(p.x(:,3)+p.radius(:));

% Define new deviatoric stress [Pa]
%walls.devs(1) = 0.0;
walls.devs(1) = 10.0e3;

% Let x- and y boundaries be periodic
params.periodic = 1; % x- and y boundaries periodic
%params.periodic = 2; % Only x-boundaries periodic

% By default, all particles are free to move horizontally
p.fixvel = zeros(p.np, 1); % Start off by defining all particles as free
shearing = 1; % 1: true, 0: false

if shearing == 1
  % Fix horizontal velocity to 0.0 of lowermost particles
  I = find(p.x(:,3) < (z_max-z_min)*0.1); % Find the lower 10%
  p.fixvel(I) = 1;  % Fix horiz. velocity
  p.vel(I,1) = 0.0; % x-value
  p.vel(I,2) = 0.0; % y-value

  % Fix horizontal velocity to 0.0 of uppermost particles
  I = find(p.x(:,3) > (z_max-z_min)*0.9); % Find the upper 10%
  p.fixvel(I) = 1;  % Fix horiz. velocity
  p.vel(I,1) = (x_max-x_min)*1.0; % x-value: One grid length per second
  p.vel(I,2) = 0.0; % y-value
end

% Zero x-axis displacement
p.xsum = zeros(p.np, 1);

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

