function [p, grids, time, params, walls] = freadbin(path, fn)
    %path = '../output/'; %Target directory
    %fn   = 'output0.bin'; %Target binary
    
    %Open binary file for reading with little endian ordering and unicode char encoding.
    %fid = fopen([path fn],'rb','ieee-le','UTF-8'); 
    fid = fopen([path fn], 'rb', 'ieee-le');

    % Read the number of dimensions and particles
    grids.nd = fread(fid, 1, 'int');
    p.np     = fread(fid, 1, 'uint32');
    
    % Read the time variables
    time.dt         = fread(fid, 1, 'float');
    time.current    = fread(fid, 1, 'double');
    time.total      = fread(fid, 1, 'double');
    time.file_dt    = fread(fid, 1, 'float');
    time.step_count = fread(fid, 1, 'uint32');
    
    %Initiate variables for faster execution
    p.x       = zeros(p.np, grids.nd); % Coordinate vector
    p.vel     = zeros(p.np, grids.nd); % Velocity vector
    p.angvel  = zeros(p.np, grids.nd); % Acceleration vector
    p.force   = zeros(p.np, grids.nd); % Linear force vector
    p.torque  = zeros(p.np, grids.nd); % Rotational torque vector
    p.fixvel  = zeros(p.np, 1);	       % Fix horizontal particle velocity: 1: yes, 0: no
    p.xsum    = zeros(p.np, 1);	       % Total particle displacement along x-axis
    p.radius  = zeros(p.np, 1);	       % Particle radius
    p.rho     = zeros(p.np, 1);        % Density
    p.k_n     = zeros(p.np, 1);        % Normal stiffness
    p.k_s     = zeros(p.np, 1);	       % Shear stiffness
    p.k_r     = zeros(p.np, 1);	       % Rolling stiffness
    p.gamma_s = zeros(p.np, 1);        % Shear viscosity
    p.gamma_r = zeros(p.np, 1);	       % Rolling viscosity
    p.mu_s    = zeros(p.np, 1);        % Inter-particle contact shear friction coefficient
    p.mu_r    = zeros(p.np, 1);        % Inter-particle contact rolling friction coefficient
    p.C       = zeros(p.np, 1);        % Inter-particle cohesion
    p.E       = zeros(p.np, 1);        % Young's modulus
    p.K       = zeros(p.np, 1);        % Bulk modulus
    p.nu      = zeros(p.np, 1);        % Poisson's ratio
    p.es_dot  = zeros(p.np, 1);        % Current shear energy dissipation
    p.es      = zeros(p.np, 1);        % Total shear energy dissipation
    p.p       = zeros(p.np, 1);        % Pressure on particle
    p.bonds   = zeros(p.np, 2);	       % Inter-particle bonds

    % Read remaining data from MATLAB binary
    grids.origo = fread(fid, [1, grids.nd], 'float');
    grids.L     = fread(fid, [1, grids.nd], 'float');
    grids.num   = fread(fid, [1, grids.nd], 'uint32');

    for j=1:p.np
        %p.id = fread(fid, 1, 'uint32');
        
        for i=1:grids.nd %Coordinates, velocity, acceleration, pre-velocity    
            p.x(j,i)      = fread(fid, 1, 'float');
            p.vel(j,i)    = fread(fid, 1, 'float');
            p.angvel(j,i) = fread(fid, 1, 'float');
            p.force(j,i)  = fread(fid, 1, 'float');
            p.torque(j,i) = fread(fid, 1, 'float');
        end
    end
    
    for j=1:p.np %Parameters with one value per particle
        p.fixvel(j)  = fread(fid, 1, 'float');
	p.xsum(j)    = fread(fid, 1, 'float');
        p.radius(j)  = fread(fid, 1, 'float');
        p.rho(j)     = fread(fid, 1, 'float');
        p.k_n(j)     = fread(fid, 1, 'float');
        p.k_s(j)     = fread(fid, 1, 'float');
	p.k_r(j)     = fread(fid, 1, 'float');
	p.gamma_s(j) = fread(fid, 1, 'float');
	p.gamma_r(j) = fread(fid, 1, 'float');
        p.mu_s(j)    = fread(fid, 1, 'float');
        p.mu_r(j)    = fread(fid, 1, 'float');
        p.C(j)       = fread(fid, 1, 'float');
        p.E(j)       = fread(fid, 1, 'float');
        p.K(j)       = fread(fid, 1, 'float');
        p.nu(j)      = fread(fid, 1, 'float');
        p.es_dot(j)  = fread(fid, 1, 'float');
        p.es(j)      = fread(fid, 1, 'float');
        p.p(j)       = fread(fid, 1, 'float');
    end
    
    %Physical, constant parameters
    %params.global = fread(fid, 1, 'ubit1');
    params.global     = fread(fid, 1, 'int');
    params.g          = fread(fid, [1, grids.nd], 'float');
    params.kappa      = fread(fid, 1, 'float');
    params.db  	      = fread(fid, 1, 'float');
    params.V_b	      = fread(fid, 1, 'float');
    params.shearmodel = fread(fid, 1, 'uint32');
    
    % Wall data
    walls.nw = fread(fid, 1, 'uint32');
    for j=1:walls.nw
        for i=1:grids.nd
            walls.n(j,i)     = fread(fid, 1, 'float');
        end
        walls.x(j)     = fread(fid, 1, 'float');
        walls.m(j)     = fread(fid, 1, 'float');
        walls.vel(j)   = fread(fid, 1, 'float');
        walls.force(j) = fread(fid, 1, 'float');
        walls.devs(j)  = fread(fid, 1, 'float');
    end
    params.periodic = fread(fid, 1, 'int');

    for j=1:p.np
      p.bonds(j,1) = fread(fid, 1, 'uint32');
      p.bonds(j,2) = fread(fid, 1, 'uint32');
      p.bonds(j,3) = fread(fid, 1, 'uint32');
      p.bonds(j,4) = fread(fid, 1, 'uint32');
    end
    
    fclose(fid);

end
