function fwritebin(path, fn, p, grids, time, params, walls)
    %path = '../input/';  %Target directory
    %fn   = '3dtest.bin'; %Target binary
    
    %Open binary file for writing, little endian ordering, unicode char encoding.
    %fid = fopen([path fn], 'wb','ieee-le','UTF-8'); 
    fid = fopen([path fn], 'wb', 'ieee-le'); 

    %fwrite(fid, grids.nd, 'ushort'); %Unsigned short int
    %fwrite(fid, p.np, 'ulong'); %Unsigned long int
    fwrite(fid, grids.nd, 'int'); 
    fwrite(fid, p.np, 'uint32'); 
    
    %Time parameters
    fwrite(fid, time.dt, 'float');
    fwrite(fid, time.current, 'double');
    fwrite(fid, time.total, 'double');
    fwrite(fid, time.file_dt, 'float');
    fwrite(fid, time.step_count, 'uint32');  

    for i=1:grids.nd %Grid origo
        fwrite(fid, grids.origo(i), 'float');
    end

    for i=1:grids.nd %Grid dimensions
        fwrite(fid, grids.L(i), 'float');
    end
    
    for i=1:grids.nd %Grid dimensions
        fwrite(fid, grids.num(i), 'uint32');
    end

    for j=1:p.np
        for i=1:grids.nd %Coordinates, velocity, acceleration, pre-velocity
            fwrite(fid, p.x(j,i), 'float');
            fwrite(fid, p.vel(j,i), 'float');
            fwrite(fid, p.angvel(j,i), 'float');
            fwrite(fid, p.force(j,i), 'float');
            fwrite(fid, p.torque(j,i), 'float');
        end
    end

    for j=1:p.np %Parameters with one value per particle
        fwrite(fid, p.fixvel(j), 'float');
	fwrite(fid, p.xsum(j), 'float');
        fwrite(fid, p.radius(j), 'float');
        fwrite(fid, p.rho(j), 'float');
        fwrite(fid, p.k_n(j), 'float');
        fwrite(fid, p.k_s(j), 'float');
        fwrite(fid, p.k_r(j), 'float');
        fwrite(fid, p.gamma_s(j), 'float');
	fwrite(fid, p.gamma_r(j), 'float');
        fwrite(fid, p.mu_s(j), 'float');
        fwrite(fid, p.mu_d(j), 'float');
        fwrite(fid, p.mu_r(j), 'float');
        fwrite(fid, p.C(j), 'float');
        fwrite(fid, p.E(j), 'float');
        fwrite(fid, p.K(j), 'float');
        fwrite(fid, p.nu(j), 'float');
        fwrite(fid, p.es_dot(j), 'float');
        fwrite(fid, p.es(j), 'float');
        fwrite(fid, p.p(j), 'float');
    end
    
    %Physical, constant parameters
    %fwrite(fid, params.global, 'ubit1');
    fwrite(fid, params.global, 'int');
    for i=1:grids.nd
      fwrite(fid, params.g(i), 'float');
    end
    fwrite(fid, params.kappa, 'float');
    fwrite(fid, params.db, 'float');
    fwrite(fid, params.V_b, 'float');
    fwrite(fid, params.shearmodel, 'uint32');
    
    fwrite(fid, walls.nw, 'uint32');
    for j=1:walls.nw
        for i=1:grids.nd
            fwrite(fid, walls.n(j,i), 'float'); % Wall normal
        end
        fwrite(fid, walls.x(j), 'float'); % Wall pos. on axis parallel to wall normal
        fwrite(fid, walls.m(j), 'float'); % Wall mass
        fwrite(fid, walls.vel(j), 'float'); % Wall vel. on axis parallel to wall normal
        fwrite(fid, walls.force(j), 'float'); % Wall force on axis parallel to wall normal
        fwrite(fid, walls.devs(j), 'float'); % Deviatoric stress on wall normal
    end
    fwrite(fid, params.periodic, 'int');

    for j=1:p.np
      fwrite(fid, p.bonds(i,1), 'uint32');
      fwrite(fid, p.bonds(i,2), 'uint32');
      fwrite(fid, p.bonds(i,3), 'uint32');
      fwrite(fid, p.bonds(i,4), 'uint32');
    end

    fclose(fid);

end
