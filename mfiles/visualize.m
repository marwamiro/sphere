function visualize(project, method, file_nr)

  lastfile = status(project);

  if exist('file_nr','var')
    filenr = file_nr;
  else
    filenr = lastfile;
  end

  figure;

  % Plot sum of kinetic energy vs. time
  if strcmpi(method, 'energy')

    disp('Energy')

    Epot   = zeros(1,lastfile);
    Ekin   = zeros(1,lastfile);
    Erot   = zeros(1,lastfile);
    Es     = zeros(1,lastfile);
    Es_dot = zeros(1,lastfile);
    Esum   = zeros(1,lastfile);

    % Load data from all output files
    for i = 1:lastfile
      fn = [project '.output' num2str(i) '.bin'];
      disp(fn);
      [p, ~, time, params, ~] = freadbin('../output/', fn);
      for j = 1:p.np
	r = p.radius(j);
	m = (4/3*pi*r*r*r*p.rho(j));
	Epot(i) = Epot(i) + m * norm(params.g) * p.x(j,3);
	Ekin(i) = Ekin(i) + 0.5 * m ...
	  * norm(p.vel(j,:)) * norm(p.vel(j,:));
	Erot(i) = Erot(i) + 0.5 * (2/5 * m * r*r) ...
	  * norm(p.angvel(j,:)) * norm(p.angvel(j,:));
      end
      Es(i)     = sum(p.es);
      Es_dot(i) = sum(p.es_dot);

      Esum(i) = Epot(i) + Ekin(i) + Erot(i) + Es(i);
      %Esum(i) = Epot(i) + Ekin(i) + Es(i);
    end

    % Time axis
    t = linspace(time.file_dt, time.current, length(Ekin));

    %disp(num2str(Ekin(:)));

    % Visualization, E_pot
    subplot(2,3,1);
    plot(t, Epot, '+-');
    title('Potential energy');
    xlabel('Time [s]');
    ylabel('Total potential energy [J]');
    box on;
    grid on;

    % Visualization, E_kin
    subplot(2,3,2);
    plot(t, Ekin, '+-');
    title('Kinetic energy');
    xlabel('Time [s]');
    ylabel('Total kinetic energy [J]');
    box on;
    grid on;

    % Visualization, E_rot
    subplot(2,3,3);
    plot(t, Erot, '+-');
    title('Rotational energy');
    xlabel('Time [s]');
    ylabel('Total rotational energy [J]');
    box on;
    grid on;

    % Visualizaiton, E_s_dot (current shear energy)
    subplot(2,3,4);
    plot(t, Es_dot, '+-');
    title('Shear energy rate');
    xlabel('Time [s]');
    ylabel('Shear energy rate [W]');
    box on;
    grid on;

    % Visualizaiton, E_s (total shear energy)
    subplot(2,3,5);
    plot(t, Es, '+-');
    title('Total shear energy');
    xlabel('Time [s]');
    ylabel('Total shear energy [J]');
    box on;
    grid on;

    % Total energy, Esum
    subplot(2,3,6);
    plot(t, Esum, '+-');
    title('Total energy: Pot + Kin + Rot + Shear');
    xlabel('Time [s]');
    ylabel('Total energy [J]');
    box on;
    grid on;

  end % End visualize energy


  % Visualize wall kinematics and force
  if strcmpi(method, 'walls')
    disp('Walls')

    [~, ~, ~, ~, walls] = freadbin('../output/', [project '.output0.bin']);

    wforce = zeros(lastfile, walls.nw);
    wvel   = zeros(lastfile, walls.nw);
    wpos   = zeros(lastfile, walls.nw);
    wdevs  = zeros(lastfile, walls.nw);

    % Load data from all output files
    for i = 1:lastfile
      fn = [project '.output' num2str(i) '.bin'];
      disp(fn);
      [~, ~, time, ~, walls] = freadbin('../output/', fn);

      for j = 1:walls.nw
	wforce(i,j) = walls.force(j);
	wvel(i,j)   = walls.vel(j);
	wpos(i,j)   = walls.x(j);
	wdevs(i,j)  = walls.devs(j);
      end
    end

    % Time axis
    t = linspace(time.file_dt, time.current, lastfile);

    % Visualization, wall position
    subplot(2,2,1);
    hold on
    for i=1:walls.nw
      p = plot(t, wpos(:,i), '+-');
      if i==2
	set(p,'Color','red')
      end
    end
    hold off
    title('Wall positions');
    xlabel('Time [s]');
    ylabel('Position [m]');
    box on;
    grid on;

    % Visualization, wall force
    subplot(2,2,2);
    hold on
    for i=1:walls.nw
      p = plot(t, wforce(:,i), '+-');
      if i==2
	set(p,'Color','red')
      end
    end
    hold off
    title('Wall forces');
    xlabel('Time [s]');
    ylabel('Force [N]');
    box on;
    grid on;

    % Visualization, wall velocity
    subplot(2,2,3);
    hold on
    for i=1:walls.nw
      p = plot(t, wvel(:,i), '+-');
      if i==2
	set(p,'Color','red')
      end
    end
    hold off
    title('Wall velocities');
    xlabel('Time [s]');
    ylabel('Velocity [m/s]');
    box on;
    grid on;

    % Visualization, wall deviatoric stresses
    subplot(2,2,4);
    hold on
    for i=1:walls.nw
      p = plot(t, wdevs(:,i), '+-');
      if i==2
	set(p,'Color','red')
      end
    end
    hold off
    title('Wall deviatoric stresses');
    xlabel('Time [s]');
    ylabel('Stress [Pa]');
    box on;
    grid on;

  end % End visualize walls


  % Visualize first output file with plotspheres
  if strcmpi(method, 'first')
    disp('Visualizing first output file')
    fn = [project '.output0.bin'];
    [p, grids, ~, ~] = freadbin('../output/', fn);
    plotspheres(p, grids, 5, 0, 1);
  end % End visualize last file


  % Visualize last output file with plotspheres
  if strcmpi(method, 'last')
    disp('Visualizing last output file')
    fn = [project '.output' num2str(lastfile) '.bin'];
    [p, grids, ~, ~] = freadbin('../output/', fn);
    plotspheres(p, grids, 5, 0, 1);
  end % End visualize last file


  % Visualize horizontal velocities (velocity profile)
  if strcmpi(method, 'veloprofile')

    disp('Visualizing velocity profile');

    % Read data
    fn = [project '.output' num2str(filenr) '.bin'];
    disp(fn);
    [p, ~, time, params, ~] = freadbin('../output/', fn);
    
    horiz_velo = zeros(2, p.np); % x,y velocities for each particle

    for j = 1:p.np
      horiz_velo(1, j) = p.vel(j, 1); % x-velocity
      horiz_velo(2, j) = p.vel(j, 2); % y-velocity
    end
    
    % Find shear velocity (for scaling first axis)
    fixidx = find(p.fixvel > 0.0);
    shearvel = max(p.vel(fixidx,1));

    % Plot x- and y velocities vs. z position
    hold on;
    plot(horiz_velo(2,:), p.x(:,3), '.b'); % y-velocity (blue)
    plot(horiz_velo(1,:), p.x(:,3), '.r'); % x-velocity (red)
    axis([-0.5*shearvel, shearvel*1.5, min(p.x(:,3)), max(p.x(:,3))]);
    title(['Velocity profile, t = ' num2str(time.current) ' s']);
    xlabel('Horizontal velocity [m/s]');
    ylabel('Vertical position [m]');
    legend('y', 'x', 'Location', 'SouthEast');
    box on;
    grid on;
    hold off;

  end % End visualize velocity profile

  if strcmpi(method, 'sheardisp')
    disp('Visualizing shear displacement, 1D');

    % Read first datafile
    [p, ~, ~, ~, ~] = freadbin('../output/',[project '.output0.bin']);

    % Read original z-position at t = 0 s.
    zpos = p.x(:,3);

    % Read last datafile
    [p, ~, ~, ~, ~] = freadbin('../output/',[project '.output' num2str(lastfile) '.bin']);

    % Plot
    plot(p.xsum(:), zpos(:), 'o');
    title(project);
    xlabel('Shear displacement [m]');
    ylabel('Initial vertical position [m]');
    box on;
    grid on;

  end % End visualize shear displacement

  if strcmpi(method, 'sheardisp2d')
    disp('Visualizing shear displacement, 2D');

    % Read first datafile
    [p, ~, ~, ~, ~] = freadbin('../output/',[project '.output0.bin']);

    % Read original z-position at t = 0 s.
    zpos = p.x(:,3);

    % Read last datafile
    [p, ~, ~, ~, ~] = freadbin('../output/',[project '.output' num2str(lastfile) '.bin']);



    % Plot
    colormap(jet)
    title(project);
    xlabel('Shear displacement [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    box on;
    grid on;
    axis equal;
    axis vis3d;
    rotate3d on;

  end % End visualize shear displacement 2D

  if strcmpi(method, 'sheardisp3d')
    disp('Visualizing shear displacement, 3D');

    % Read last datafile
    [p, ~, ~, ~, ~] = freadbin('../output/',[project '.output' num2str(lastfile) '.bin']);

    % Plot
    scatter3(p.xsum(:), p.x(:,2), p.x(:,3), 30, p.x(:,3), 'filled');
    colormap(jet)
    title(project);
    xlabel('Shear displacement [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    box on;
    grid on;
    axis equal;
    axis vis3d;
    rotate3d on;

  end % End visualize shear displacement 3D

  % Visualize shear stresses
  if strcmpi(method, 'shear')

    [p, grids, time, params, walls] = freadbin('../output/',[project '.output0.bin']);

    disp('Visualizing shear stress dynamics')
    xdisp    = zeros(1, lastfile+1); % Shear displacement
    sigma    = zeros(1, lastfile+1); % Effective normal stress
    tau      = zeros(1, lastfile+1); % Shear stress
    dilation = zeros(1, lastfile+1); % Dilation

    % Calculate the shear velocity
    fixedidx_all = find(p.fixvel(:) > 0.0); % All particles with a fixed horiz. vel.
    shearvel = max(p.vel(fixedidx_all,1)); 

    % Surface area
    A = grids.L(1) * grids.L(2); % x-length * y-length

    % Load data from all output files
    for i = 0:lastfile
      fn = [project '.output' num2str(i) '.bin'];
      disp(fn);
      [p, ~, time, ~, walls] = freadbin('../output/', fn);

      xdisp(i+1)    = time.current * shearvel;
      sigma(i+1)    = walls.force(1) / A;
      tau(i+1)      = shearstress(p, A);
      dilation(i+1) = walls.x;
    end

    % Plot stresses
    subplot(2,1,1);
    plot(xdisp, sigma, 'o-g',  xdisp, tau, '+-r');
    title('Stress dynamics');
    xlabel('Shear distance [m]');
    ylabel('Stress [Pa]');
    legend('\sigma`','\tau');
    box on;
    grid on;

    % Plot dilation
    subplot(2,1,2);
    plot(xdisp, dilation, '+-b');
    title('Dilation');
    xlabel('Shear distance [m]');
    ylabel('Upper wall pos. [m]');
    box on;
    grid on;
    
  end

end % End function
