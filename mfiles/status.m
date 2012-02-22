function lastfile = status(project)

%  status() writes status of current model run

st = load(['../output/' project '.status.dat']);

disp(['Status: time = ',num2str(st(1)),', ',num2str(st(2)),'% done, filenr = ',num2str(st(3))]);

lastfile = st(3);

end
