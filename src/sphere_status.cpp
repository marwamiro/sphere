#include <iostream>
#include <cstdio>

int main(int argc, char *argv[])
{
    using std::cout;

    // Read path to current working directory
    char *cwd;
    cwd = getcwd (0, 0);
    if (! cwd) {  // Terminate program execution if path is not obtained
        cout << "getcwd failed\n";
        return 1; // Return unsuccessful exit status
    }

    // Simulation name/ID read from first input argument
    if (argc != 2) {
        cout << "You need to specify the simulation ID as an input parameter,\n"
            << "e.g. " << argv[0] << " 3dtest\n";
        return 1;
    }

    char *sim_name = argv[1];

    // Open the simulation status file
    FILE *fp;
    char file[1000]; // Complete file path+name variable
    sprintf(file,"%s/output/%s.status.dat", cwd, sim_name);

    if ((fp = fopen(file, "rt"))) {
        float time_current;
        float time_percentage;
        unsigned int file_nr;
        fscanf(fp, "%f%f%d", &time_current, &time_percentage, &file_nr);

        cout << "Reading " << file << ":\n"
            << " - Current simulation time:  " << time_current << " s\n"
            << " - Percentage completed:     " << time_percentage << "%\n"
            << " - Latest output file:       " 
            << sim_name << ".output" << file_nr << ".bin\n";

        fclose(fp);

        return 0; // Exit program successfully

    }
}
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
