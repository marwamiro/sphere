#!/bin/sh
rm /home/adc/code/sphere-cfd/output/*.{vtu,vti}; cd ~/code/sphere-cfd && cmake .  && make -j2 && cd ~/code/sphere-cfd/python && ipython ns_cons_of_mass.py
