#!/usr/bin/env gnuplot
# Call this script with sid and gamma variables, e.g.
#  $ gnuplot -e "sid='testrun'; gamma='4.3'; xmin='0.0'; xmax='1.0'; ymin='0.0'; ymax='1.0'" plotts.gp

set title sid.", $\\gamma$ = ".gamma

set term pngcairo size 50 cm,40 cm
set out "../../img_out/".sid."-ts-x1x3.png"

set palette defined (0 "blue", 0.5 "gray", 1 "red")

set xlabel "$\\x^1$"
set ylabel "$\\x^3$"
set cblabel "Pressure [Pa]"

set xrange [xmin:xmax]
set yrange [ymin:ymax]

set size ratio -1

plot "../data/".sid."-ts-x1x3.txt" with circles palette fs transparent solid 0.4 noborder t "Particle",\
         "../data/".sid."-ts-x1x3-circ.txt" with circles lt 1 lc rgb "black" notitle,\
         "../data/".sid."-ts-x1x3-arrows.txt" using 1:2:3:4 with vectors head filled lt 1 lc rgb "black" t "Rotation",\
         "../data/".sid."-ts-x1x3-velarrows.txt" using 1:2:3:4 with vectors head filled lt 1 lc rgb "white" t "Translation",\
         "../data/".sid."-ts-x1x3-slips.txt" using 1:2:3:4 with vectors head filled lt 1 lc rgb "green" t "Slip"

