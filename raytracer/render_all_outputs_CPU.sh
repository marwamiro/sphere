#!/bin/bash

FILES=`ls ../output/*.bin`

XRES=800
YRES=800

WORKHORSE=CPU

echo "# Rendering PPM images and converting to JPG"
for F in ../output/*.bin
do
  (BASE=`basename $F`;
  ./rt $WORKHORSE $F $XRES $YRES ../img_out/$BASE.ppm;
  convert ../img_out/$BASE.ppm ../img_out/$BASE.jpg;)
  #rm ../img_out/$BASE.ppm &)
done

#echo "# Converting PPM files to JPG using ImageMagick in parallel"
#for F in ../img_out/*.ppm
#do
#  (BASE=`basename $F`; convert $F $F.jpg &)
#done

#echo "# Removed temporary files"
#rm ../img_out/*.ppm

