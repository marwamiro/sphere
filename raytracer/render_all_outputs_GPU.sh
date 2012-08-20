#!/bin/bash

FILES=`ls ../output/*.bin`

XRES=800
YRES=800

WORKHORSE=GPU

echo "# Rendering PPM images"
for F in ../output/*.bin
do
  BASE=`basename $F`
  ./rt $WORKHORSE $F $XRES $YRES ../img_out/$BASE.ppm > /dev/null
  if [ $? -ne 0 ] ; then
    echo "Error rendering $F, trying again..."
    ./rt $WORKHORSE $F $XRES $YRES ../img_out/$BASE.ppm > /dev/null
  fi
done

echo "# Converting PPM files to JPEG using ImageMagick in parallel"
for F in ../img_out/*.ppm
do
  (BASE=`basename $F`; convert $F $F.jpg &)
done

#echo "# Removed temporary files"
#rm ../img_out/*.ppm

