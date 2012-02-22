#!/bin/bash

XRES=800
YRES=600

WORKHORSE=GPU

echo "# Rendering PPM images"
for F in ../output/*.bin
do
  BASE=`basename $F`
  OUTFILE=$BASE.ppm.jpg
  if [ -e ../img_out/$OUTFILE ]
  then
    echo $OUTFILE exists.
  else
    ./rt $WORKHORSE $F $XRES $YRES ../img_out/$BASE.ppm pressure 4000 > /dev/null
    if [ $? -ne 0 ] ; then
      echo "Error rendering $F, trying again..."
      ./rt $WORKHORSE $F $XRES $YRES ../img_out/$BASE.ppm pressure 4000 > /dev/null
    fi
  fi
done

echo "# Converting PPM files to JPEG using ImageMagick in parallel"
for F in ../img_out/*.ppm
do
  (BASE=`basename $F`; convert $F $F.jpg &)
done

sleep 5
echo "# Removing temporary PPM files"
rm ../img_out/*.ppm

