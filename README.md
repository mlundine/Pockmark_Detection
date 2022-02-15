# Pockmark_Detection

This repository will hold trained pockmark detectors, including a YOLOv5 CNN and a pix2pix GAN.

It will also hold an annotation set that can be added to for further enhancement of the detectors.

Code to run preprocessing tools, the detectors, and post-processing tools is included in pockmark_detection_main.py. 

# Requirements

Hardware: Windows machine with NVIDIA graphics card

Software: Windows10, Anaconda

Python Libraries: All requirements are listed in three environment files (pockmark_detection.yml, yolo_pockmark.yml, and pix2pix_pockmark.yml).

Use conda to create these three environemnts.

conda env create --file pockmark_detection.yml

conda env create --file yolo_pockmark.yml

conda env create --file pix2pix_pockmark.yml

# Running the models

Open up Anaconda Prompt and activate the 'pockmark_detection' environment (conda activate pockmark_detection)

Open pockmark_detection_main.py in IDLE or another Python IDE of your choice.

The only function to run is 'main'. You feed it the file path to the bathy grid (as a geotiff) and the EPSG code for that grid's coordinate system.

Ex: main(r'.../bathygrid.tif', 1234)

This will do the following:

1. Tile the grid into square geotiffs of size 256 x 256 in pixels; convert the geotiff tiles into jpegs.
2. Run the yolo and pix2pix models.
3. Convert the yolo and pix2pix outputs into shapefiles.
4. Convert the bathy grid and yolo/pix2pix shapefiles into kmls.

Timed elapsed for entire script (tiling, conversion, model runs, gis file construction) on Passamaquoddy Bay: 
