# Pockmark_Detection

![yolo](/images/pass_dem_yolo.png)

![pix2pix](/images/pass_dem_gan.png)

This repository will hold trained pockmark detectors, including a YOLOv5 CNN and a pix2pix GAN.

It will also hold an annotation set that can be added to for further enhancement of the detectors.

Code to run preprocessing tools, the detectors, and post-processing tools is included in pockmark_detection_main.py. 

See [yolov5 repo](https://github.com/ultralytics/yolov5) and [pix2pix repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for those two models' original source code.

# Requirements

Hardware: Windows machine with NVIDIA graphics card

Software: Windows10, Anaconda, Google Earth and/or QGIS to visualize results

Python Libraries: All requirements are listed in three environment files (pockmark_detection.yml, yolo_pockmark.yml, and pix2pix_pockmark.yml).

Use conda to create these three environments.
	
	cd pockmark_detection/envs

	conda env create --file pockmark_detection.yml

	conda env create --file yolo_pockmark.yml

	conda env create --file pix2pix_pockmark.yml

# Running the models

Download/clone this repo. 

Also download the trained weights under Releases.

Yolo weights (pockmark_yolov5s.pt) should be placed in Pockmark_Detection/trained_models/yolo.

pix2pix weights (latest_net_D.pth and latest_net_G.pth) should be placed in Pockmark_Detection/pix2pix_modules/checkpoints/pockmark_gan.

Open up Anaconda Prompt and activate the 'pockmark_detection' environment.
	
	cd Pockmark_Detection

	conda activate pockmark_detection

Open pockmark_detection_main.py in IDLE or another Python IDE of your choice.

The only function to run is 'main'. You feed it the file path to the bathy grid (as a geotiff) and the EPSG code for that grid's coordinate system.

	main(r'.../bathygrid.tif', 1234)

This will do the following:

1. Tile the grid into square geotiffs of size 256 x 256 in pixels; convert the geotiff tiles into jpegs.
2. Run the yolo and pix2pix models.
3. Convert the yolo and pix2pix outputs into shapefiles.
4. Convert the bathy grid and yolo/pix2pix shapefiles into kmls.

# Troubleshooting

Make sure all requirements are downloaded to the correct environments. Sometimes libaries need to be downloaded manually.

Make sure the input grid is a single band geotiff containing bathymetry values. Also make sure the grid contains pockmarks.

Make sure you are using a Windows10 machine with NVIDIA graphics card. This code will not work with a Mac.

# Examining/Improving Results

Results should be examined/refined with GIS software.

Running on new data might trick up the current models. Training them on new data will help them adapt.

Bumping the threshold up (default is 0.24) for the yolo model will help limit false positives.

Trying finer resolutions (up to 1m) for the input DEM can improve results. Including overlap in the tiles can help with edge effects.

# Retraining

To re-train a yolov5 model, see github/mlundine/tensorflow_app or [yolov5 repo](https://github.com/ultralytics/yolov5).

To re-train a pix2pix model, you need bathy grids and masks (255 = pockmark, 0 = other), in jpeg or png format.
These should be randomly split into a train (60%), test (20%), and validation (20%) set.
Then they should be arranged into directories of this format:
A/train
A/test
A/val
B/train
B/test
B/val

To construct the training images for a pix2pix model, use setup_datasets(). 
	setup_datasets(home,
                       name,
                       foldA,
                       foldB):
    		"""
    		Setups annotation pairs for pix2pix training
    		inputs:
    		home: parent directory for annotations (str) (ex: r'pix2pix_modules/datasets/MyProject')
    		name: project name (str)
    		foldA: path to A annotations (str)
    		foldB: path to B annotations (str)
		"""


Once the dataset is set up, training can commence. Use the function train_model(). 
This function will save new weigthts to the pix2pix_modules/checkpoints directory.

	train_model(model_name,
                    model_type,
                    dataroot,
                    n_epochs = 100):
   	 	"""
    		Trains pix2pix or cycle-GAN model
   		inputs:
    		model_name: name for your model (str)
    		model_type: either 'pix2pix' or 'cycle-GAN' (str)
    		dataroot: path to training/test/val directories (str)
    		n_epochs (optional): number of epochs to train for (int)
		"""
