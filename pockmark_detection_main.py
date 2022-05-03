import os
import warnings
import glob
from osgeo import gdal
from gdal_modules import gdal_retile
from gdal_modules import gdal_functions_app as gda
import geopandas as gpd
import timeit



warnings.filterwarnings('ignore')
root = os.getcwd()

def project_grid(path_to_input,
                 path_to_output,
                 output_epsg):
    """
    Projects raster into different coordinate system
    inputs:
    path_to_input: filepath to input raster
    path_to_output: filepath to save the output raster
    output_epsg: EPSG code (####) for output raster
    outputs:
    path_to_output: filepath to the output raster    
    """
    destination_epsg = 'EPSG:'+str(output_epsg)
    warp = gdal.Warp(path_to_output,path_to_input,dstSRS=destination_epsg)
    warp = None # Closes the files
    return path_to_output

def resize_grid(path_to_input,
                path_to_output,
                xres=1,
                yres=1):
    """
    Resizes input raster
    inputs:
    path_to_input: filepath to input raster
    path_to_output: filepath to save the output raster
    xres: x cell size (length scale is determined by the input coordinate system, typically meters)
    yres: y cell size (length scale is determined by the input coordinate system, typically meters)
    outputs:
    path_to_output: filepath to the output raster
    """
    resample_alg = 'near'
    opt = gdal.WarpOptions(xRes=xres, yRes=yres, resampleAlg=resample_alg)
    ds = gdal.Warp(path_to_output, path_to_input, options=opt)
    ds = None
    return path_to_output

def tile_grid(path_to_input,
              path_to_output,
              tile_size=256,
              overlap=25):
    """
    Takes a large bathymetry grid and slices it into square tiles
    Inputs:
    path_to_input: file path to a geotiff bathy grid
    path_to_output: directory path to save geotiff tiles
    tile_size: height/width of tiles in pixels
    overlap: overlap of tiles in pixels
    """
    tiler = gdal_retile
    tiler.Verbose = False
    tiler.TileHeight = tile_size
    tiler.TileWidth = tile_size
    tiler.Overlap = overlap
    tiler.Names = [path_to_input]
    tiler.TargetDir = path_to_output + r'/'
    tiler.TileIndexName = os.path.splitext(os.path.basename(path_to_input))[0] + '_tiles_shapefile'
    tiler.main()
    gda.delete_empty_images(path_to_output)
    print('Geotiff tiles and shapefile saved to ' + path_to_output)
    
def get_tile_metadata(input_folder):
    saveFile = os.path.join(input_folder, 'metadata.csv')
    num_images, metadata_csv = gda.gdal_get_coords_and_res(input_folder, saveFile)
    print('Metadata saved to ' + saveFile)
    return num_images, metadata_csv
    
def convert_tif_to_jpeg(input_tiff_folder,
                        output_jpeg_folder,
                        size=256):
    """
    Converts geotiffs to jpegs
    inputs:
    tiff_folder: directory path to geotiff tiles
    jpeg_folder: directory path to save jpegs
    """
    gda.gdal_convert(input_tiff_folder,
                     output_jpeg_folder,
                     '.tif',
                     '.jpeg',
                     size=256)
    print('jpegs saved to ' + output_jpeg_folder)
    
def run_yolo(source,
             weights,
             threshold):
    """
    Runs yolo model
    inputs:
    source: path to folder of jpegs
    weights: path to yolo weights file
    threshold: confidence threshold to run detector at
    outputs:
    yolo_results: folder where yolo labels are saved to
    """
    output_folder = os.path.join(root, 'yolo_modules', 'results')
    yolo_detect = os.path.join(root, 'yolo_modules', 'Lundine_yolo_detect.py')
    cmd0 = 'conda deactivate & conda activate yolo_pockmark & '
    cmd1 = 'python ' + yolo_detect + ' --weights '
    cmd2 = weights + ' --source ' + source
    cmd3 = ' --conf ' + str(threshold) + ' --project ' + output_folder
    cmd4 = ' --save-txt --save-conf --img-size 256'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4
    os.system(full_cmd)
    yolo_results = os.path.join(output_folder, 'detect', 'labels')
    return yolo_results
    
    
def run_pix2pix(source,
                num_images):
    """
    Runs pix2pix model
    inputs:
    source: path to folder of jpegs
    num_images: number of images in that folder
    outputs:
    save_folder: folder where the fake images are saved
    """
    pix2pix_detect = os.path.join(root, 'pix2pix_modules', 'test.py')
    cmd0 = 'conda deactivate & conda activate pix2pix_pockmark & '
    cmd1 = 'python ' + pix2pix_detect
    cmd2 = ' --dataroot ' + source
    cmd3 = ' --model test'
    cmd4 = ' --name '+ os.path.join(root, 'pix2pix_modules', 'checkpoints','pockmark_gan')
    cmd5 = ' --netG unet_256'
    cmd6 = ' --netD basic'
    cmd7 = ' --dataset_mode single'
    cmd8 = ' --norm batch'
    cmd9 = ' --num_test ' + str(num_images)
    cmd10 = ' --preprocess none'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10
    os.system(full_cmd)    
    save_folder = os.path.join(root, 'pix2pix_modules', 'checkpoints', 'pockmark_gan', 'test_latest', 'images')
    return save_folder

def google_earth_outputs(dem, yolo_shape, gan_shape, tile_folder):
    google_earth_folder = os.path.join(tile_folder, 'google_earth')
    superoverlay = os.path.join(google_earth_folder, os.path.splitext(os.path.basename(dem))[0]+'.kml')
    yolo_kml = os.path.join(google_earth_folder, 'yolo_detections.kml')
    gan_kml = os.path.join(google_earth_folder, 'gan_detections.kml')
    try:
        os.mkdir(google_earth_folder)
    except:
        pass
    #convert dem to superoverlay
    cmd0 = 'gdal_translate -scale -of KMLSUPEROVERLAY ' + dem + ' ' + superoverlay
    #os.system(cmd0)
    #convert gan shapefile to kml
    cmd1 = 'ogr2ogr -f ' + '"'+'KML' + '" ' + yolo_kml + ' ' + yolo_shape
    os.system(cmd1)
    #convert yolo shapefile to kml
    cmd2 = 'ogr2ogr -f ' + '"'+'KML' + '" ' + gan_kml + ' ' + gan_shape
    os.system(cmd2)


def process_detection_results(yolo_results,
                              gan_results,
                              metadata_csv,
                              geotiff_tiles,
                              epsg,
                              shapefile_folder):
    """
    Outputs GIS files for pockmark detection results
    Inputs:
    yolo_results: path to yolo results (ex .../detect/labels)
    gan_results: path to gan results (ex .../test_latest/images)
    metadata_csv: path to the geotiff tile metatdata
    geotiff_tiles: path to the geotiff tiles
    epsg: epsg code
    """
    #yolo tranlate bboxes
    yolo_geobbox_csv = os.path.join(os.path.dirname(yolo_results),'yolo_detections_geo.csv')
    gda.yolotranslate_bboxes(yolo_results,
                             yolo_geobbox_csv,
                             metadata_csv)
    print('Yolo detections converted to geographic coordinates')
    #yolo csv to shapefile
    yolo_shape = os.path.join(shapefile_folder, 'yolo_detections.shp')
    gda.write_polygon_shp(yolo_geobbox_csv,
                          yolo_shape,
                          epsg)
    print('Yolo detections written to shapefile')

    #get gan fake images
    gan_images = []
    for im in glob.glob(gan_results + '/*.png'):
        if im.find('fake')>0:
            gan_images.append(im)
        else:
            continue
        
    #make a folder to put geotiff gan outputs in
    gan_geotiff_outputs = os.path.join(os.path.dirname(gan_results), 'geotiffs')
    try:
        os.mkdir(gan_geotiff_outputs)
    except:
        pass
    
    #gan png to geotiff
    gda.batchPNGtoGeotiff(gan_images,
                          metadata_csv,
                          gan_geotiff_outputs,
                          geotiff_tiles,
                          epsg)
    print('GAN pngs converted to geotiffs')
    
    #gan geotiff to shape
    gda.raster_to_polygon_batch(gan_geotiff_outputs)
    print('GAN geotiffs converted to shapefiles')
    
    
    #gan merge shapes
    gan_shape = os.path.join(shapefile_folder, 'pix2pix_pockmarks.shp')
    gda.mergeShapes(gan_geotiff_outputs, gan_shape)
    print('GAN shapefiles merged')
    
    #filter gan
    filtered_gan_shape = os.path.join(shapefile_folder, 'pix2pix_pockmarks_filtered.shp')
    map = gpd.read_file(yolo_shape)
    gan_df = gpd.read_file(gan_shape)
    intersect_polygons = gpd.sjoin(gan_df, map, op = 'intersects')
    intersect_polygons['area'] = intersect_polygons.geometry.area
    subset = intersect_polygons[intersect_polygons['area']>20]
    geoms = subset.geometry.unary_union
    dissolved = gpd.GeoDataFrame(geometry=[geoms])
    dissolved.crs = subset.crs
    dissolved = dissolved.explode().reset_index(drop=True)
    dissolved.to_file(filtered_gan_shape)
    print('Filtered GAN shapefile written')

    return yolo_shape, filtered_gan_shape

def setup_datasets(home,
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
    root = os.getcwd()
    combine = os.path.join(root, 'pix2pix_modules', 'datasets','combine_A_and_B.py')
    cmd0 = 'conda deactivate & conda activate pix2pix_pockmark & '
    cmd1 = 'python ' + combine 
    cmd2 = ' --fold_A ' + foldA
    cmd3 = ' --fold_B ' + foldB
    cmd4 = ' --fold_AB ' + home
    cmd5 = ' --no_multiprocessing'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5
    os.system(full_cmd)
    
def train_model(model_name,
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
    root = os.getcwd()
    pix2pix_train = os.path.join(root, 'pix2pix_modules', 'train.py')

    cmd0 = 'conda deactivate & conda activate pix2pix_pockmark & '
    cmd1 = 'python ' + pix2pix_train
    cmd2 = ' --dataroot ' + dataroot
    cmd3 = ' --model ' + model_type
    cmd4 = ' --name ' + model_name#change this as input
    cmd5 = ' --netG unet_256'
    cmd6 = ' --netD basic'
    cmd7 = ' --preprocess none'
    cmd8 = ' --checkpoints_dir ' + os.path.join(root, 'pix2pix_modules', 'checkpoints')
    cmd9 = ' --n_epochs ' + str(n_epochs)
    cmd10 = ' --input_nc 1 --output_nc 1'
    cmd11 = ' --display_id -1'
    cmd12 = ' --gpu_ids 0'
    full_cmd = cmd0+cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd12
    os.system(full_cmd)
    print('Training Finished')
    
    
def main(input_dem,
         epsg,
         threshold=0.24):
    """
    Will tile up your input bathy grid and convert it to jpegs.
    Runs the yolov5 and pix2pix models on those jpegs.
    Converts results into shapefiles in the coordinate system specified by the EPSG code.
    
    Inputs:
    input_dem: a filepath to a geotiff bathy grid (str)
    epsg: epsg code (int)
    threshold: threshold (0 to 0.99) for yolo (float)
    """


    
    ##Define Some Folders, make them if they don't exist yet
    tile_folder = os.path.join(root, os.path.splitext(os.path.basename(input_dem))[0]+'_bathy_tiles')
    output_tile_folder_geotiff = os.path.join(tile_folder, 'geotiffs')
    output_tile_folder_jpeg = os.path.join(tile_folder, 'jpegs')
    output_shapefile_folder = os.path.join(tile_folder, 'shapefiles')
    try:
        os.mkdir(tile_folder)
        os.mkdir(output_tile_folder_geotiff)
        os.mkdir(output_tile_folder_jpeg)
        os.mkdir(output_shapefile_folder)
    except:
        pass

    ##Step 1: Tile the input, save metadata, convert to jpegs
    tile_grid(input_dem,output_tile_folder_geotiff)
    num_images, metadata_csv = get_tile_metadata(output_tile_folder_geotiff)
    convert_tif_to_jpeg(output_tile_folder_geotiff, output_tile_folder_jpeg, size=256)

    ##Step 2: Run models
    yolo_weights = os.path.join(root, 'trained_models', 'yolo', 'pockmark_yolov5s.pt')
    yolo_results = run_yolo(output_tile_folder_jpeg, yolo_weights, threshold)
    print('Yolo results saved')
    gan_results = run_pix2pix(output_tile_folder_jpeg, num_images)
    print('pix2pix results saved')
    
    ##Step 3: Process results
    yolo_shape, filtered_gan_shape = process_detection_results(yolo_results,
                                                               gan_results,
                                                               metadata_csv,
                                                               output_tile_folder_geotiff,
                                                               epsg,
                                                               output_shapefile_folder)
    ##Step 4: Google Earth Outputs
    google_earth_outputs(input_dem, yolo_shape, filtered_gan_shape, tile_folder)
    print('Google Earth files written')



