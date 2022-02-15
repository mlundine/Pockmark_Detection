# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import os
from osgeo import gdal, ogr, osr
import osgeo.gdalnumeric as gdn
import numpy as np
import glob
import pandas as pd
import urllib
import shapefile
import cv2
def gdal_open(image_path):
    ### read in image to classify with gdal
    driverTiff = gdal.GetDriverByName('GTiff')
    input_raster = gdal.Open(image_path)
    nbands = input_raster.RasterCount
    prj = input_raster.GetProjection()
    gt = input_raster.GetGeoTransform()
    ### create an empty array, each column of the empty array will hold one band of data from the image
    ### loop through each band in the image nad add to the data array
    data = np.empty((input_raster.RasterYSize, input_raster.RasterXSize, nbands))
    for i in range(1, nbands+1):
        band = input_raster.GetRasterBand(i).ReadAsArray()
        data[:, :, i-1] = band
    input_raster = None
    return data, prj, gt
def yolotranslate_bboxes(inFolder,
                         saveFile,
                         coords_file):
    geobox = []
    geobox.append(['file', 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'])
    for chunk in pd.read_csv(coords_file, chunksize=1):
        res = chunk.iloc[0,5]
        width = (chunk.iloc[0,3]-chunk.iloc[0,1])/res
        height = (chunk.iloc[0,4]-chunk.iloc[0,2])/chunk.iloc[0,6]
        for i in range(len(chunk)):
            filename = os.path.splitext(os.path.basename(chunk.iloc[i,0]))[0]
            bboxFile = os.path.join(inFolder, filename+'.txt')
            try:
                bboxFile = pd.read_csv(bboxFile, sep=" ", header=None)
            except:
                continue
            bboxFile.columns = ["class", "x_center", "y_center", 'width', 'height', "conf"]
            for j in range(len(bboxFile)):
                xmin = chunk.iloc[i,1]+res*(bboxFile.iloc[j,1]-bboxFile.iloc[j,3]/2)*width
                ymin = chunk.iloc[i,4]-res*(bboxFile.iloc[j,2]-bboxFile.iloc[j,4]/2)*height
                xmax = chunk.iloc[i,1]+res*(bboxFile.iloc[j,1]+bboxFile.iloc[j,3]/2)*width
                ymax = chunk.iloc[i,4]-res*(bboxFile.iloc[j,2]+bboxFile.iloc[j,4]/2)*height
                score = bboxFile.iloc[j,5]
                label = bboxFile.iloc[j,0]
                geobox.append([filename, xmin, ymin, xmax, ymax, score, label])
    np.savetxt(saveFile, geobox, delimiter=",", fmt='%s')
    
def gdal_get_coords_and_res(folder, saveFile):
    """
    Takes a folder of geotiffs and outputs a csv with bounding box coordinates and x and y resolution
    inputs:
    folder (string): filepath to folder of geotiffs
    saveFile (string): filepath to csv to save to
    """
    
    myList = []
    myList.append(['file', 'xmin', 'ymin', 'xmax', 'ymax', 'xres', 'yres'])
    for dem in glob.glob(folder + '/*.tif'):
        src = gdal.Open(dem)
        xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
        xmax = xmin + (src.RasterXSize * xres)
        ymin = ymax + (src.RasterYSize * yres)
        myList.append([dem, xmin, ymin, xmax, ymax, xres, -yres])
        src = None
    np.savetxt(saveFile, myList, delimiter=",", fmt='%s')
    df = pd.read_csv(saveFile)
    num_images = len(df)
    return num_images, saveFile


def gdal_convert(inFolder, outFolder, inType, outType, size=256):
    """
    Converts geotiffs and erdas imagine images to .tif,.jpg, .png, or .img
    inputs:
    inFolder (string): folder of .tif or .img images
    outFolder (string): folder to save result to
    inType (string): extension of input images ('.tif' or '.img')
    outType (string): extension of output images ('.tif', '.img', '.jpg', '.png')
    """
    
    for im in glob.glob(inFolder + '/*'+inType):
        imName = os.path.splitext(os.path.basename(im))[0]
        outIm = os.path.join(outFolder, imName+outType)
        if outType == '.npy':
            raster = gdal.Open(im)
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount+1)]
            arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
            arr = np.transpose(arr, [1,2,0])
            np.save(outIm, arr)
            raster = None
        if outType == '.jpeg':
            raster = gdal.Open(im)
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount+1)]
            arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
            arr = np.transpose(arr, [1,2,0])
            if np.shape(arr)[0]!=size or np.shape(arr)[1]!=size:
                raster = None
                arr = None
                continue
            stats = [raster.GetRasterBand(i).GetStatistics(True, True) for i in range(1,raster.RasterCount+1)]
            vmin, vmax, vmean, vstd = zip(*stats)
            if raster.RasterCount > 1:
                bandList = [1,2,3]
            else:
                bandList = [1]
            gdal.Translate(outIm, raster, scaleParams = list(zip(*[vmin, vmax])), bandList = bandList)
            raster = None
            arr = None
        else:
            raster = gdal.Open(im)
            gdal.Translate(outIm, raster)
            raster = None
            
def raster_to_polygon(raster_path):
    """
    Converts raster with discrete pixel values to polygons
    inputs:
    raster_path: the input raster filepath
    """
    module = os.path.join(os.getcwd(), 'gdal_modules', 'gdal_polygonize.py')
    shape_path = os.path.splitext(raster_path)[0]+'poly.shp'
    os.system('python ' + module + ' ' + raster_path + ' ' + shape_path)

def raster_to_polygon_batch(folder):
    """
    Converts a folder of rasters to shapefiles
    inputs:
    folder: filepath to folder of geotiffs
    """
    for raster in glob.glob(folder + '/*.tif'):
        raster_to_polygon(raster)

def mergeShapes(folder, outShape):
    """
    Merges a bunch of shapefiles. Sshapefiles have to have same fields
    in attribute table.
    inputs:
    folder: filepath to folder with all of the shapefiles
    outShape: filepath to file to save to, has to have .shp extension.
    """
    module = os.path.join(os.getcwd(), 'gdal_modules', 'MergeSHPfiles_cmd.py')
    os.system('python '+ module + ' ' + folder + ' ' + outShape)


def delete_empty_images(path_to_folder):
    """
    deletes geotiffs that are all zeros, good for cleaning up tiling results
    inputs:
    path_to_folder (str): filepath to the folder with images that you want to delete
    """
    for image in glob.glob(path_to_folder + '/*.tif'):
        array = gdal_open(image)[0]
        no_data = len(np.unique(array))
        if no_data < 5:
            cmd = 'gdalmanage delete ' + image
            os.system(cmd)
        array = None
        


def write_polygon_shp(inFile, outFile, epsg):
    # create the shapefile
    driver = ogr.GetDriverByName("Esri Shapefile")
    ds = driver.CreateDataSource(outFile)
    # create the spatial reference system
    srs =  osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    
    layr1 = ds.CreateLayer('polygon',srs, ogr.wkbPolygon)
    # create the field
    layr1.CreateField(ogr.FieldDefn('score', ogr.OFTReal))
    layr1.CreateField(ogr.FieldDefn('label',ogr.OFTString))

    for chunk in pd.read_csv(inFile, chunksize=1, engine='python'):
        score = float(chunk.iloc[0,5])
        x_min = float(chunk.iloc[0,1])
        y_min = float(chunk.iloc[0,2])
        x_max = float(chunk.iloc[0,3])
        y_max = float(chunk.iloc[0,4])
        label = 'pockmark'
        
        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_min,y_max)
        ring.AddPoint(x_max,y_max)
        ring.AddPoint(x_max, y_min)
        
        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        

        # Create the features and set values
        defn = layr1.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetField('score', score)
        feat.SetField('label', label)
        feat.SetGeometry(poly)
        layr1.CreateFeature(feat)     
    # close the shapefile
    ds.Destroy()


def pngToGeotiff(png_path,
                 csv_path,
                 output_path,
                 geotiff_folder,
                 epsg_code):
    """

    inputs:
    png_path: path to the png containing image data
    kml_path: path to the kml containing spatial extent
    output_path: path to save geotiff to (ends with .tif)
    """

    ##open png as numpy array, get shape
    image = cv2.imread(png_path)
    rows,cols,bands = np.shape(image)
    coloridx = np.any(image != [0, 0, 0], axis=-1)
    blackidx = np.all(image == [0, 0, 0], axis=-1)
    image[coloridx] = [50,50,50]
    image = image[:,:,0]
    bands=1
    
    ## get the geotransform and coordinate system from the kml
    geotransform, proj = parseCoords(csv_path, png_path, rows, cols, geotiff_folder, epsg_code)

    ## save geotiff
    driverTiff = gdal.GetDriverByName('GTiff')
    out_tiff = driverTiff.Create(output_path, cols, rows, bands, gdal.GDT_Int16)
    out_tiff.SetGeoTransform(geotransform)
    out_tiff.SetProjection(proj.ExportToWkt())
    for i in range(1,bands+1):
        out_tiff.GetRasterBand(i).SetNoDataValue(0)
        out_tiff.GetRasterBand(i).WriteArray(image[:,:])

    ## clean up
    out_tiff = None
    image = None
    
def parseCoords(csv_path,
                image_name,
                nrows,
                ncols,
                geotiff_folder,
                epsg_code):
    """
    reads the xmin,xmax,ymin,and ymax from reefmaster kml
    inputs:
    kml_path: filepath to the kml
    epsg_code (optional): in case the data is not in wgs84 lat lon
    outputs:
    geotransform: gdal geotransform object
    proj: gdal projection object
    """
    im_name = os.path.splitext(os.path.basename(image_name))[0]
    g_name = im_name+'.tif'
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        df['file'][i] = os.path.splitext(os.path.basename(df['file'][i]))[0]+'_fake'+'.tif'
    df_filter = df[df['file']==g_name]

    ymax = np.array(df_filter['ymax'])[0]
    ymin = np.array(df_filter['ymin'])[0]
    xmax = np.array(df_filter['xmax'])[0]
    xmin = np.array(df_filter['xmin'])[0]
    xres = np.array(df_filter['xres'])[0]
    yres = np.array(df_filter['yres'])[0]
    
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0,-yres)
    proj = osr.SpatialReference()                 # Establish its coordinate encoding
    proj.ImportFromEPSG(epsg_code)                     # This one specifies WGS84 lat long.
    
    return geotransform, proj

def batchPNGtoGeotiff(folder,
                      csv_file,
                      output_folder,
                      geotiff_folder,
                      epsg_code):

    for image in folder:
        base = os.path.splitext(os.path.basename(image))[0]
        out = base + '.tif'
        out = os.path.join(output_folder, out)
        pngToGeotiff(image, csv_file, out, geotiff_folder, epsg_code)            
    
