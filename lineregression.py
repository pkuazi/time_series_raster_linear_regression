import numpy as np
import os, gdal, sys, osr
from pandas import Series, DataFrame
import pandas as pd
from sklearn.cluster import KMeans

# calculate the slope and intercept of time series raster data
# root_dir = '/mnt/mfs/zjh/rspm25/PM25_CIESIN/china'
root_dir = '/mnt/win/data/PM25/PM25_CHINA/'
# t_series = ['199801_200012', '199901_200112','200001_200212','200101_200312','200201_200412','200301_200512','200401_200612', '200501_200712','200601_200812', '200701_200912', '200801_201012','200901_201112', '201001_201212']
t_series = [str(x) for x in range(1999,2017)]
t_num = len(t_series)

# dst_dir ='/mnt/mfs/zjh/rspm25/PM25_CIESIN/timevariance'
dst_dir = '/mnt/win/data/PM25/time_trend'

def rst2arr(rasterfile):
    dataset = gdal.OpenShared(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize #xsize means number of columns
    ysize = dataset.RasterYSize # ysize means number of rows
    #proj = dataset.GetProjection()
    #geotrans = dataset.GetGeoTransform()
    #noDataValue = band.GetNoDataValue()
    print( "Output file:", rasterfile, 'size: ', xsize, ysize)
    rastervalue = band.ReadAsArray(xoff=0, yoff=0, win_xsize=xsize, win_ysize=ysize)
    return rastervalue

def timevec(root_dir):
    raw_files = os.listdir(root_dir)
    arr_list = []
    for i in range(t_num):
        for file in raw_files:
            if t_series[i] in file and file.endswith('.tif'):
                file_dir = os.path.join(root_dir,file)
                rstarr = rst2arr(file_dir)
                arr_list.append(rstarr)
                arr_arr = np.array(arr_list)
    vec_arr = arr_arr.transpose((1,2,0))
    return vec_arr

# def cluster_kmeans(x_arr):
#     y_pred = KMeans(n_clusters=8).fit_predict(x_arr)
#     return y_pred

def arr2rst(arr, dst_file):
    rst = "/mnt/mfs/zjh/rspm25/PM25_CIESIN/china/cnPM25_199801_200012.tif"
    dataset = gdal.OpenShared(rst)
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    #noDataValue = band.GetNoDataValue()
    # output the array in geotiff format
    xsize, ysize = arr.shape
    dst_format = 'GTiff'
    dst_nbands = 1
    #dst_datatype = gdal.GDT_UInt32
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(geotrans)
    dst_ds.SetProjection(proj)
    #dst_ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    dst_ds.GetRasterBand(1).WriteArray(arr)
    #return dst_file

def fea_extract(record):
    num = len(record)
    x = np.arange(num)
    y = record

    cof = np.polyfit(x, y, 1)
    slope = cof[0]
    intercept = cof[1]

    return slope, intercept

if __name__ == '__main__':

    vec_arr = timevec(root_dir)
    x_size, y_size, dim_num = vec_arr.shape

    slope_arr = np.array([])
    intercept_arr = np.array([])
    for i in range(x_size):
        for j in range(y_size):
            x = np.arange(t_num)
            y = vec_arr[i,j]
            cof = np.polyfit(x, y, 1)
            slope = cof[0]
            intercept = cof[1]

            slope_arr = np.append(slope_arr,slope)
            intercept_arr = np.append(intercept_arr,intercept)

    slope_arr = slope_arr.reshape(x_size,y_size)
    intercept_arr = intercept_arr.reshape(x_size,y_size)

    slope_rst = os.path.join(dst_dir, 'time_slope_float.tif')
    intercept_rst = os.path.join(dst_dir, 'time_intercept_float.tif')
    arr2rst(slope_arr,slope_rst)
    arr2rst(intercept_arr,intercept_rst)


