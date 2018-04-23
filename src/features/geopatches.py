import numpy as np
import os
from osgeo import osr, gdal,gdal_array
import h5py

def get_img(loc_img, **keyword_parameters):
    ds = gdal.Open(loc_img, gdal.GA_ReadOnly)
    image_datatype = ds.GetRasterBand(1).DataType
    image = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount),
                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
    for b in range(ds.RasterCount):
        band = ds.GetRasterBand(b + 1)
        image[:, :, b] = band.ReadAsArray()
        
    # band formulas from https://www.indexdatabase.de
    if ('extraindexbands' in keyword_parameters):
        for indexband in keyword_parameters['extraindexbands']:
            # ndvi [nir, red], this is 0 index referenced, not actually related to the satellite band order
            if (indexband['type'] == 'ndvi'):
                image = np.dstack((image,((image[:,:,indexband['nir']] - image[:,:,indexband['red']]) / (image[:,:,indexband['nir']] + image[:,:,indexband['red']]))))

            # evi [nir, red, blue], this is 0 index referenced, not actually related to the satellite band order
            if (indexband['type'] == 'evi'):
                image = np.dstack((image,(2.5 * ((image[:,:,indexband['nir']] - image[:,:,indexband['red']]) / ((image[:,:,indexband['nir']] + 6 * image[:,:,indexband['red']] - 7.5 * image[:,:,indexband['blue']]) + 1)))))

    # force correct proportions for image
    if ('forceproportion' in keyword_parameters):
        image = image[:(image.shape[0] // keyword_parameters['forceproportion'][0] * keyword_parameters['forceproportion'][0]),\
                      :(image.shape[1] // keyword_parameters['forceproportion'][1] * keyword_parameters['forceproportion'][1]),\
                      :]
    
    return image

def get_label(loc_label, **keyword_parameters):
    ds = gdal.Open(loc_label, gdal.GA_ReadOnly)
    image_datatype = ds.GetRasterBand(1).DataType
    image = np.zeros((ds.RasterYSize, ds.RasterXSize, 1),
                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
    band = ds.GetRasterBand(1)
    image[:, :, 0] = band.ReadAsArray()

    # Expand pixel by multiple
    if 'repeatlabel' in keyword_parameters:
        image = np.repeat(np.repeat(image,keyword_parameters['repeatlabel'],axis=1),keyword_parameters['repeatlabel'],axis=0)

    # force correct proportions for image
    if ('forceproportion' in keyword_parameters):
        image = image[:(image.shape[0] // keyword_parameters['forceproportion'][0] * keyword_parameters['forceproportion'][0]),\
                      :(image.shape[1] // keyword_parameters['forceproportion'][1] * keyword_parameters['forceproportion'][1]),\
                      :]

    return image

# https://github.com/dariopavllo/road-segmentation/blob/master/helpers.py

# Create patches from single label image and return list of patches
def label_crop(im, w, h, stride):
    list_patches = []
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    for i in range(0,imgheight - h + 1,stride):
        for j in range(0,imgwidth - w + 1,stride):
            im_patch = im[i:i+h, j:j+w]
            list_patches.append(im_patch)
    return list_patches

# Create patches from single data image and return list of patches
def img_crop(im, w, h, stride, padding):
    list_patches = []
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,imgheight + padding - h + 1,stride):
        for j in range(padding,imgwidth + padding - w + 1,stride):
            im_patch = im[i-padding:i+h+padding, j-padding:j+w+padding, :]
            list_patches.append(im_patch)
    return list_patches

# Create patches from many data images and return linearized arrays (1st dimension is each image)
def create_patches_img(X, width, height, stride, padding):
    img_patches = np.asarray([img_crop(X[i], width, height, stride, padding) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches

# Create patches from many label images and return linearized arrays (1st dimension is each image)
def create_patches_label(X, width, height, stride):
    img_patches = np.asarray([label_crop(X[i], width, height, stride) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
    return img_patches

# Create patches for both data and label images and stack 1st dimension (remove individual image grouping)
def create_data(list_imgdir, dict_imgprop, list_labeldir, dict_labelprop, **keyword_parameters):
    # dictionary properties for image: width, height, stride, padding
    # dictionary properties for label: width, height, stride
    
    arrays_image = []
    arrays_label = []
    
    for imgdir in list_imgdir:
        arrays_image.append(get_img(imgdir, **keyword_parameters))
        
    for labeldir in list_labeldir:
        arrays_label.append(get_label(labeldir, **keyword_parameters))

    patches_data = create_patches_img(np.stack(arrays_image, axis=0),dict_imgprop['width'],dict_imgprop['height'],dict_imgprop['stride'],dict_imgprop['padding'])
    patches_label = create_patches_label(np.stack(arrays_label, axis=0),dict_labelprop['width'],dict_labelprop['height'],dict_labelprop['stride'])
    
    assert patches_data.shape[0] == patches_label.shape[0], 'number of data patches and label patches do not match'
    
    if ('save_location' in keyword_parameters):
        with h5py.File(keyword_parameters['save_location'], 'w') as hf:
            hf.create_dataset("patches_data",  data=patches_data)
            hf.create_dataset("patches_label",  data=patches_label)
        print ('Saved data in', keyword_parameters['save_location'])
    
    return patches_data, patches_label

# Load h5py data
def load_data(save_location):
    with h5py.File(save_location, 'r') as hf:
        patches_data = hf['patches_data'][:]
        patches_label = hf['patches_label'][:]
    return patches_data, patches_label 