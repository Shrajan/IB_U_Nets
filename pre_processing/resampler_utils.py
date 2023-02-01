# DOCUMENT INFORMATION
'''
    Project Name: IB U-Nets
    File Name   : resampler_utils.py
    Code Author : Shrajan Bhandary
    Created on  : 02 July 2021
    Program Description:
        This program contains the necessary functions to perform 3D and 4D data resampling.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    02 July 2021       |     Shrajan Bhandary     |  Implemented necessary functions.   |
    |    21 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS

import SimpleITK as sitk
import numpy as np

# IMPLEMENTATION

def getImgExtent(sitkImage, newImage):
    imgSize =sitkImage.GetSize()
    xExtent = [0, imgSize[0]]
    yExtent = [0, imgSize[1]]
    zExtent = [0, imgSize[2]]
    
    minValues = [float('inf'),float('inf'),float('inf')]
    maxValues = [float('-inf'),float('-inf'),float('-inf')]
    
    for xVal in xExtent:
        for yVal in yExtent:
            for zVal in zExtent:
                worldCoordinates = sitkImage.TransformIndexToPhysicalPoint((xVal,yVal,zVal))
                idxCoordinates = newImage.TransformPhysicalPointToIndex(worldCoordinates)
                for idx in range(0,len(minValues)):
                    if idxCoordinates[idx] < minValues[idx]:
                        minValues[idx] = idxCoordinates[idx]
                    if idxCoordinates[idx] > maxValues[idx]:
                        maxValues[idx] = idxCoordinates[idx]

    minWorldValues = newImage.TransformIndexToPhysicalPoint(minValues)
    voxelExtent = np.subtract(maxValues,minValues)
    return minWorldValues,voxelExtent 

'''
Functions "get3dslice" and parts of "resample_4D_images" are from https://discourse.itk.org/t/resampleimagefilter-4d-images/2172/2
Author: SachidanandAlle
Date Taken: 02 July 2021
'''
def get3dslice(image, slice=0):
    size = list(image.GetSize())
    if len(size) == 4:
        size[3] = 0
        index = [0, 0, 0, slice]

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(image)
    return image

def resample_4D_image(sitkImage, newSpacing, interpolation="trilinear", 
                        newDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), change_spacing=False, change_direction=False):
    """
    4D input image will be resampled.
    """    
    # Resample 4D (SITK Doesn't support directly; so iterate through slice and get it done)
    #new_data_list = []
    size = list(sitkImage.GetSize())
    for s in range(size[3]):
        img = get3dslice(sitkImage, s)
        img = resample_3D_image(sitkImage=img, newSpacing=newSpacing, interpolation=interpolation, newDirection=newDirection, change_spacing=change_spacing, change_direction=change_direction )
        break # Get only the first slice T2 modality. 

    return img

def resample_3D_image(sitkImage, newSpacing, interpolation="trilinear", 
                        newDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                        change_spacing=False, change_direction=False, newOrigin=None, postProcess=False):
    """
    3D input image will be resampled and resized as required.
    """    

    resImgFiler = sitk.ResampleImageFilter()
    if change_spacing:
        resImgFiler.SetOutputSpacing(newSpacing)
    else:
        resImgFiler.SetOutputSpacing(sitkImage.GetSpacing())
    
    if interpolation == "BSpline":
        resImgFiler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation == "nearest":
        resImgFiler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == "trilinear":
        resImgFiler.SetInterpolator(sitk.sitkLinear)

    resampledImage = sitk.Image(5, 5, 5, sitk.sitkInt16)
    if change_direction:
        resampledImage.SetDirection(newDirection)
    else:
        resampledImage.SetDirection(sitkImage.GetDirection())

    if newOrigin is None:
        resampledImage.SetOrigin(sitkImage.GetOrigin())
    else:
        resampledImage.SetOrigin(newOrigin)

    if change_spacing:
        resampledImage.SetSpacing(newSpacing)
    else:
        resampledImage.SetSpacing(sitkImage.GetSpacing())
            
    [newOrigin, newSize]= getImgExtent(sitkImage, resampledImage)
    
    # Ensuring a minimum size of 16 is present in the last dimension. 
    # This is only needed for the MSD-prostate dataset.
    if newSize[-1] < 16 and postProcess is False:
        newSize[-1] = 17
    
    resImgFiler.SetSize(sitk.VectorUInt32(newSize.tolist()))
    resImgFiler.SetOutputOrigin( newOrigin )
    if change_direction:
        resImgFiler.SetOutputDirection(newDirection )
    else:
        resImgFiler.SetOutputDirection(sitkImage.GetDirection())

    trans=sitk.Transform()
    trans.SetIdentity()
    resImgFiler.SetTransform( trans ) 
    resampledImage = resImgFiler.Execute(sitkImage)

    return resampledImage
