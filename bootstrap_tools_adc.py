import sys
import os
from subprocess import call
from itertools import combinations
from joblib import Parallel, delayed

## math imports
import numpy as np
import nibabel as nb

## dipy imports
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.ivim import IvimModel

from geometric_averages_python import geometric_average

from computeLinearADC_image_torch import *


def load_bvals(bvalsfilename):
    
    assert os.path.exists(bvalsfilename)
    
    # Load bvals 
    fo = open( bvalsfilename,'r')
    lines = fo.readlines()
    fo.close()
    bvals = [ int(bb) for bb in lines[0].split(' ')]
    bvals = [round(bval/10)*10 for bval in bvals]
    
    return bvals

def predict_adc_signal(b0,adc,bvalues): 
    
    # make sure bvals are floats 
    bvalues = np.array(bvalues).astype(b0.dtype)
    
    # remove nans in both arrays
    b0_nans = np.isnan(b0)
    adc_nans = np.isnan(adc)
    nans=np.logical_or(b0_nans, adc_nans)
    b0[nans] = 0 
    adc[nans] = 0 
    
    # get signal estimate from IVIM parameter estimates
    b0 = b0[...,np.newaxis] #expand with one new axis so that b* parameter can be broadcast
    adc = adc[...,np.newaxis]
    x,y,z,t = b0.shape

    bvals = np.array(bvalues)
    bvals_ = np.tile(bvals,(x,y,z,1))

    # DEBUGGING
    #     b0 = np.tile(b0,(3,1,1,1,1))
    #     b0= np.squeeze(b0)
    #     b0=np.moveaxis(b0, 0,-1)
    #     x,y,z,t = nz[0][0],nz[1][0],nz[2][0],nz[3][0]
    #     b0[x,y,z,t]
    #     adc[x,y,z,t]
    #     seg[x,y,z]
    #     bvals_[x,y,z,t]
    #     r0=bvals_*adc
    #     r1=-1*bvals_*adc
    #     r2=np.exp(-1*bvals_*adc)
    #     r3=b0*(np.exp(-1*bvals_*adc))
    #     r0[x,y,z,t]
    #from IPython import embed; embed()
    
    signal_image = b0*(np.exp(-1*bvals_*adc))
    #signal_image = np.moveaxis(signal_image,3,0)
    return signal_image
    

# Discard predefined directions from data and fit dipy ivim 
def discardDirections_adc( dwiData, bvals, index_directions_to_discard, numAllImages ):
    
    # define an array of 1s = len(all images ) = all bvalues and all directions  
    list_all_images = np.ones((numAllImages,), dtype=np.int)
    
    
    ######
    ## remove directions from every bvalue>0
    ######
    
    # get index of directions to keep 
    list_all_images[index_directions_to_discard] = 0
    index_directions_to_keep = np.where( list_all_images )[0]
    
    # reduce 4D nifti  
    dwiData_reduced = dwiData[:,:,:,index_directions_to_keep]
    
    # reduce gtab 
    bvals=np.array(bvals)
    bvals_reduced=bvals[index_directions_to_keep]

    return dwiData_reduced, bvals_reduced

def predict_from_bootstrapped3_adc( gg, numAllImages, numDirections, dwiData, bvals, seg=None):

    # select gradients from gtab sequence
    index_directions_to_discard = np.array([ np.arange(gradDir,numAllImages,numDirections) for gradDir in gg ], dtype=np.int).ravel()   
    
    # discard directions from data 
    dwiData_reduced, bvals_reduced = discardDirections_adc( dwiData, bvals, index_directions_to_discard, numAllImages )
    
    # compute averaged signal
    dwiData_reduced_averaged = geometric_average(dwiData_reduced,bvals_reduced)
    bvals_averaged = sorted(np.unique(bvals))
    
    ### CUDA START 
    # read files 
    im_seg = get_segmentation(dwiData_reduced_averaged, segfile=None)
    
    # init variables 
    x,y,z,t=dwiData_reduced_averaged.shape
    nz = np.nonzero(im_seg)
    L=len(nz[0])
    adc = np.zeros((x,y,z))
    b0 = np.zeros((x,y,z))

    # compute via cuda 
    #b0_default,adc_default = computeLinearADC_image(dwiData_averaged,bvals_averaged,seg)
    # fit adc model
    #b0,adc = computeLinearADC_image(dwiData_reduced_averaged,bvals_averaged,seg)    
    im_seg_r = np.reshape(im_seg,(x*y,z,t))
    adc_est, b0_est = computeLinearADC_torch_image_batch_cuda(bvals = bvals_averaged,signal = im_seg_r)
    adc = torch.reshape(adc_est,(x,y,z))
    b0 = torch.reshape(b0_est,(x,y,z))
    adc = adc.cpu().numpy()
    b0 = b0.cpu().numpy()
    #b0_default,adc_default = bo,adc
    ### CUDA END    
        
    return np.array([b0,adc])



################################################################################################
# UNUSED 
################################################################################################

def predict_from_bootstrapped2_adc( gg, numB0s, numAllImages, numDirections, dwiData, bvals, dwiPredict, seg=None):

    # select gradients from gtab sequence
    #index_directions_to_discard = np.array([ np.arange(numB0s+gradDir,numAllImages,numDirections) for gradDir in gg ], dtype=np.int).ravel()   
    # Modifying the code as we will be extracting directions b50 also
    index_directions_to_discard = np.array([ np.arange(gradDir,numAllImages,numDirections) for gradDir in gg ], dtype=np.int).ravel()   
    
    # discard directions from data 
    dwiData_reduced, bvals_reduced = discardDirections_adc( dwiData, bvals, index_directions_to_discard, numAllImages )
    
    # compute averaged signal
    dwiData_reduced_averaged = geometric_average(dwiData_reduced,bvals_reduced)
    bvals_averaged = sorted(np.unique(bvals))
    
    # fit adc model
    b0,adc = computeLinearADC_image(dwiData_reduced_averaged,bvals_averaged,seg)
    
    # predict DWI signal from estimated parameters
    dwiPredict=predict_adc_signal(b0,adc,bvals_averaged)
        
    return dwiPredict




def ravel_no_zeros_nans(a,b,seg=None):
    """ return two arrays with no zeros and no nans in ravel shape that are matched in shape to each other """
    
    if seg is not None: 
        # get non zeros from mask 
        nz = np.where(seg>0)
        a=a[nz[0], nz[1], nz[2], :]
        b=b[nz[0], nz[1], nz[2], :]
    else:
        x,y,z,t = a.shape 
        a=np.reshape(a, (x*y*z,t))
        b=np.reshape(b, (x*y*z,t))

    # get non zero indices of first image (b0)
    a_0 = a[:,0]        
    nz2 = np.where(a_0>0)    
    a=a[nz2]
    b=b[nz2]
    b_0 = b[:,0]        
    nz3 = np.where(b_0>0)    
    a=a[nz3]
    b=b[nz3]
    
    # get no nan indices  
    nn = np.where(~np.isnan(a+b))
        
    return a[nn], b[nn]
    

def ravel_no_zeros_nans_by_bval(a,b,bvals,seg=None):
    
    pass # NOT FINISHED 
    """ return two arrays with no zeros and no nans in ravel shape that are matched in shape to each other """
    
    assert seg is not None
    
    # get non zeros from mask 
    nz = np.where(seg>0)
    a=a[nz[0], nz[1], nz[2], :]
    b=b[nz[0], nz[1], nz[2], :]
    
    

    # get non zero indices of first image (b0)
    a_0 = a[:,0]        
    nz2 = np.where(a_0>0)    
    a=a[nz2]
    b=b[nz2]
    b_0 = b[:,0]        
    nz3 = np.where(b_0>0)    
    a=a[nz3]
    b=b[nz3]
    
    # get no nan indices  
    nn = np.where(~np.isnan(a+b))
        
    return a[nn], b[nn]
            