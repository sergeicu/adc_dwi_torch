"""Usage:


    Args: 
        discardDirs - number of directions to discard 
        filename - 4D nifti filename 
        segname - 3D segmentation filename 
        bvalsfilename 
        savedir - save directory 
"""


# Separate error calculation from bootstrap. 

# Predict mse over a mask
import sys 

import warnings

from sklearn.metrics import mean_squared_error as mse 

from bootstrap_tools_adc import *

from computeLinearADC_image_torch import *

if __name__ == '__main__':
    
    
    discardDirs = int(sys.argv[1])
    niftipath = sys.argv[2]
    segfilename=sys.argv[3]
    bvalsfilename=sys.argv[4]
    savedir=sys.argv[5]+'/'
    
    ####################################
    # inputs 
    ####################################

    num_cores = 40
    numOfDirections2Discard = [discardDirs]
    savenifti = True

    ####################################
    # load data and params 
    ####################################
    
    # load bvals and bvecs    
    bvals = load_bvals(bvalsfilename)

    # Get params 
    numAllImages = len(bvals)
    numDirections = len(np.where(np.array(bvals)==np.unique(bvals)[1])[0])
    num2PermuteFrom = numDirections # total number of directions which we would be permuting from 
    nii_basename = os.path.basename(niftipath).replace(".nii.gz", "_") # assumes that files will be name as svr.nii.gz or vol.nii.gz

    # load nifti 
    assert os.path.exists(niftipath)
    imo = nb.load(niftipath)
    dwiData = imo.get_fdata()
    
    # load mask 
    assert os.path.exists(segfilename)
    seg = nb.load(segfilename).get_fdata()    
    assert dwiData.shape[:-1] == seg.shape

    
    ####################################
    # default prediction (no directions removed) 
    ####################################

    # compute averaged signal
    dwiData_averaged = geometric_average(dwiData,bvals)
    bvals_averaged = sorted(np.unique(bvals))   
    
    # fit ADC model

    ### CUDA START 
    # read files 
    im_seg = get_segmentation(dwiData_averaged, segfile=None)
    
    # init variables 
    x,y,z,t=dwiData_averaged.shape
    nz = np.nonzero(im_seg)
    L=len(nz[0])
    adc = np.zeros((x,y,z))
    b0 = np.zeros((x,y,z))

    # compute via cuda 
    #b0_default,adc_default = computeLinearADC_image(dwiData_averaged,bvals_averaged,seg)
    im_seg_r = np.reshape(im_seg,(x*y,z,t))
    adc_est, b0_est = computeLinearADC_torch_image_batch_cuda(bvals = bvals_averaged,signal = im_seg_r)
    adc = torch.reshape(adc_est,(x,y,z))
    b0 = torch.reshape(b0_est,(x,y,z))
    adc = adc.cpu()
    b0 = b0.cpu()
    b0_default,adc_default = b0.numpy(),adc.numpy()
    ### CUDA END


    # predict DWI signal from estimated parameters
    dwiPredict_default=predict_adc_signal(b0_default,adc_default,bvals_averaged)

    ####################################
    # Bootstrap - discardDirections
    ####################################

    # calculate error for base case 
    dwiPredict = [dwiPredict_default]
    paramPredict = [np.array([b0_default, adc_default])]
    
    assert os.path.exists(savedir)
    for dirs2Discard in numOfDirections2Discard:

        print(dirs2Discard)

        # compute all combinations of gradient directions
        dirs2DiscardPermutations = list(combinations( range(num2PermuteFrom), dirs2Discard ))

        # for every combination
        print("Starting bootstrap")
        #result = Parallel(n_jobs=num_cores)(delayed(predict_from_bootstrapped3_adc)(gg, numAllImages, numDirections, dwiData, bvals, seg) for gg in dirs2DiscardPermutations )
        #result = predict_from_bootstrapped3_adc[dirs2DiscardPermutations[0], numAllImages, numDirections, dwiData, bvals, seg)
        result = []
        for iii, gg in enumerate(dirs2DiscardPermutations):
            print(f"{iii}/{len(dirs2DiscardPermutations)}")

            out = predict_from_bootstrapped3_adc(gg, numAllImages, numDirections, dwiData, bvals, seg)
            result.append(out)
            #result = predict_from_bootstrapped3_adc[dirs2DiscardPermutations[0], numAllImages, numDirections, dwiData, bvals, seg)

        #from IPython import embed; embed()
        # prepend
        paramPredict = paramPredict + result

        # save predicted nifti     
        print("Saving images")        
        for i, paramPredict_i in enumerate(paramPredict):

            # unpack
            b0_i = paramPredict_i[0,:,:,:]
            adc_i = paramPredict_i[1,:,:,:]

            # predict DWI signal from estimated parameters
            dwiPredict_i=predict_adc_signal(b0_i,adc_i,bvals_averaged)                

            # create new nifti objects 
            imonew_dwi = nb.Nifti1Image(dwiPredict_i, affine=imo.affine,header=imo.header)
            imonew_b0 = nb.Nifti1Image(b0_i, affine=imo.affine,header=imo.header)
            imonew_adc = nb.Nifti1Image(adc_i, affine=imo.affine,header=imo.header)

            # create savenames
            if i==0:
                savename_dwi = savedir + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_nobootstrap.nii.gz"
                savename_b0 = savedir + "b0_" + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_nobootstrap.nii.gz"
                savename_adc = savedir + "adc_" + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_nobootstrap.nii.gz"
            else:
                savename_dwi = savedir + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_iter_" + str(i).zfill(2) + ".nii.gz"
                savename_b0 = savedir + "b0_" + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_iter_" + str(i).zfill(2) + ".nii.gz"
                savename_adc = savedir + "adc_" + nii_basename + "dirs2Discard_" +str(dirs2Discard) + "_iter_" + str(i).zfill(2) + ".nii.gz"
                
            # save to files
            nb.save(imonew_dwi,savename_dwi)
            nb.save(imonew_b0,savename_b0)
            nb.save(imonew_adc,savename_adc)
                

    print("Done")
    
