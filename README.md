# adc_dwi_torch

Example: 

```
# average 4D DWI nifti across repetitions (DWI directions) 
python geometric_averages_python.py mc.nii.gz

# fit ADC (CPU only)
nifti4D=mc_gaveraged.nii.gz
bvalfile=mc_gaveraged.bval
savedir=$PWD/test
mkdir $savedir
python computeLinearADC_image_torch.py $nifti4D $bvalfile $savedir "2D_batch"


# fit ADC (requires GPU)
nifti4D=mc_gaveraged.nii.gz
bvalfile=mc_gaveraged.bval
savedir=$PWD/test
mkdir $savedir
python computeLinearADC_image_torch.py $nifti4D $bvalfile $savedir "cuda"


# bootstrap ADC (remove N directions from each bvalue - N times - as a permutation, and fit ADC N times.)
nifti4D=mc.nii.gz
bvalfile=mc.bval
directios_to_remove=1
segmentation=seg.nii.gz
python $c2 $directios_to_remove $nifti4D $segmentation $bvalfile $savedir


```
