
# fit ADC 
nifti4D=mc.nii.gz
bvalfile=mc.bval
savedir=$PWD/test
mkdir $savedir
python computeLinearADC_image_torch.py $nifti4D $bvalfile $savedir


# bootstrap ADC (remove N directions from each bvalue - N times - as a permutation )
directios_to_remove=1
segmentation=seg.nii.gz
python $c2 $directios_to_remove $nifti4D $segmentation $bvalfile $savedir