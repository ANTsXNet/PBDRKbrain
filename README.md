# PBDRKbrain
prior-based deformable brain registration to the kirby template

Register whole head data to the symmetric kirby template

if you run Rscript Scripts/affine2kirby.R image.nii.gz /tmp/ARSE
the script will output:

* /tmp/ARSE0GenericAffine.mat  - translation tx

* /tmp/ARSElearnedAffine.nii.gz  - affine transformed image

* /tmp/ARSEview[1,2].png  - images to help review results

* /tmp/ARSElearnedAffine.mat - affine tx


To register a brain image ( ie brain after brain extraction ):

```
Rscript Scripts/affinebrain2kirby.R Data/0028442_DLBS_brain.nii.gz /tmp/ARSE
```

Note: 

```
Sys.setenv("CUDA VISIBLE DEVICES"=-1)
```

sets CPU - comment this out in the src/xxx.R to use GPU.

THIS IS A WIP - SUBJECT TO ADDITIONAL NETWORK TRAINING

NOTES:

* could be much faster if we just did a quick CoM alignment - but this likely wont generalize as well

* should implement a batch mode to take advantage of the GPU

