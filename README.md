# PBDRKbrain
prior-based deformable brain registration to the kirby template

Data can be downloaded from : 

[https://figshare.com/articles/antsrnetpriorreg/7531625](https://figshare.com/articles/antsrnetpriorreg/7531625)

Register brain data to the symmetric kirby template

the input image should already be affine transformed to the template

if you run Rscript Scripts/PBDRKbrainModelPredict.R image.nii.gz /tmp/ARSE optional-doAff

doAff will run an affine registration internally

the script will output:

* /tmp/ARSElearned.nii.gz  - transformed image

* /tmp/ARSEview[1,2].png  - images to help review results

* /tmp/ARSElearnedWarp.nii.gz - deformable tx

Note:

```
Sys.setenv("CUDA VISIBLE DEVICES"=-1)
```

sets CPU - comment this out in the src/xxx.R to use GPU.

THIS IS A WIP - SUBJECT TO ADDITIONAL NETWORK TRAINING

NOTES:

* should implement a batch mode to take advantage of the GPU
