# this sets to use CPU - comment out for GPU
Sys.setenv("CUDA_VISIBLE_DEVICES"=-1)
library( abind )
library( ANTsRNet )
library( ANTsR )
library( keras )
library( tensorflow )

args <- commandArgs( trailingOnly = TRUE )
if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript PBDRKbrainModelPredict.R",
    " imageFile.ext outputPrefix \n will output transform and transformed image, maybe visualization aids" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args[2]
  }

print( inputFileName )

rdir = "./"
template = antsImageRead( paste0( rdir, "Data/S_template3_BrainCerebellum.nii.gz"))
normimg <-function( img  ) {
  resampleImage( img, 2  ) %>% iMath("Normalize" )
}

regressionModel = load_model_hdf5( "./Data/PBDRKbrainModel.h5" )

# efficient prediction
basisMatrixFN = "Data/warpBasis_matrix.mha"
basisMat = as.matrix( antsImageRead( basisMatrixFN ) )
warpmask = antsImageRead( "Data/warpBasis_mask.nii.gz" )
#################################################################
refB = normimg( template )
ncomp = 1
sumMI = 0
print("Begin Transform Estimate")
t1=Sys.time()
for ( comp in 1:ncomp ) {
    if ( comp == 1 ) affimg = antsImageRead( inputFileName )
    if ( comp == 1 ) aa = antsImageClone( affimg )
    temp = normimg(aa)
    xshape = c( 1, dim(temp), 1 )
    newp = predict(regressionModel, array( as.array(temp), dim = xshape ) )
    predVecs2 = basisMat %*% t( newp )
    totalWarpB2 = vectorToMultichannel( predVecs2, warpmask )
    wt = 1.0/ncomp
    print( totalWarpB2 )
    tx = paste0( outputFileName,(1:comp)+10, 'learnedWarp.nii.gz' )
    antsImageWrite( totalWarpB2*wt, paste0( outputFileName,comp+10,'learnedWarp.nii.gz' ) )
    print( tx )
    aa = antsApplyTransforms( template, affimg, tx, verbose=F )
    mival = antsImageMutualInformation( template, aa )
    if ( comp == ncomp ) sumMI = sumMI + mival
    print( paste( 'comp:', comp, 'mi:', mival ) )
    }
print( paste("Done: MI = ",sumMI ))
t2=Sys.time()
print(t2-t1)
wout = paste0( outputFileName, "learned.nii.gz" )
antsImageWrite( aa, wout )
