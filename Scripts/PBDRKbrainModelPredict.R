# this sets to use CPU - comment out for GPU
Sys.setenv("CUDA_VISIBLE_DEVICES"=-1)
library( abind )
library( ANTsRNet )
library( ANTsR )
library( keras )
library( tensorflow )

args <- commandArgs( trailingOnly = TRUE )
doAff = FALSE
if( length( args ) < 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript PBDRKbrainModelPredict.R",
    " imageFile.ext outputPrefix doAffine \n will output transform and transformed image, maybe visualization aids" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args[2]
  if ( length( args ) > 2 ) doAff = as.numeric( args[3] )
  }
print( paste( "input:", inputFileName, "outputPre:", outputFileName,
  "doAff:", doAff ) )

rdir = "./"
template = antsImageRead( paste0( rdir, "Data/S_template3_BrainCerebellum.nii.gz"))
normimg <-function( img ) {
  resampleImage( img, c(128,128,128), useVoxels=T ) %>%
    n3BiasFieldCorrection( 2 ) %>%
    iMath(  'TruncateIntensity', 0.01, 0.98 ) %>%
    iMath("Normalize" ) %>%
#    histogramMatchImage(  template ) %>%
    iMath("Normalize" )
}

nzimg <- function( img, nzlevel = 0.01 ) {
  makeImage( img*0+1, rnorm(prod(dim(img)),0.0,nzlevel))
}

normimg1 <-function( img ) {
  resampleImage( img, c(128,128,128), useVoxels=T  ) %>%
    iMath(  'TruncateIntensity', 0.01, 0.98 ) %>%
    n3BiasFieldCorrection( 2 ) %>%
    iMath("Normalize" )
}

regressionModel = load_model_hdf5( "./Data/PBDRKbrainModel.h5" )

# efficient prediction
# basisMatrixFN = "Data/warpBasis_matrix.mha"
# basisMat = as.matrix( antsImageRead( basisMatrixFN ) )
# warpmask = antsImageRead( "Data/warpBasis_mask.nii.gz" )
#################################################################
refB = normimg( template )
warpmask = refB * 0 + 1
ncomp = 1
sumMI = 0
print("Begin Transform Estimate")
t1=Sys.time()
wts = c( 0.5, 0.25, 0.13, 0.05, 0.05, 0.05, 0.05, 0.05 )
wts = c( 0.9, 0.05, 0.025 )
wts = c( 1  )
for ( comp in 1:length(wts)) {
    if ( comp == 1 ) {
      affimg = antsImageRead( inputFileName )
      if ( doAff == 1 )
        affimg = antsRegistration( template, affimg, "Affine",
          outprefix = outputFileName )$warpedmovout
      if ( doAff == 2 )
        affimg = antsRegistration( template, affimg, "Affine",
          regIterations = c( 50, 0 ),
          outprefix = outputFileName )$warpedmovout
      }
    if ( comp == 1 ) aa = antsImageClone( affimg )
    temp = normimg( aa )
    xshape = c( 1, dim(temp), 2 )
    myarr = array( dim = xshape )
    myarr[1,,,,1] = as.array( temp )
    myarr[1,,,,2] = as.array( refB )
    newp = predict(regressionModel, myarr )
#    predVecs2 = basisMat %*% t( newp )
    totalWarpB2 = resampleImage( vectorToMultichannel( newp, warpmask ), dim(template), useVoxels=T )
    wt = wts[comp]
    tx = paste0( outputFileName,(1:comp)+10, 'learnedWarp.nii.gz' )
    antsImageWrite( totalWarpB2*wt, paste0( outputFileName,comp+10,'learnedWarp.nii.gz' ) )
    aa = antsApplyTransforms( template, affimg, tx, verbose=F )
    mival = antsImageMutualInformation( template, aa )
    if ( comp == ncomp ) sumMI = sumMI + mival
    print( paste( 'comp:', comp, 'mi:', mival ) )
    }
mival0 = antsImageMutualInformation( template, affimg )
print( paste("Done: MI = ",mival0, ' => ',sumMI ))
t2=Sys.time()
print(t2-t1)
wout = paste0( outputFileName, "learned.nii.gz" )
antsImageWrite( aa, wout )
