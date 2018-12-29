# this sets to use CPU - comment out for GPU
Sys.setenv("CUDA_VISIBLE_DEVICES"=-1)
library( abind )
library( ANTsRNet )
library( ANTsR )
library(keras)
library( tensorflow )

args <- commandArgs( trailingOnly = TRUE )
dlbsid = "28400"
dlbsid = "28498"
dlbsid = "28640"
if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript affinebrain2kirby.R",
    " imageFile.ext outputPrefix \n will output matrix and transformed image" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args[2]
  }

print( inputFileName )

rdir = "./"
templateH = antsImageRead( paste0( rdir, "Data/S_template3.nii.gz"))
templateSeg = antsImageRead( paste0( rdir, "Data/S_template_BrainCerebellum-malf_6Labels.nii.gz"))
brainSeg = thresholdImage( templateSeg, 1, 6 )
template = templateH * thresholdImage( templateSeg, 1, 6 )
normimg <-function( img, scl=4 ) {
  iMath( img  %>% iMath( "PadImage", 0 ), "Normalize" ) %>%
    resampleImage( 4  )
}

numRegressors = 10
refH = normimg( template )
input_shape <- c( dim( refH ), refH@components )
regressionModel = load_model_hdf5( "./Data/regiwDeformationBasisalg_Brain2_scl4regressionModel.h5" )
basis = data.matrix( read.csv( "Data/regiwDeformationBasisalg_Brain_scl4regressionModelbasis.csv" ) )
mns = as.numeric( read.csv( "Data/regiwDeformationBasisalg_Brain_scl4regressionModelmn.csv" )[,1] )
newimg = antsImageRead( inputFileName ) %>% iMath("Normalize")
centerOfMassTemplate <- getCenterOfMass( template )
centerOfMassTemplate = c( 101.3143, 140.8385, 140.7970 ) # this was used in training
# 101.3280 140.9331 140.7518
centerOfMassImage <- getCenterOfMass( newimg )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
tarimg = applyAntsrTransformToImage( xfrm, newimg, template )
nimg = normimg( tarimg, scl=scl )
nimgarr = array( as.array(nimg), dim = c(1, dim(refH), 1) )
nimgarr2 = array( as.array(nimg)-as.array(refH), dim = c(1, dim(refH), 1) )
# predParams = regressionModel %>% predict( list(nimgarr,nimgarr2), verbose = 1 )
predParams = regressionModel %>% predict( nimgarr, verbose = 1 )

inp =  basis %*% ( mns +  predParams[1,] )
affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
setAntsrTransformFixedParameters( affTx, centerOfMassTemplate[1:3] )
setAntsrTransformParameters( affTx, inp )
affTx = invertAntsrTransform( affTx )
trnmat = paste0( outputFileName, "translation.mat" )
writeAntsrTransform( xfrm, trnmat )
affmat = paste0( outputFileName, "learnedAffine.mat" )
writeAntsrTransform( affTx, affmat )
tarimgHi = antsApplyTransforms( templateH, newimg, trnmat )
tx = c( affmat, trnmat )
learnedi2 = antsApplyTransforms( template, newimg, tx  )
bestmi =  antsImageMutualInformation( template, learnedi2 )
print( antsImageMutualInformation( template, tarimgHi ) )
print( antsImageMutualInformation( template, learnedi2 ) )
affout = paste0( outputFileName, "learnedAffine.nii.gz" )
antsImageWrite( learnedi2, affout )

inp1 = inp
fignms = paste0( outputFileName, c("view1.png", "view2.png" ) )
print( fignms )
myq = 0.8
plot( learnedi2*100*1, template, outname=fignms[1], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = myq )
plot( learnedi2*100*1, template, outname=fignms[2], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = myq )
print("done with single tx")
q("no")

# derka

# do a voting type of thing
set.seed(1)
ntx = 22
myarr = array( dim = c( ntx, dim( refH ), 1 ) )
xfrmList = list()
txmat = matrix( 0, nrow=ntx, ncol=3 )
for ( k in 1:ntx ) {
  if ( k > 1 ) txmat[k,]  = rnorm(3,0,0.01)*antsGetSpacing(template)
  centerOfMassTemplate = c( 101.3143, 140.8385, 140.7970 ) # this was used in training
  centerOfMassImage <- getCenterOfMass( newimg )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate + txmat[k,] )
  tarimg = applyAntsrTransformToImage( xfrm, newimg, template )
  nimg = normimg( tarimg, scl=scl )
  myarr[k,,,,1]= as.array( nimg )
  xfrmList[[k]] = xfrm
  }
predParams = regressionModel %>% predict( myarr, verbose = 1 )
bestk=0
for ( k in 1:ntx ) {
  inp =  basis %*% ( mns +  predParams[k,] )
  if ( k == 1 ) inp = basis %*% ( colMeans( predParams ) + mns )
  affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
  setAntsrTransformFixedParameters( affTx, centerOfMassTemplate )
  setAntsrTransformParameters( affTx, inp )
  trnmat = paste0( outputFileName, "translation.mat" )
  writeAntsrTransform( xfrmList[[k]], trnmat )
  affmat = paste0( outputFileName, "learnedAffine.mat" )
#  writeAntsrTransform( invertAntsrTransform( affTx ), affmat )
  writeAntsrTransform( affTx, affmat )

  tx = c( affmat, trnmat )
  learnedi2 = antsApplyTransforms( template, newimg, tx  )
  refmi =  antsImageMutualInformation( template, learnedi2 )
  print( paste(k, refmi)) 
  if ( refmi < bestmi ) { bestk = k; bestmi = refmi }
}
##############################################
print( paste( 'bestk',bestk,'bestmi',bestmi) )
k=bestk
inp =  basis %*% ( mns +  predParams[k,] )
if ( k == 1 ) inp = basis %*% ( colMeans( predParams ) + mns )
affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
setAntsrTransformFixedParameters( affTx, centerOfMassTemplate )
setAntsrTransformParameters( affTx, inp )
trnmat = paste0( outputFileName, "translation.mat" )
writeAntsrTransform( xfrmList[[k]], trnmat )
affmat = paste0( outputFileName, "learnedAffine.mat" )
# writeAntsrTransform( invertAntsrTransform( affTx ), affmat )
writeAntsrTransform( affTx, affmat )
affout = paste0( outputFileName, "learnedAffine.nii.gz" )
tx = c( affmat, trnmat )
learnedi = antsApplyTransforms( templateH, newimg, tx )
antsImageWrite( learnedi, affout )

if ( TRUE ) {
fignms = paste0( outputFileName, c("view1v.png", "view2v.png" ) )
print( fignms ) 
myq = 0.8
plot( learnedi*100*1, template, outname=fignms[1], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = myq )
plot( learnedi*100*1, template, outname=fignms[2], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = myq )
print("done")
}
q("no")

dreg = antsRegistration( template, newimg, "Affine" )

print( antsImageMutualInformation( template, dreg$warpedmovout ) )

dreg = antsRegistration( refH, newimg, "Affine" )

print( antsImageMutualInformation( refH, dreg$warpedmovout ) )


# aderkaea
q("no")
plot(  refH, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
dreg = antsRegistration( templateH, tarimg, "SyNOnly", initialTransform =  invertAntsrTransform( affTx ) )
plot(  dreg$warpedmovout*100*1, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
plot( template, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )

flowImages <- function( Xin, batchSize, scl, fullY )
{
  function() {
    nzlevel = sample( c(0,0.001,0.005,0,0,00.01,0.02,0,0.001,0) )[1]
    xshapearr = c( batchSize, xshape )
    myBatch = sample( 1:nrow(Xin) )[1:batchSize]
    Xout = array( Xin[myBatch,,,,], dim = xshapearr )
    nzimg = array( rnorm( prod( dim( Xout ) ), 0, nzlevel ), dim=xshapearr )
    Y = fullY[ myBatch, ]
    return( list( Xout+nzimg, Y ) )
  }
}

#
# fit
tinds = sample( 1:length(fns) )[1:8]
trY = data.matrix( paramMat[-tinds,] )
trX = array( X[-tinds,,,,1], dim = c(nrow(trY), xshape ) )

if ( scl == 4 ) mybs = 16 else mybs = 4
tdgenfun <- flowImages( trX, mybs, scl=scl, fullY=trY )
test = tdgenfun()

# onm = 'diffeoDeformationBasisalg_Diffeo_scl2regressionModelGood.h5'
if ( file.exists( onm ) )
  regressionModel = load_model_hdf5( onm )

# test
teY = paramMat[tinds,]
teX = array( X[tinds,,,,1], dim = c(nrow(teY), xshape ) )
if ( ! exists( "doFit" ) ) doFit = FALSE
if ( doFit )
  for ( mytr in 1:100 ) {
    track <- regressionModel %>% fit_generator(
      generator = tdgenfun,
      steps_per_epoch =  5,
      epochs = 20, verbose=2, validation_data=list(teX,teY)  )
    save_model_hdf5( regressionModel, onm )
    save_model_weights_hdf5( regressionModel, onmw )
    }

# efficient prediction
basisMatrixFN = "warpBasis/warpBasis_matrix.mha"
basisMat = as.matrix( antsImageRead( basisMatrixFN ) )
t1=Sys.time()
pp = teY * 0
for ( k in 1:nrow(pp) )
  pp[k,] = predict(regressionModel, array( teX[k,,,,1], dim = c( 1, xshape ) ) )
predVecs = basisMat %*% t( pp )
t2=Sys.time()
predVecsTru = basisMat %*% t( teY )
print("prediction time")
print( paste( ( t2 - t1 ) / nrow(teY), 'per sample') )
wfns = Sys.glob( "warpBasis/warpBasis*gz")
warpmask = splitChannels( antsImageRead( wfns[1] ) )[[1]] * 0 + 1
#################################################################
refB = normimg( template, 2 )
avgImg = warpmask * 0
avgAff = warpmask * 0
ncomp = 2
sumMI = 0
for ( ww in 1:length(tinds) ) {
  print( paste( 'test', ww, tinds[ ww ] ) )
  for ( comp in 1:ncomp ) {
    affimg = antsImageRead( fns[ tinds[ww] ] )
    if ( comp == 1 ) aa = antsImageClone( affimg )
    newp = predict(regressionModel, array( as.array(normimg(aa,scl)), dim = c( 1, xshape ) ) )
    predVecs2 = basisMat %*% t( newp )
    totalWarpB2 = vectorToMultichannel( predVecs2, warpmask )
    wt = 1.0/ncomp
    antsImageWrite( totalWarpB2*wt, paste0( '/tmp/tempw',comp+10,'.nii.gz' ) )
    tx = paste0( '/tmp/tempw',(1:comp)+10, '.nii.gz' )
    aa = antsApplyTransforms( warpmask, affimg, tx, verbose=F )
    mival = antsImageMutualInformation( refB, aa )
    if ( comp == ncomp ) sumMI = sumMI + mival
    print( paste( 'comp:', comp, 'mi:', mival ) )
    }
  }
  print( sumMI )
# [1] -2.808823 ncomp 2
# [1] -19.13917
# totally independent data
refB = normimg( template, 2 )
for ( rep in 1:2 ) {
if ( rep == 1)
  indep = normimg( antsImageRead( "/tmp/ARSElearnedAffine.nii.gz"), 2 ) %>% iMath("Normalize") else indep = iMath(aa,"Normalize")
temp = normimg( indep, scl ) %>% iMath("Normalize")
teX = array( as.array(temp), dim = c(1, dim( temp ), 1 ) )
pp = predict( regressionModel, teX )
predVecs = basisMat %*% t( pp )
totalWarpB = vectorToMultichannel( predVecs[ , 1 ], warpmask )
antsImageWrite( totalWarpB * 1/rep, '/tmp/temp.nii.gz' )
aa = antsApplyTransforms( refB, indep, '/tmp/temp.nii.gz', verbose=F )
# plot( indep,  nslices=40, ncolumns=10, axis=3 )
# if ( rep == 1)
#  plot( aa,  nslices=40, ncolumns=10, axis=3 )
print( paste( rep,
  antsImageMutualInformation( refB, indep ),
  antsImageMutualInformation( refB, aa ) ) )
}
