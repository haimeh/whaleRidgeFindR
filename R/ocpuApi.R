#' @title hashFromImage 
#' @usage curl -v http://localhost:8004/ocpu/library/whaleRidgeFindR/R/hashFromImage/json\
#' -F "imageobj=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test2.jpg"
#' 
#' hashFromImage(imageobj = "yourfile1.jpg")
#' @details \code{traceFromImage} wrapper for use through opencpu.
#' opencpu passes temp object name to function followed by \code{traceToHash}
#' Processes an image(cimg) containing a whaleRidge. 
#' First the image undergoes cleanup through a variety of filters and glare removal via
#' \code{constrainSizeFinImage} and \code{fillGlare}
#' These processes help enhance edge clarity.
#' The trailing edge is highlighted via neural network. 
#' The image is then cropped down to the trailing edge for efficiency purposes.
#' The canny edges are then extracted from the crop and passed to 
#' \code{traceFromCannyEdges}
#' which isolates coordinates for the trailing edge. These coordinates are then passed to
#' \code{extractAnnulus}
#' which collects image data used for identification.
#' Both the coordinates and the image annulus are then returned.
#' @param imageobj character vector which denotes image file "directory/whaleRidgeImage.JPG"
#' @return Value of type list containing:
#' "hash" vector specifying an individual
#' "coordinates" a matrix of coordinates
#' @export

hashFromImage <- function(imageobj, pathNet=NULL, hashNet=NULL)
{

#hashFromImage("~/Work/2023/whaleMatch/images/train2022/000000000001.jpg")
   #mxnetModel <- mxnet::mx.model.load(file.path( system.file("extdata", package="whaleRidgeFindR"),'whaleRidge_triplet32_4096_whaleRidgeal'), 5600)
  if(class(imageobj)=="character" && length(imageobj)==1){
      traceResults <- traceFromImage(whaleRidge=load.image(imageobj),
                                     startStopCoords = NULL,
                                     pathNet = pathNet)
      if(is.null(traceResults[[1]]) | is.null(traceResults[[2]])){return(traceResults)}
      hashResult <- traceToHash(traceData=list(traceResults$annulus), mxnetModel=hashNet)
      edgeCoords <- traceResults$coordinates
      return(list("hash"=hashResult,"coordinates"=edgeCoords))
  }else{stop()}
}

#' @title forkTimeout
#' @details wrapper to enforce time limit for use through opencpu.
#' Enforces hard limit via forking.
#' @param  expr function to evaluate with time limit
#' @param  timeout time limit
#' @param  onTimeoutReturn function to evaluate if we go over time limit
#' @return eresult of expr or result of onTimoutReturn
forkTimeout <- function(expr, timeout=64, onTimeoutReturn = NULL){
  #loadNamespace("parallel")
  #loadNamespace("tools")
  env <- parent.frame()

  child <- parallel::mcparallel(eval(expr, env), mc.interactive=NA)
  out <- parallel::mccollect(child, wait=FALSE, timeout=timeout)

  if(is.null(out)){ # Timed out with no result: kill.
    tools::pskill(child$pid)
    out <- onTimeout
    suppressWarnings(parallel::mccollect(child)) # Clean up.
  }else{
    out <- out[[1L]]
  }
  return(out)
}

#' @title hashesFromImages
#' @usage curl -v http://localhost:8004/ocpu/library/whaleRidgeFindR/R/hashesFromImages/json\
#' -F "img1=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test2.jpg"\
#' -F "img2=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test3.jpg"\
#' -F "test=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test4.jpg"\
#' -F "misc=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test7.jpg"
#' 
#' @details \code{traceFromImage} wrapper for use through opencpu.
#' opencpu passes temp object name to function followed by \code{traceToHash}
#' Processes an image(cimg) containing a whaleRidge. 
#' First the image undergoes cleanup through a variety of filters and glare removal via
#' \code{constrainSizeFinImage} and \code{fillGlare}
#' These processes help enhance edge clarity.
#' The trailing edge is highlighted via neural network. 
#' The image is then cropped down to the trailing edge for efficiency purposes.
#' The canny edges are then extracted from the crop and passed to 
#' \code{traceFromCannyEdges}
#' which isolates coordinates for the trailing edge. These coordinates are then passed to
#' \code{extractAnnulus}
#' which collects image data used for identification.
#' Both the coordinates and the image annulus are then returned.
#' @param  variable number of character vectors which denotes image file "directory/whaleRidgeImage.JPG" to be processed in parallel
#' @return list of two sub lists respectively containing:
#' "hash" vector specifying an individual
#' "coordinates" a 2 column matrix of coordinates
#' each element in this list is nammed with the argument passed in via the api
#' @export
hashesFromImages <- function(...){
  cores=8
  pathNet=NULL
  hashNet=NULL
  if(all(sapply(list(...),class)=="character")){
    #annulus_coordinates = parallel::mclapply(list(...), forkTimeout(traceFromImgWrapper(imageName), timeout=64, onTimeoutReturn = list("hash"=NULL,"coordinates"=NULL)), mc.cores=cores)
    annulus_coordinates = parallel::mclapply(list(...), forkTimeout(function(imageName){
        returnObj <- list()
        traceResults <- traceFromImage(whaleRidge=imager::load.image(imageName),
                       startStopCoords = NULL,
                       pathNet = NULL)
        returnObj[paste0(imageName,"_ann")] <- list(traceResults$annulus)
        returnObj[paste0(imageName,"_coo")] <- list(traceResults$coordinates)
        return(returnObj)
      }, timeout=64, onTimeoutReturn = list("hash"=NULL,"coordinates"=NULL)), mc.cores=cores)
    annulusImgs <- sapply(annulus_coordinates,function(x)
                          {x_tmp <- x[1]; names(x_tmp) <- substr(names(x_tmp), 0,nchar(names(x_tmp))-4); return(x_tmp)} )
    edgeCoords <- sapply(annulus_coordinates,function(x)
                         {x_tmp <- x[2]; names(x_tmp) <- substr(names(x_tmp), 0,nchar(names(x_tmp))-4); return(x_tmp)} )
    return(list("hash"=traceToHash(traceData=annulusImgs, mxnetModel=hashNet),
                "coordinates"=edgeCoords))
  }else{stop()}
}
#hashesFromImages(img0="~/Work/2023/whaleTrace/images/train2022/000000000001.jpg",img1="/home/jaimerilian/Work/2023/whaleTrace/images/train2022/000000000005.jpg")
#forkTimeout(traceFromImgWrapper("~/Work/2023/whaleTrace/images/train2022/000000000001.jpg"))



#' @title hashFromImageAndEdgeCoord 
#' @usage curl -v http://localhost:8004/ocpu/library/whaleRidgeFindR/R/hashFromImageAndEdgeCoord/json \
#' -F "imageobj=@C:/Users/jathompson/Documents/Testingdb/jensImgs/test2.jpg"\
#' -F "xvec=[6,7,8,7,6,5,5,6,7,8,9]"\
#' -F "yvec=[3,4,5,6,6,5,6,6,7,8,9]"
#' 
#' hashFromImageAndEdgeCoord(
#'   imageobj = "yourfile1.jpg",
#'   xvec=c(3,4,5,6,6,5,6,7,8),
#'   yvec=c(6,7,8,7,6,5,5,6,7)
#' )
#' @details \code{extractAnnulus} wrapper for use through opencpu.
#' if coordinates are generated from whaleRidgeFindR, \code{constrainSizeFinImage} 
#' should be called by setting boundResize = 1
#' opencpu passes temp object name to function followed by \code{traceToHash}
#' Coordinates should denote the pixels along the trailing edge of the whaleRidge
#' \code{extractAnnulus}
#' which collects image data used for identification.
#' Coordinates assume the upper left corner is denoted as 1,1 (recall, R is 1 indexed)
#' @param imageobj character vector which denots image file "directory/whaleRidgeImage.JPG"
#' @param xCoordinates x pixel coordinates for data extraction
#' @param yCoordinates y pixel coordinates for data extraction
#' @return hash assiciated with the provided image and trailing edge
#' @export

hashFromImageAndEdgeCoord <- function(imageobj,xvec,yvec,boundResize=F)
{
  if(boundResize)
  {
    whaleRidgeImg <- constrainSizeFinImage(load.image(imageobj),2000,750)
  }else{
    whaleRidgeImg <- load.image(imageobj)
  }
  annulus <- extractAnnulus(imageFromR=whaleRidgeImg,xCoordinates=xvec,yCoordinates=yvec)
  return(traceToHash(list(annulus)))
}
