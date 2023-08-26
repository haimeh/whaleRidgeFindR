
mx.model.init.params_cust <- function (symbol, input.shape, fixed.shape, output.shape, initializer, ctx)
{

    if (!mxnet:::is.MXSymbol(symbol))
        stop("symbol needs to be MXSymbol")
    arg_lst <- list(symbol = symbol)
    arg_lst <- append(arg_lst, input.shape)
    arg_lst <- append(arg_lst, output.shape)
    arg_lst <- append(arg_lst, fixed.shape)

    slist <- do.call(mxnet:::mx.symbol.infer.shape, arg_lst)
    if (is.null(slist)) stop("Not enough information to get shapes")

    arg.params <- mxnet:::mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown = TRUE)
    #arg.params <- mx.init.create(initializer,  
    #                             slist$arg.shapes[-which(names(slist$arg.shapes) %in% names(fixed.shape))]
    #                             ,ctx, skip.unknown = TRUE)
       aux.params <- mxnet:::mx.init.create(initializer, slist$aux.shapes,ctx, skip.unknown = FALSE)
    return(list(arg.params = arg.params, aux.params = aux.params, slist))
}


mx.simple.bind_cust <- function(symbol, ctx, dtype ,grad.req = "null", fixed.param = NULL, slist, ...) {


  if (!mxnet:::is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  #slist <- symbol$infer.shape(list(...))

  if (is.null(slist)) {
    stop("Need more shape information to decide the shapes of arguments")
  }
  #print(slist$arg.shapes)
  #if ( any(sapply(slist$arg.shapes,anyNA)) ) browser()

  arg.arrays <- sapply(slist$arg.shapes, function(shape) {
    mx.nd.array(array(0,shape), ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  aux.arrays <- sapply(slist$aux.shapes, function(shape) {
    mx.nd.array(array(0,shape), ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
    if (nm %in% fixed.param) {
      print("found fixed.param")
      "null"
    } else if (!endsWith(nm, "label") && !endsWith(nm, "data")) {
      grad.req
    } else {
      "null"
    }
  })
  print("BOUND")
  return(mxnet:::mx.symbol.bind(symbol, ctx,
                 arg.arrays=arg.arrays,
                 aux.arrays=aux.arrays,
                 grad.reqs = grad.reqs))
}

#Error in packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] -  :
#  std::exception

predict.MXFeedForwardModel_cust <- function(
	model, 
	X, 
	ctx = NULL, 
	array.batch.size = 128, 
	array.layout = "auto",
    allow.extra.params = TRUE)
{
    if (is.serialized(model))
        model <- mxnet:::mx.unserialize(model)
    if (is.null(ctx))
        ctx <- mxnet:::mx.ctx.default()
    if (is.array(X) || is.matrix(X)) {
        if (array.layout == "auto") {
            array.layout <- mxnet:::mx.model.select.layout.predict(X,
                model)
        }
        if (array.layout == "rowmajor") {
            X <- t(X)
        }
    }
    X <-mxnet:::mx.model.init.iter(X, NULL, batch.size = array.batch.size,
        is.train = FALSE)
    X$reset()
    if (!X$iter.next())
        stop("Cannot predict on empty iterator")
    dlist = X$value()
    ## extract shape based on symbol name
    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #namedShapes <- lapply(model$symbol$arguments,function(x){
	#			  as.integer(strsplit(gsub("\\(([^()]*)\\)|.", "\\1", x, perl=T),",")[[1]])
	#   }
    #)
    #names(namedShapes) <- model$symbol$arguments
    #fixed.shapes <- namedShapes[sapply(namedShapes,function(x)length(x)!=0)]
    ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	fixed.shapes <- append(sapply(model$arg.params,dim),sapply(model$aux.params,dim))

    input.names <- names(dlist)
    input.shape <- sapply(input.names, function(n){dim(dlist[[n]])}, simplify = FALSE)
    input.shape <- input.shape[1]
    #fixed.shapes <- append(namedShapes,lapply(fixed.param,dim))
    initialized <- mx.model.init.params_cust(symbol=model$symbol, 
				   input.shape=input.shape, 
				   fixed.shape=fixed.shapes, 
				   output.shape=NULL, 
				   initializer=mx.init.uniform(0), 
				   ctx=ctx)
    params <- initialized[1:2]
    slist <- initialized[[3]]
    arg_lst <- list(symbol = model$symbol, ctx = ctx, data = dim(dlist$data), grad.req = "null", slist=slist)







    pexec <- do.call(mx.simple.bind_cust, arg_lst)
    if (allow.extra.params) {
        model$arg.params[!names(model$arg.params) %in% arguments(model$symbol)] <- NULL
    }
	mxnet:::mx.exec.update.arg.arrays(pexec, model$arg.params, match.name = TRUE)
    mxnet:::mx.exec.update.aux.arrays(pexec, model$aux.params, match.name = TRUE)
    packer <- mxnet:::mx.nd.arraypacker()
    X$reset()
    while (X$iter.next()) {
        dlist = X$value()
        mxnet:::mx.exec.update.arg.arrays(pexec, list(data = dlist$data),
            match.name = TRUE)
        mxnet:::mx.exec.forward(pexec, is.train = FALSE)
        out.pred <- mxnet:::mx.nd.copyto(pexec$ref.outputs[[1]], mx.cpu())
        padded <- X$num.pad()
        oshape <- dim(out.pred)
        ndim <- length(oshape)
        packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
    }
    X$reset()
    return(packer$get())
}




#' @title traceToHash 
#' @description Function which takes the output of \code{traceFromImage} and returns objects used for matching
#' @return Value of type list containing:
#' "coordinates" a dataframe of coordinates
#' "annulus" a 3 channel image of isolated features
traceToHash <- function(traceData,
                        mxnetModel = NULL)
{
  
  whaleRidgeIter <- getRefClass("whaleRidgeIter",where = as.environment(".whaleRidgeFindREnv"))
  if (is.null(mxnetModel))
 {
    #stop("No model in traceToHash")
    mxnetModel <- mxnet::mx.model.load(file.path( system.file("extdata", package="whaleRidgeFindR"),'SWA_8annulus_cluster_refine2_03-17-May-2023_fin'), 00)
  }
  print("iter")
  iterInputFormat <- sapply(traceData,function(x){as.numeric(resize(x,size_x = 200,interpolation_type = 6))})
  dim(iterInputFormat) <- c(200,16,3,length(traceData))
  # browser()
  #dataIter <- whaleRidgeIter$new(data = iterInputFormat,
  #                        data.shape = 200)
  print("embed")
  #is.mx.dataiter <- function(x) {
  #    any(is(x, "Rcpp_MXNativeDataIter") || is(x, "Rcpp_MXArrayDataIter"))
  #}
  
  #netEmbedding <- mxnet:::predict.MXFeedForwardModel(mxnetModel,

  netEmbedding <- try(predict.MXFeedForwardModel_cust(model=mxnetModel,
                                                     #X=dataIter,
                                                     X=iterInputFormat,
                                                     array.layout = "colmajor",
                                                     ctx= mx.cpu(),
                                                     allow.extra.params=T))
  if(class(netEmbedding)=="try-error"){browser()}

  rm(dataIter)
  gc()
  #dim(netEmbedding) <- c(32,length(traceData),2)
  #netEmbedding <- apply(netEmbedding, 1:2, mean)
  
  print("NeuralNet embedding complete")
  hashList <- lapply(seq_len(ncol(netEmbedding)), function(i) netEmbedding[,i])
  print("listified")
  print(names(traceData))
  names(hashList) <- names(traceData)
  print("labeled")
  return(hashList)
}

#' @title distanceToRef
#' @description Function to calculate the distances from a single query hash to a reference catalogue
#' @param queryHashData vector containing a hash for matching
#' @param referenceHashData matrix containing a reference catalogue of hashes
#' @export 
distanceToRef <- function(queryHash,
                          referenceHash)
{
  if(length(referenceHash)>0 && !is.null(queryHash))
  {
    diff <- apply(referenceHash,2,
                  function(x,queryHash)
                  {
                    distance <- sqrt(sum((as.numeric(x)-as.numeric(queryHash))^2))
                    return(if(!is.nan(distance)){distance}else{0})
                  },queryHash=queryHash)
    return(diff)
  }
}


#' @title distanceToRefParallel 
#' @description Function performing batched matching between a query catalogue to a reference catalogue
#' To call from opencpu, basic format resembles hashes={[0.1, 1.5, 2.2, 3.0],[6.0, 3.3, 4.1, 5.3]}
#' where each vector denotes the featues extracted from an image of a dorsal whale ridge.
#' @usage curl http://localhost:8004/ocpu/library/whaleRidgeFindR/R/distanceToRefParallel/json\
#'  -d "{\
#'  \"queryHashData\":{\"unk1\":[1,2,3]},\
#'  \"referenceHashData\":{\"sal\":[-1,2,4],\"bob\":[-1,-2,-4]},\
#'  \"justIndex\":0}"\
#'  -H "Content-Type: application/json"
#' @param queryHashData dataframe (or list) containing the hashes for matching
#' @param referenceHashData dataframe (or list) containing a reference catalogue of hashes
#' @param batchSize int denoting the number of query instances to process at once
#' @param counterEnvir r environment object to hold a progress counter for display purposes
#' @param displayProgressInShiny bool denoting if function is called inside an rshiny instance
#' @param justIndex bool denoting if comparison should just return index dataframes or ordered names (mainly for opencpu api)
#' @return list of two dataframes. The dataframes are formatted as follows:
#' If justIndex is true:
#' Each row denotes a query image in the same order as provided in the function call. 
#' ie If the first hash in queryHashData was extracted from an image of dolphin "alice", the first row contains matches to dolphin "alice" 
#' Each column represents a potential match from the referenceHashData, 
#' Columns are ordered by proximity of match from with the closest match being in column 1
#' The index refers to the referenceHashData list provided. 
#' If column 1 for dolphin "alice" is 12, then the 12th element in referenceHashData is the best match.
#' "sortingIndex" denotes the element from best to worst match in the reference catalogue.
#' "distances" denotes the distance to the index specified in "sorting Index"
#' If justIndex is false:
#' Two named lists are returned:
#' "sortingIndex" list of vectors containing referenceHashData names, ordered by proximety of match. Each list is named after the queryHashData element it represents
#' "distances" list of vectors containing distances to each match. Each list is named after the queryHashData element it represents
#' @export 
distanceToRefParallel <- function(queryHashData,
                                  referenceHashData,
                                  batchSize = 500,
                                  returnLimit = min(100,length(referenceHashData)),
                                  counterEnvir=new.env(),
                                  displayProgressInShiny=F,
                                  justIndex=T)
{
  fullQueryIndeces <- seq_len(length(queryHashData))
  queryChunkIndex <- split(fullQueryIndeces, ceiling(seq_along(fullQueryIndeces)/batchSize))
  chunkListIndex <- 1
  mxdistanceChunks <- list()
  sortingIndexChunks <- list()
  
  referenceArray <- mx.nd.expand.dims(data=mx.nd.transpose(
    mx.nd.array(data.matrix(data.frame(referenceHashData)))), axis=2)
  
  for(index in queryChunkIndex)
  {
    queryArrayChunk <- mx.nd.expand.dims(data=mx.nd.transpose(
      mx.nd.array(data.matrix(data.frame(queryHashData[index])))), axis=1)
    
    mxdistance <- mx.nd.sqrt(
      mx.nd.nansum(
        mx.nd.square(
          mx.nd.broadcast.sub(
            lhs = queryArrayChunk,
            rhs = referenceArray
          )
        ),axis = 0
      )
    )
    
    
    # browser()
    sortingIndexChunks[[chunkListIndex]] <- mx.nd.take(mx.nd.argsort(mxdistance,axis = 0)+1,mx.nd.array(0:(returnLimit-1)),axis=0)
    mxdistanceChunks[[chunkListIndex]] <- mx.nd.topk(mxdistance,k=returnLimit,axis = 0,is_ascend = T,ret_typ = 'value')
    
    
    rm(queryArrayChunk,mxdistance)
    gc()
    
    chunkListIndex = chunkListIndex+1
    counterEnvir$progressTicker = counterEnvir$progressTicker+length(index)
    
    if(displayProgressInShiny)
    {
      incProgress(length(index)/counterEnvir$length,
                  detail = paste(counterEnvir$progressTicker,"of",counterEnvir$length),
                  session = counterEnvir$reactiveDomain)
    }
  }
  mxdistances <-  mx.nd.concat(mxdistanceChunks,dim = 1)
  mxsortingIndex <-  mx.nd.concat(sortingIndexChunks,dim = 1)
  
  
  # sortingIndex <- as.data.frame(as.array(mx.nd.transpose(mx.nd.argsort(mxdistances,axis = 0)+1)))
  
  distances <- as.data.frame(as.array(mxdistances))
  sortingIndex <- as.array(mxsortingIndex)
  
  #clear the nd arrays
  rm(mxdistanceChunks,mxdistances,sortingIndexChunks,mxsortingIndex,referenceArray)
  gc()
  
  if(justIndex)
  {
    return(list(distances=distances,sortingIndex=as.data.frame(sortingIndex)))
  }else{
    nameTable <- apply(t(sortingIndex),1,function(x)names(referenceHashData)[x])
    # single queries need to be turned back from vectors
    if(nrow(distances)<=1){nameTable <- as.data.frame(t(nameTable))}
    #rownames(nameTable) <- names(queryHashData)
    
    nameTable <- setNames(lapply(split(nameTable, seq(nrow(nameTable))),as.character), names(queryHashData))
    distances <- setNames(lapply(split(distances, seq(nrow(distances))),as.numeric), names(queryHashData))
    
    return(list(distances=distances,sortingIndex=nameTable))
  }
  # sortingIndex <- as.data.frame(as.array(mx.nd.transpose(mxsortingIndex)))
}

