#
#   Copyright 2007-2021 by the individuals mentioned in the source code history
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

setClass(Class = "MxExpectationRAM",
	representation = representation(
		A = "MxCharOrNumber",
		S = "MxCharOrNumber",
		F = "MxCharOrNumber",
		M = "MxCharOrNumber",
		thresholds = "MxCharOrNumber",
		dims = "character",
		usePPML = "logical",
		ppmlData = "MxData",
		UnfilteredExpCov = "matrix",
	    numStats = "numeric",
	    between = "MxOptionalCharOrNumber",
    isProductNode = "MxOptionalLogical",
	    verbose = "integer",
	    .rampartCycleLimit = "integer",
	    .rampartUnitLimit = "integer",
	    .useSufficientSets = "logical",
	    .forceSingleGroup = "logical",
	    .analyzeDefVars = "logical",
	    .maxDebugGroups = "integer",
    .optimizeMean = "integer",
    .useSparse = "logical"
	),
	contains = "BaseExpectationNormal")

setMethod("initialize", "MxExpectationRAM",
	function(.Object, A, S, F, M, dims, thresholds, threshnames,
           between, verbose, useSparse, expectedCovariance, expectedMean, discrete,
           selectionVector, expectedFullCovariance, expectedFullMean,
           data = as.integer(NA), name = 'expectation') {
		.Object@name <- name
		.Object@A <- A
		.Object@S <- S
		.Object@F <- F
		.Object@M <- M
		.Object@data <- data
		.Object@dims <- dims
		.Object@thresholds <- thresholds
    .Object@discrete <- discrete
    .Object@.discreteCheckCount <- TRUE
    .Object@selectionVector <- selectionVector
		.Object@threshnames <- threshnames
		.Object@usePPML <- FALSE
		.Object@UnfilteredExpCov <- matrix()
		.Object@between <- between
		.Object@verbose <- verbose
		.Object@.rampartCycleLimit <- as.integer(NA)
		.Object@.rampartUnitLimit <- as.integer(NA)
		.Object@.forceSingleGroup <- FALSE
		.Object@.analyzeDefVars <- TRUE
		.Object@.useSufficientSets <- TRUE
		.Object@.maxDebugGroups <- 0L
		.Object@.optimizeMean <- 2L
    .Object@.useSparse <- useSparse
    .Object@expectedCovariance <- expectedCovariance
    .Object@expectedMean <- expectedMean
    .Object@expectedFullCovariance <- expectedFullCovariance
    .Object@expectedFullMean <- expectedFullMean
    .Object@.canProvideSufficientDerivs <- FALSE
		return(.Object)
	}
)

setMethod("genericExpDependencies", signature("MxExpectationRAM"),
	function(.Object, dependencies) {
    dependencies <- callNextMethod()
	sources <- c(.Object@A, .Object@S, .Object@F, .Object@M, .Object@thresholds, .Object@between)
	sources <- sources[!is.na(sources)]
  sink <- .Object@name
    sink <- c(sink, .Object@expectedCovariance, .Object@expectedMean,
              .Object@expectedFullCovariance, .Object@expectedFullMean)
	dependencies <- imxAddDependency(sources, sink, dependencies)
	return(dependencies)
})

setMethod("qualifyNames", signature("MxExpectationRAM"),
	function(.Object, modelname, namespace) {
    .Object <- callNextMethod()
		.Object@name <- imxIdentifier(modelname, .Object@name)
    for (sl in c('A', 'S', 'F', 'M', 'thresholds', 'between',
                 'expectedCovariance', 'expectedMean',
                 'expectedFullCovariance', 'expectedFullMean')) {
      slot(.Object, sl) <- imxConvertIdentifier(slot(.Object, sl), modelname, namespace, TRUE)
    }
    .Object
})

setMethod("genericExpRename", signature("MxExpectationRAM"),
	function(.Object, oldname, newname) {
    .Object <- callNextMethod()
    for (sl in c('A', 'S', 'F', 'M', 'thresholds', 'between',
                 'expectedCovariance', 'expectedMean',
                 'expectedFullCovariance', 'expectedFullMean')) {
      slot(.Object, sl) <- renameReference(slot(.Object, sl), oldname, newname)
    }
    .Object
})

setMethod("genericExpFunConvert", signature("MxExpectationRAM"),
	function(.Object, flatModel, model, labelsData, dependencies) {
		modelname <- imxReverseIdentifier(model, .Object@name)[[1]]
		name <- .Object@name
		aMatrix <- .Object@A
		sMatrix <- .Object@S
		fMatrix <- .Object@F
		mMatrix <- .Object@M
		data <- .Object@data
		if(is.na(data)) {
			msg <- paste("The RAM expectation function",
				"does not have a dataset associated with it in model",
				omxQuotes(modelname),
				"\nSee ?mxData() to see how to add data to your model")
			stop(msg, call. = FALSE)
		}
		mxDataObject <- flatModel@datasets[[.Object@data]]
		if (.hasSlot(.Object, "between") && length(.Object@between)) {
			sapply(.Object@between, function(bName) {
				zMat <- flatModel[[ bName ]]
				if (is.null(zMat)) {
					msg <- paste("Level transition matrix", omxQuotes(bName),
						     "listed in", omxQuotes(name), "is not found")
					stop(msg, call. = FALSE)
				}
				expName <- paste0(zMat@joinModel, imxSeparatorChar, 'expectation')
				upperF <- flatModel[[ flatModel@expectations[[ expName ]]$F ]]
				lowerF <- flatModel[[ fMatrix ]]

				if (length(rownames(zMat)) != length(colnames(lowerF))) {
					msg <- paste("Join mapping matrix", zMat@name,
						     "must have", length(colnames(lowerF)), "rows:",
						     omxQuotes(colnames(lowerF)))
					stop(msg, call. = FALSE)
				}
				lowerMatch <- rownames(zMat) == colnames(lowerF)
				if (any(!lowerMatch)) {
					msg <- paste("Join mapping matrix", zMat@name,
						     "needs mapping rows for",
						     omxQuotes(colnames(lowerF)[!lowerMatch]))
					stop(msg, call. = FALSE)
				}
				if (length(colnames(zMat)) != length(colnames(upperF))) {
					msg <- paste("Join mapping matrix", zMat@name,
						     "must have", length(colnames(upperF)), "columns:",
						     omxQuotes(colnames(upperF)))
					stop(msg, call. = FALSE)
				}
				upperMatch <- colnames(zMat) == colnames(upperF)
				if (any(!upperMatch)) {
					msg <- paste("Join mapping matrix", zMat@name,
						     "needs mapping columns for",
						     omxQuotes(colnames(upperF)[!upperMatch]))
					stop(msg, call. = FALSE)
				}
			})
		}
		checkNumericData(mxDataObject)
		verifyObservedNames(mxDataObject@observed, mxDataObject@means, mxDataObject@type, flatModel, modelname, "RAM")
		fMatrix <- flatModel[[fMatrix]]@values
		if (is.null(dimnames(fMatrix))) {
			msg <- paste("The F matrix of model",
				omxQuotes(modelname), "does not contain dimnames")
			stop(msg, call. = FALSE)
		}
		if (is.null(dimnames(fMatrix)[[2]])) {
			msg <- paste("The F matrix of model",
				omxQuotes(modelname), "does not contain colnames")
			stop(msg, call. = FALSE)
		}
		hasMeanModel <- !is.na(mMatrix)
		mMatrix <- flatModel[[mMatrix]]
		if (hasMeanModel && !is.null(mMatrix)) {
			means <- dimnames(mMatrix)
			if (is.null(means)) {
				msg <- paste("The M matrix associated",
				"with the RAM expectation function in model",
				omxQuotes(modelname), "does not contain dimnames.")
				stop(msg, call. = FALSE)
			}
			meanRows <- means[[1]]
			meanCols <- means[[2]]
			if (!is.null(meanRows) && length(meanRows) > 1) {
				msg <- paste("The M matrix associated",
				"with the RAM expectation function in model",
				omxQuotes(modelname), "is not a 1 x N matrix.")
				stop(msg, call. = FALSE)
			}
			if (!identical(dimnames(fMatrix)[[2]], meanCols)) {
				msg <- paste("The column names of the F matrix",
					"and the column names of the M matrix",
					"in model",
					omxQuotes(modelname), "do not contain identical",
					"names.")
				stop(msg, call. = FALSE)
			}
		}
		translatedNames <- modelManifestNames(fMatrix, modelname)
		if (length(translatedNames)) {
			.Object@dataColumnNames <- translatedNames
			.Object@dataColumns <- generateDataColumns(flatModel, translatedNames, data)
			if (mxDataObject@type == 'raw') {
				threshName <- .Object@thresholds
				verifyThresholds(flatModel, model, labelsData, data, translatedNames, threshName)
				if (length(mxDataObject@observed) == 0) {
					.Object@data <- as.integer(NA)
				}
				if (single.na(.Object@dims)) {
					.Object@dims <- translatedNames
				}
			} else {
				.Object@thresholds <- as.integer(NA)
				targetNames <- observedDataNames(mxDataObject)
				if (!setequal(translatedNames, targetNames)) {
					varsNotInData <- translatedNames[!(translatedNames %in% targetNames)]
					msg <- paste("The names of the manifest",
						     "variables in the F matrix of model",
						     omxQuotes(modelname), "does not match the",
						     "dimnames of the observed covariance matrix.")
					if (length(varsNotInData) > 0) {
						msg <- paste(msg,
							     "To get you started, the following variables are used but",
							     "are not in the observed data:",
							     omxQuotes(varsNotInData))
					}
					stop(msg, call. = FALSE)
				}
			}
		} else {
			.Object@thresholds <- as.integer(NA)
		}
    .Object@selectionPlan <- prepSelectionPlan(.Object@selectionPlan, colnames(fMatrix))
		if(length(.Object@dims) > nrow(fMatrix) && length(translatedNames) == nrow(fMatrix)){
			.Object@dims <- translatedNames
		}
    callNextMethod(.Object, flatModel, model, labelsData, dependencies)
})

setMethod("genericNameToNumber", signature("MxExpectationRAM"),
	  function(.Object, flatModel, model) {
      .Object <- callNextMethod()
		  name <- .Object@name
    for (sl in c('A', 'S', 'F', 'M', 'between',
                 'expectedFullCovariance', 'expectedFullMean')) {
		  slot(.Object,sl) <- imxLocateIndex(flatModel, slot(.Object,sl), name)
    }
      .Object
	  })

setMethod("genericGetExpected", signature("MxExpectationRAM"),
	  function(.Object, model, what, defvar.row=1, subname=model@name) {
		  ret <- callNextMethod()
		  Aname <- .modifyDottedName(subname, .Object@A, sep=".")
		  Sname <- .modifyDottedName(subname, .Object@S, sep=".")
		  Fname <- .modifyDottedName(subname, .Object@F, sep=".")
		  Mname <- .modifyDottedName(subname, .Object@M, sep=".")
		  A <- mxEvalByName(Aname, model, compute=TRUE, defvar.row=defvar.row)
		  S <- mxEvalByName(Sname, model, compute=TRUE, defvar.row=defvar.row)
		  F <- mxEvalByName(Fname, model, compute=TRUE, defvar.row=defvar.row)
		  I <- diag(1, nrow=nrow(A))
      # need to compute covariance when there is Pearson selection
      ImA <- solve(I-A)
      origCov <- list()
      origCov[[1]] <- ImA %*% S %*% t(ImA)
      if (single.na(.Object@selectionVector)) {
        cov <- origCov[[1]]
      } else {
        selPlan <- .Object@selectionPlan
        selVecName <- .modifyDottedName(subname, .Object@selectionVector)
        selVec <- mxEvalByName(selVecName, model, compute=TRUE, defvar.row=defvar.row)
        sx <- 1L
        rx <- 1L
        curStep <- selPlan[sx,'step']
        nc <- origCov[[sx]]
        newCov <- list()
        while (rx <= nrow(selPlan)) {
          nc[selPlan[rx,'from'],selPlan[rx,'to']] <- selVec[rx,1]
          nc[selPlan[rx,'to'],selPlan[rx,'from']] <- selVec[rx,1]
          if (rx == nrow(selPlan) || (rx < nrow(selPlan) && curStep != selPlan[rx+1,'step'])) {
            newCov[[sx]] <- nc
            cov <- mxPearsonSelCov(origCov[[sx]], newCov[[sx]])
            if (rx < nrow(selPlan)) {
              sx <- sx + 1
              origCov[[sx]] <- cov
              nc <- origCov[[sx]]
              curStep <- selPlan[sx,'step']
            }
          }
          rx <- rx + 1
        }
      }
		  if (any(c('covariance','covariances') %in% what)) {
			  ret[['covariance']] <- F %*% cov %*% t(F)
		  }
		  if (any(c('slope','slopes') %in% what)) {
				if (!single.na(Mname)){
          latents <- setdiff(colnames(F), rownames(F))
          M <- model[[ Mname ]]
          exo <- latents[grep('data.', M[,latents]$labels, fixed=TRUE)]
          if (length(exo)) {
            ret[['slope']] <- A[rownames(F), exo]
          }
        }
      }
		  if (any(c('mean','means') %in% what)) {
				if(single.na(Mname)){
					mean <- matrix( , 0, 0)
				} else {
					Mname <- .modifyDottedName(subname, Mname, sep=".")
					M <- mxEvalByName(Mname, model, compute=TRUE, defvar.row=defvar.row)
					fullMean <- M %*% t(solve(I-A))
          if (!single.na(.Object@selectionVector)) {
            for (sx in 1:length(origCov)) {
              fullMean <- t(mxPearsonSelMean(origCov[[sx]], newCov[[sx]], t(fullMean)))
            }
          }
          mean <- fullMean %*% t(F)
			  }
				ret[['means']] <- mean
			}
			ret
})

##' omxGetRAMDepth
##'
##' Get the potency of a matrix for inversion speed-up
##'
##' @param A MxMatrix object
##' @param maxdepth Numeric. maximum depth to check
##' @details This function is used internally by the \link{mxExpectationRAM} function
##' to determine how far to expand \eqn{(I-A)^{-1} = I + A + A^2 + A^3 + ...}.  It is
##' similarly used by \link{mxExpectationLISREL} in expanding \eqn{(I-B)^{-1} = I + B + B^2 + B^3 + ...}.
##' In many situations \eqn{A^2} is a zero matrix (nilpotent of order 2).  So when \eqn{A} has large
##' dimension it is much faster to compute \eqn{I+A} than \eqn{(I-A)^{-1}}.
omxGetRAMDepth <- function(A, maxdepth = nrow(A) - 1) {
	mxObject <- A
	aValues <- matrix(0, nrow(mxObject), ncol(mxObject))
	defvars <- apply(mxObject@labels, c(1,2), imxIsDefinitionVariable)
	squarebrackets <- mxObject@.squareBrackets
	aValues[mxObject@free] <- 1
	aValues[mxObject@values != 0] <- 1
	aValues[defvars] <- 1
	aValues[squarebrackets] <- 1
	depth <- generateDepthHelper(aValues, aValues, 0, maxdepth)
	#print(depth)
	depth
}

generateDepthHelper <- function(aValues, currentProduct, depth, maxdepth) {
	#print(currentProduct)
	if (depth > maxdepth) {
		return(as.integer(NA))
	}
	if (all(currentProduct == 0)) {
		return(as.integer(depth))
	} else {
		return(generateDepthHelper(aValues, currentProduct %*% aValues, depth + 1, maxdepth))
	}
}

modelManifestNames <- function(fMatrix, modelName) {
	retval <- character()
	if (length(fMatrix) == 0) return(retval)
	colNames <- dimnames(fMatrix)[[2]]
	for(i in 1:nrow(fMatrix)) {
		irow <- fMatrix[i,]
		matches <- which(irow == 1)
		if (length(matches) != 1) {
			err <- paste("The model",
				omxQuotes(modelName), "does not contain",
				"a valid F matrix")
			stop(err, call. = FALSE)
		}
		retval[[i]] <- colNames[[matches[[1]]]]
	}
	return(retval)
}

updateRAMdimnames <- function(flatExpectation, flatJob) {
	fMatrixName <- flatExpectation@F
	mMatrixName <- flatExpectation@M
	if (is.na(mMatrixName)) {
		mMatrix <- NA
	} else {
		mMatrix <- flatJob[[mMatrixName]]
	}
	fMatrix <- flatJob[[fMatrixName]]
	if (is.null(fMatrix)) {
		modelname <- getModelName(flatExpectation)
		stop(paste("Unknown F matrix name",
			omxQuotes(simplifyName(fMatrixName, modelname)),
			"detected in the RAM expectation function",
			"of model", omxQuotes(modelname)), call. = FALSE)
	}
	dims <- flatExpectation@dims
	if (!is.null(dimnames(fMatrix)) && !single.na(dims) &&
		!identical(dimnames(fMatrix)[[2]], dims)) {
		modelname <- getModelName(flatExpectation)
		msg <- paste("The F matrix associated",
			"with the RAM expectation function in model",
			omxQuotes(modelname), "contains dimnames and",
			"the expectation function has specified dimnames")
		stop(msg, call.=FALSE)
	}
	if (is.null(dimnames(fMatrix)) && !single.na(dims)) {
		dimnames(flatJob[[fMatrixName]]) <- list(c(), dims)
	}

	if (!isS4(mMatrix) && (is.null(mMatrix) || is.na(mMatrix))) {
		return(flatJob)
	}

	if (!is.null(dimnames(mMatrix)) && !single.na(dims) &&
		!identical(dimnames(mMatrix), list(NULL, dims))) {
		modelname <- getModelName(flatExpectation)
		msg <- paste("The M matrix associated",
			"with the RAM expectation function in model",
			omxQuotes(modelname), "contains dimnames and",
			"the expectation function has specified dimnames")
		stop(msg, call.=FALSE)
	}

	if (is.null(dimnames(mMatrix)) && !single.na(dims)) {
		dimnames(flatJob[[mMatrixName]]) <- list(NULL, dims)
	}

	return(flatJob)
}

setMethod("genericExpAddEntities", "MxExpectationRAM",
	  function(.Object, job, flatJob, labelsData) {
      job <- constrainCorData(.Object, nrow(job[[ .Object$F ]]), job, flatJob)

		  ppmlModelOption <- job@options$UsePPML
		  if (is.null(ppmlModelOption)) {
			  enablePPML <- (getOption("mxOptions")$UsePPML == "Yes")
		  } else {
			  enablePPML <- (ppmlModelOption == "Yes")
		  }

		  if (enablePPML) {
			  aMatrix <- job[[.Object@A]]
			  aMatrixFixed <- !is.null(aMatrix) && is(aMatrix, "MxMatrix") && all(!aMatrix@free)
			  enablePPML <- aMatrixFixed
		  }

		  if (enablePPML) {
			  job <- PPMLTransformModel(job)
			  job@.newobjects <- TRUE
		  }

		  return(job)
	  })

setMethod("genericExpConvertEntities", "MxExpectationRAM",
	function(.Object, flatModel, namespace, labelsData) {
		if(is.na(.Object@data)) {
			modelname <- getModelName(.Object)
			msg <- paste("The RAM expectation function",
				"does not have a dataset associated with it in model",
				omxQuotes(modelname))
			stop(msg, call.=FALSE)
		}

		flatModel <- updateRAMdimnames(.Object, flatModel)

		if (flatModel@datasets[[.Object@data]]@type != 'raw') {
			return(flatModel)
		}

		flatModel <- updateThresholdDimnames(.Object, flatModel, labelsData)

		return(flatModel)
	}
)

##' imxSimpleRAMPredicate
##'
##' This is an internal function exported for those people who know
##' what they are doing.
##'
##' @param model model
imxSimpleRAMPredicate <- function(model) {
	if (is.null(model$expectation) || !is(model$expectation, "MxExpectationRAM")) {
		return(FALSE)
	}
	nameA <- model$expectation@A
	nameS <- model$expectation@S
	A <- model[[nameA]]
	S <- model[[nameS]]
	if (is.null(A) || is.null(S)) {
		return(FALSE)
	}
	return(is(A, "MxMatrix") && is(S, "MxMatrix"))
}

mxExpectationRAM <- function(A="A", S="S", F="F", M = NA, dimnames = NA, thresholds = NA,
                             threshnames = dimnames, ..., between=NULL, verbose=0L, .useSparse=NA,
                             expectedCovariance=NULL, expectedMean=NULL,
                             discrete = as.character(NA), selectionVector = as.character(NA),
                             expectedFullCovariance=NULL, expectedFullMean=NULL) {

	prohibitDotdotdot(list(...))

	if (typeof(A) != "character") {
		msg <- paste("argument 'A' is not a string",
			"(the name of the 'A' matrix)")
		stop(msg)
	}
	if (typeof(S) != "character") {
		msg <- paste("argument 'S' is not a string",
			"(the name of the 'S' matrix)")
		stop(msg)
	}
	if (typeof(F) != "character") {
		msg <- paste("argument 'F' is not a string",
			"(the name of the 'F' matrix)")
		stop(msg)
	}
	if (!(single.na(M) || typeof(M) == "character")) {
		msg <- paste("argument M is not a string",
			"(the name of the 'M' matrix)")
		stop(msg)
	}
	if (is.na(M)) M <- as.integer(NA)
	if (single.na(thresholds)) thresholds <- as.character(NA)
	if (single.na(dimnames)) dimnames <- as.character(NA)
	if (!is.vector(dimnames) || typeof(dimnames) != 'character') {
		stop("Dimnames argument is not a character vector")
	}
	if (length(thresholds) != 1) {
		stop("Thresholds argument must be a single matrix or algebra name")
	}
	if (length(dimnames) == 0) {
		stop("Dimnames argument cannot be an empty vector")
	}
	if (length(dimnames) > 1 && any(is.na(dimnames))) {
		stop("NA values are not allowed for dimnames vector")
	}
	threshnames <- checkThreshnames(threshnames)
	return(new("MxExpectationRAM", A, S, F, M, dimnames, thresholds, threshnames,
             between, as.integer(verbose), as.logical(.useSparse),
             expectedCovariance, expectedMean, discrete, selectionVector,
             expectedFullCovariance, expectedFullMean))
}

displayMxExpectationRAM <- function(expectation) {
	cat("MxExpectationRAM", omxQuotes(expectation@name), '\n')
	cat("$A :", omxQuotes(expectation@A), '\n')
	cat("$S :", omxQuotes(expectation@S), '\n')
	cat("$F :", omxQuotes(expectation@F), '\n')
	if (is.na(expectation@M)) {
		cat("$M :", expectation@M, '\n')
	} else {
		cat("$M :", omxQuotes(expectation@M), '\n')
	}
	if (single.na(expectation@dims)) {
		cat("$dims : NA \n")
	} else {
		cat("$dims :", omxQuotes(expectation@dims), '\n')
	}
	if (single.na(expectation@thresholds)) {
		cat("$thresholds : NA \n")
	} else {
		cat("$thresholds :", omxQuotes(expectation@thresholds), '\n')
	}
	if (single.na(expectation@discrete)) {
		cat("$discrete : NA \n")
	} else {
		cat("$discrete :", omxQuotes(expectation@discrete), '\n')
	}
	if (length(expectation@between)) {
		cat("$between :", omxQuotes(expectation@between), fill=TRUE)
	}
	invisible(expectation)
}

setMethod("print", "MxExpectationRAM", function(x,...) {
	displayMxExpectationRAM(x)
})

setMethod("show", "MxExpectationRAM", function(object) {
	displayMxExpectationRAM(object)
})


#------------------------------------------------------------------------------
setMethod("genericGenerateData", signature("MxExpectationRAM"),
	function(.Object, model, nrows, subname, empirical, returnModel, use.miss,
		 .backend, nrowsProportion, silent)
	{
	  fellner <- length(model$expectation$between)
	  if (!fellner) {
	    return(generateNormalData(model, nrows, subname, empirical, returnModel, use.miss,
				      .backend, nrowsProportion, silent))
	  } else {
	    if (!use.miss) {
	      stop("use.miss=FALSE is not implemented for relational models")
	    }
	    if (length(nrows) || length(nrowsProportion)) {
	      stop("Specification of the number of rows is not supported for relational models")
	    }
	    generateRelationalData(model, returnModel, .backend, subname, empirical)
	  }
	})
