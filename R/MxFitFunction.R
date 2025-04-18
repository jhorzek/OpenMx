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


#
# The virtual base class for all fit functions
#
##' MxBaseFitFunction
##'
##' The virtual base class for all fit functions. This is an internal class and should not be used directly.
##'
##' @aliases
##' $,MxBaseFitFunction-method
##' $<-,MxBaseFitFunction-method
##' @seealso
##' \link{mxFitFunctionAlgebra}, \link{mxFitFunctionML}, \link{mxFitFunctionMultigroup},
##' \link{mxFitFunctionR}, \link{mxFitFunctionWLS}, \link{mxFitFunctionRow},
##' \link{mxFitFunctionGREML}
##' @rdname MxBaseFitFunction-class
setClass(Class = "MxBaseFitFunction",
	 representation = representation(
		info = "list",
		dependencies = "integer",
		expectation = "integer",
		vector = "logical",
		rowDiagnostics = "logical",
    penalties = "MxOptionalCharOrNumber",
    applyPenalty = "logical",
		result = "matrix", "VIRTUAL"),
	 contains = "MxBaseNamed")

##' @title MxFitFunction
##' @name MxFitFunction-class
##'
##' @description
##' This is an internal class and should not be used directly.
##'
##' @aliases
##' MxFitFunction
##' MxFitFunction-class
##' @rdname MxFitFunction-class
setClassUnion("MxFitFunction", c("NULL", "MxBaseFitFunction"))

setGeneric("genericFitDependencies",
	function(.Object, flatModel, dependencies) {
	return(standardGeneric("genericFitDependencies"))
})

setGeneric("genericFitRename",
	function(.Object, oldname, newname) {
	return(standardGeneric("genericFitRename"))
})

setGeneric("genericFitInitialMatrix",
	function(.Object, flatModel) {
	return(standardGeneric("genericFitInitialMatrix"))
})

setGeneric("genericFitNewEntities",
	function(.Object) {
	return(standardGeneric("genericFitNewEntities"))
})


setGeneric("genericFitFunConvert",
	function(.Object, flatModel, model, labelsData, dependencies) {
	return(standardGeneric("genericFitFunConvert"))
})

setGeneric("generateReferenceModels",
	function(.Object, model, distribution, equateThresholds) {
	return(standardGeneric("generateReferenceModels"))
})

setMethod("generateReferenceModels", "MxBaseFitFunction",
	function(.Object, model, distribution) {
		msg <- paste("Don't know how to make ", omxQuotes(distribution),
			" distribution reference models for a model with a ",
			     class(.Object), " fit function.", sep="")
		stop(msg)
	})

setMethod("genericFitInitialMatrix", "MxBaseFitFunction",
	function(.Object, flatModel) {
		return(matrix(as.double(NA), 1, 1))
})

setMethod("genericFitInitialMatrix", "NULL",
	function(.Object, flatModel) {
		return(NULL)
})

setMethod("$", "MxBaseFitFunction", imxExtractSlot)

setReplaceMethod("$", "MxBaseFitFunction",
	function(x, name, value) {
        if(name == "result") {
            stop("You cannot set the result of an fit function.  Use mxRun() to populate the result, or mxEval() to compute it.")
        }
		return(imxReplaceSlot(x, name, value, check=TRUE))
	}
)

setMethod("names", "MxBaseFitFunction", slotNames)

##' Add dependencies
##'
##' If there is an expectation, then the fitfunction should always
##' depend on it. Hence, subclasses that implement this method must
##' ignore the passed-in dependencies and use "dependencies <-
##' callNextMethod()" instead.
##'
##' @param .Object fit function object
##' @param flatModel flat model that lives with .Object
##' @param dependencies accumulated dependency relationships

setMethod("genericFitDependencies", "MxBaseFitFunction",
	function(.Object, flatModel, dependencies) {
        name <- .Object@name
        modelname <- imxReverseIdentifier(flatModel, .Object@name)[[1]]
        expectName <- paste(modelname, "expectation", sep=".")
        if (!is.null(flatModel[[expectName]])) {
            dependencies <- imxAddDependency(expectName, .Object@name, dependencies)
        }
		return(dependencies)
})

setMethod("genericFitDependencies", "NULL",
	function(.Object, flatModel, dependencies) {
		return(dependencies)
})

setMethod("genericFitRename", "MxBaseFitFunction",
	function(.Object, oldname, newname) {
		return(.Object)
})

setMethod("genericFitRename", "NULL",
	function(.Object, oldname, newname) {
		return(NULL)
})

setMethod("genericFitNewEntities", "MxBaseFitFunction",
	function(.Object) {
		return(NULL)
})

setGeneric("genericFitConvertEntities",
	function(.Object, flatModel, namespace, labelsData) {
	return(standardGeneric("genericFitConvertEntities"))
})

setGeneric("genericFitAddEntities",
	function(.Object, job, flatJob, labelsData) {
	return(standardGeneric("genericFitAddEntities"))
})

setMethod("genericFitConvertEntities", "MxBaseFitFunction",
	function(.Object, flatModel, namespace, labelsData) {
		return(flatModel)
})

setMethod("genericFitConvertEntities", "NULL",
	function(.Object, flatModel, namespace, labelsData) {
		return(flatModel)
})

setMethod("genericFitAddEntities", "MxBaseFitFunction",
	function(.Object, job, flatJob, labelsData) {
		return(job)
})

setMethod("genericFitAddEntities", "NULL",
	function(.Object, job, flatJob, labelsData) {
		return(job)
})

fitFunctionAddEntities <- function(model, flatModel, labelsData) {

	fitfunctions <- flatModel@fitfunctions

	if (length(fitfunctions) == 0) {
		return(model)
	}

	for(i in 1:length(fitfunctions)) {
		model <- genericFitAddEntities(fitfunctions[[i]], model, flatModel, labelsData)
	}

	return(model)
}

fitFunctionModifyEntities <- function(flatModel, namespace, labelsData) {

	fitfunctions <- flatModel@fitfunctions

	if (length(fitfunctions) == 0) {
		return(flatModel)
	}

	for(i in 1:length(fitfunctions)) {
		flatModel <- genericFitConvertEntities(fitfunctions[[i]], flatModel, namespace, labelsData)
	}

	return(flatModel)
}

convertFitFunctions <- function(flatModel, model, labelsData, dependencies) {
	retval <- lapply(flatModel@fitfunctions, genericFitFunConvert,
		flatModel, model, labelsData, dependencies)
	return(retval)
}

setMethod("initialize", signature("MxBaseFitFunction"),
	function(.Object, ...) {
		.Object@name <- 'fitfunction'
    .Object@vector <- FALSE
    .Object@applyPenalty <- TRUE
    .Object@penalties <- c()
		.Object
	})

setMethod("qualifyNames", signature("MxBaseFitFunction"),
	function(.Object, modelname, namespace) {
		.Object@name <- imxIdentifier(modelname, .Object@name)
    .Object@penalties <- imxConvertIdentifier(.Object@penalties, modelname, namespace)
		.Object
  })

setMethod("genericFitFunConvert", signature("MxBaseFitFunction"),
          function(.Object, flatModel, model, labelsData, dependencies) {
            if (length(.Object@penalties) == 0) {
              .Object@penalties <- imxIdentifier(model@name, names(model@penalties))
            }
            name <- .Object@name
            if (.Object@vector && .Object@applyPenalty && length(.Object@penalties) > 0) {
              stop(paste("Model", omxQuotes(model@name), "has penalties",
                         omxQuotes(.Object@penalties), "but it is not clear",
                         "how to apply these penalties because",
                         omxQuotes(paste0(name, "$vector")), "is TRUE!\n  (The mind boggles)"))

            }
            .Object@penalties <- imxLocateIndex(flatModel, .Object@penalties,
                                                paste0(name, "@penalties"))
            .Object
          })

#Maybe export this as an imx* function?:
AllMLFitFunction <- function(model,submodels=TRUE){
	out <- TRUE
	if(length(model$fitfunction)){
		out <- out && (is(model$fitfunction,"MxFitFunctionML") || is(model$fitfunction,"MxFitFunctionMultigroup"))
	}
	else{
		return(FALSE) #<--It doesn't have an ML fitfunction, because it has no fitfunction at all.
	}
	if(submodels && length(model$submodels) > 0){
		out <- out && all(sapply(model@submodels, function(x){
			return(length(x$fitfunction) && (is(x$fitfunction,"MxFitFunctionML") || is(x$fitfunction,"MxFitFunctionMultigroup")))
				}))
	}
	return(out)
}
#^^^This function assumes that OpenMx will raise a runtime error in the case of every fitfunction being multigroup.
