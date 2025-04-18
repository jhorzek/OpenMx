#
#   Copyright 2007-2020 by the individuals mentioned in the source code history
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


#------------------------------------------------------------------------------
# Author: Michael D. Hunter
# Date: 2015-03-11
# Filename: MxMI.R
# Purpose: Write a function for model modification
#------------------------------------------------------------------------------

# TODO
# Get the submodel and multigroup model MI working
# Write automatic model search algorithm: 1. forward entry (start with minimal base model and add free parameters until no longer merited)



mxMI <- function(model, matrices=NA, full=TRUE){
	warnModelCreatedByOldVersion(model)
	if(single.na(matrices)){
		matrices <- names(model$matrices) #names of them rather
		if (is(model$expectation, "MxExpectationRAM")) {
			matrices <- setdiff(matrices, model$expectation$F)
		}
	}
	if(imxHasWLS(model)){stop("modification indices not implemented for WLS fitfunction")}
	param <- omxGetParameters(model)
	param.names <- names(param)
	gmodel <- omxSetParameters(model, free=FALSE, labels=param.names)
	#same as original model but with all parameters fixed
	mi.r <- NULL
	mi.f <- NULL
	epc.f <- NULL
	a.names <- NULL
	new.models <- list()
	for(amat in matrices){
		matObj <- model[[amat]]
		freemat <- matObj$free
		sym.sel <- upper.tri(freemat, diag=TRUE)
		notSymDiag <- !(is(gmodel[[amat]])[1] %in% c("DiagMatrix", "SymmMatrix"))
		for(i in 1:length(freemat)){
			# only walk the lower triangle of Diag and Symm matrices.
			if(freemat[i]==FALSE && ( notSymDiag || sym.sel[i]==TRUE )){
				tmpLab <- gmodel[[amat]]$labels[i]
				plusOneParamModel <- model
				if(length(tmpLab) > 0 && !is.na(tmpLab)){
					gmodel <- omxSetParameters(gmodel, labels=tmpLab, free=TRUE)
					plusOneParamModel <- omxSetParameters(plusOneParamModel, labels=tmpLab, free=TRUE)
				} else{
					#free single parameter of model that was fixed in original model
					gmodel[[amat]]$free[i] <- TRUE
					# create a new model with all the free params of the orig
					# PLUS the new one under consideration
					plusOneParamModel[[amat]]$free[i] <- TRUE
				}
				# specific adjustments for zero matrices
				if(is(gmodel[[amat]])[1] %in% c("ZeroMatrix")){
					cop <- gmodel[[amat]]
					newSingleParamMat <- mxMatrix("Full", nrow=nrow(cop),
						ncol=ncol(cop), values=cop$values, free=cop$free,
						labels=cop$labels, name=cop$name, lbound=cop$lbound,
						ubound=cop$ubound, dimnames=dimnames(cop))
					bop <- plusOneParamModel[[amat]]
					newPlusOneParamMat <- mxMatrix("Full", nrow=nrow(bop),
						ncol=ncol(bop), values=bop$values, free=bop$free,
						labels=bop$labels, name=bop$name, lbound=bop$lbound,
						ubound=bop$ubound, dimnames=dimnames(bop))
				# specific adjustments for symmetric and diagonal matrices
				} else if(is(gmodel[[amat]])[1] %in% c("DiagMatrix", "SymmMatrix")){
					cop <- gmodel[[amat]]
					newSingleParamMat <- mxMatrix("Symm", nrow=nrow(cop),
						ncol=ncol(cop), values=cop$values,
						free=(cop$free | t(cop$free)), labels=cop$labels,
						name=cop$name, lbound=cop$lbound, ubound=cop$ubound,
						dimnames=dimnames(cop))
					bop <- plusOneParamModel[[amat]]
					newPlusOneParamMat <- mxMatrix("Symm", nrow=nrow(bop),
						ncol=ncol(bop), values=bop$values,
						free=(bop$free | t(bop$free)), labels=bop$labels,
						name=bop$name, lbound=bop$lbound, ubound=bop$ubound,
						dimnames=dimnames(bop))
				# no adjustments. just fill in
				} else {
					newSingleParamMat <- gmodel[[amat]]
					newPlusOneParamMat <- plusOneParamModel[[amat]]
				}
				gmodel[[amat]] <- newSingleParamMat
				plusOneParamModel[[amat]] <- newPlusOneParamMat
				
				# The custom compute plan.  Only do derivatives
				custom.compute <- mxComputeSequence(list(mxComputeNumericDeriv(checkGradient=FALSE),
									 mxComputeReportDeriv()))
				
				# Create and run the single-parameter model for the LISREL-type/partial/[lower bound] MI
				gmodel <- mxModel(gmodel, custom.compute)
				grun <- try(mxRun(gmodel, silent = FALSE, suppressWarnings = FALSE, unsafe=TRUE)) #suppress Warnings =TRUE
				if (is(grun, "try-error")) {
					# oops, not happy about that parameter (e.g., diag of RAM's A)
					gmodel <- omxSetParameters(gmodel, labels=names(omxGetParameters(gmodel)), free=FALSE)
					next
				}
				
				# restricted MI
				grad <- grun$output$gradient #get gradient
				hess <- grun$output$hessian #get Hessian
				modind <- 0.5*grad^2/hess #use 0.5*g^2/k
				
				if(full==TRUE){
					custom.compute.smart <- mxComputeSequence(list(
					    mxComputeNumericDeriv(knownHessian=model$output$hessian, checkGradient=FALSE),
					    mxComputeReportDeriv()))
					# Create and run the all-plus-one-parameter model for the Mplus-type/full MI
					plusOneParamRun <- mxRun(mxModel(plusOneParamModel, custom.compute.smart), silent = FALSE, suppressWarnings = FALSE, unsafe=TRUE)
					
					# full MI
					grad.full <- plusOneParamRun$output$gradient
					#grad.full[is.na(grad.full)] <- 0
					grad.full[which(names(grad.full)!=names(omxGetParameters(gmodel)))] <- 0
					hess.full <- plusOneParamRun$output$hessian
					solve_hess_full <- try(solve(hess.full),silent=F)
					if(is(solve_hess_full,"try-error") && length(grep("computationally singular",solve_hess_full[1]))){
						solve_hess_full <- chol2inv(chol(hess.full))
					}
					modind.full <- 0.5*t(matrix(grad.full)) %*% solve_hess_full %*% matrix(grad.full)
					if(sum(grad.full != 0) == 1){
						exppar.full <- - modind.full/grad.full[grad.full != 0]
					} else {stop("Something strange in the neighborhood.\nFound a one-parameter model with more than one parameter.\nPost this to the OpenMx forums.")}
					# Note: the above should be safe, but assumes only one grad.full is not zero
				} else {
					modind.full <- NULL
					exppar.full <- NULL
				}
				
				n.names <- names(omxGetParameters(grun))
				if(length(modind) > 0){
					a.names <- c(a.names, n.names)
					mi.r <- c(mi.r, modind)
					mi.f <- c(mi.f, modind.full)
					epc.f <- c(epc.f, exppar.full)
					new.models <- c(new.models, plusOneParamModel)
				}
				gmodel <- omxSetParameters(gmodel, labels=names(omxGetParameters(gmodel)), free=FALSE)
			}
		}
		names(mi.r) <- a.names
		if(full==TRUE) {names(mi.f) <- a.names; names(epc.f) <- a.names}
		names(new.models) <- a.names
	}
	# not yet tested
	if(length(model$submodels) > 0){
		for(asubmodel in names(model$submodels)){
			ret <- c(ret, mxMI(asubmodel)) #probably won't work.
		}
	}
	if(is.null(mi.f)) mi.f <- rep(NA, length(mi.r))
	if(is.null(epc.f)) epc.f <- rep(NA, length(mi.r))
	retList <- list2DF(list(MI=mi.r, MI.Full=mi.f, plusOneParamModels=new.models, EPC=epc.f), nrow=length(mi.r))
	return(retList)
}


