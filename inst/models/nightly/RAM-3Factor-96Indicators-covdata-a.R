#
#   Copyright 2007-2018 by the individuals mentioned in the source code history
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


# ---------------------------------------------------------------------
# Program: RAM-3Factor-12Indicators.R
#  Author: Steven M. Boker
#    Date: Fri Jul 30 13:45:12 EDT 2010
#
# This program is a factor model using standard RAM.
#
# ---------------------------------------------------------------------
# Revision History
#    -- Fri Jul 30 13:45:12 EDT 2010
#      Created RAM-3Factor-12Indicators.R.
#
# ---------------------------------------------------------------------

# ----------------------------------
# Read libraries and set options.

library(OpenMx)

options(width=100)
set.seed(10)

# ---------------------------------------------------------------------
# Data for factor model.

numberSubjects <- 1000
numberFactors <- 3
numberIndPerFactor <- 32
numberIndicators <- numberIndPerFactor*numberFactors # must be a multiple of numberFactors

XMatrix <- matrix(rnorm(numberSubjects*numberFactors, mean=0, sd=1), numberSubjects, numberFactors)

tLoadings <- c(1, seq(.5, .9, length.out=(numberIndPerFactor-1)), rep(0, numberIndPerFactor*2),
  rep(0, numberIndPerFactor*1), 1, seq(.5, .9, length.out=(numberIndPerFactor-1)), rep(0, numberIndPerFactor*1),
  rep(0, numberIndPerFactor*2), 1, seq(.5, .9, length.out=(numberIndPerFactor-1)))
BMatrix <- matrix(tLoadings, numberFactors, numberIndicators, byrow=TRUE)
UMatrix <- matrix(rnorm(numberSubjects*numberIndicators, mean=0, sd=1), numberSubjects, numberIndicators)
YMatrix <- XMatrix %*% BMatrix + UMatrix

cor(XMatrix)

dimnames(YMatrix) <- list(NULL, paste("X", 1:numberIndicators, sep=""))

round(cor(YMatrix), 3)
round(cov(YMatrix), 3)

indicators <- paste("X", 1:numberIndicators, sep="")
totalVars <- numberIndicators + numberFactors

# ----------------------------------
# Build an orthogonal simple structure factor model

latents <- paste("F", 1:numberFactors, sep="")

uniqueLabels <- paste("U_", indicators, sep="")
meanLabels <- paste("M_", latents, sep="")
factorVarLabels <- paste("Var_", latents, sep="")

latents1 <- latents[1]
indicators1 <- indicators[1:numberIndPerFactor]
loadingLabels1 <- paste("b_F1", indicators[1:numberIndPerFactor], sep="") 
latents2 <- latents[2]
indicators2 <- indicators[numberIndPerFactor+(1:numberIndPerFactor)]
loadingLabels2 <- paste("b_F2", indicators[numberIndPerFactor+(1:numberIndPerFactor)], sep="") 
latents3 <- latents[3]
indicators3 <- indicators[(2*numberIndPerFactor)+(1:numberIndPerFactor)]
loadingLabels3 <- paste("b_F3", indicators[(2*numberIndPerFactor)+(1:numberIndPerFactor)], sep="") 

threeFactorOrthogonal <- mxModel("threeFactorOrthogonal",
    type="RAM",
    manifestVars=c(indicators),
    latentVars=c(latents,"dummy1"),
    mxPath(from=latents1, to=indicators1, 
           arrows=1, connect="all.pairs",
           free=TRUE, values=.2, lbound = 0, ubound = 1, 
           labels=loadingLabels1),
    mxPath(from=latents2, to=indicators2, 
           arrows=1, connect="all.pairs",
           free=TRUE, values=.2, lbound = 0, ubound = 1,
           labels=loadingLabels2),
    mxPath(from=latents3, to=indicators3, 
           arrows=1, connect="all.pairs",
           free=TRUE, values=.2, lbound = 0, ubound = 1,
           labels=loadingLabels3),
    mxPath(from=latents1, to=indicators1[1], 
           arrows=1, 
           free=FALSE, values=1),
    mxPath(from=latents2, to=indicators2[1], 
           arrows=1, 
           free=FALSE, values=1),
    mxPath(from=latents3, to=indicators3[1], 
           arrows=1, 
           free=FALSE, values=1),
    mxPath(from=indicators, 
           arrows=2, 
           free=TRUE, values=.2, 
           labels=uniqueLabels),
    mxPath(from=latents,
           arrows=2, 
           free=TRUE, values=.8, 
           labels=factorVarLabels),
    mxPath(from="one", to=indicators, 
           arrows=1, free=FALSE, values=0),
    mxPath(from="one", to=c(latents), 
           arrows=1, free=TRUE, values=.1, 
           labels=meanLabels),
    mxData(observed=cov(YMatrix), means=apply(YMatrix, 2, mean), 
	numObs=nrow(YMatrix), type="cov")
    )

threeFactorOrthogonalOut <- mxRun(threeFactorOrthogonal)
# Cannot reproduce getting a better fit value of 104964.3
# set.seed(10)
# threeFactorOrthogonalOut <- mxTryHard(threeFactorOrthogonalOut,fit2beat=104964.4)
summary(threeFactorOrthogonalOut)
omxCheckCloseEnough(threeFactorOrthogonalOut$output$fit, 105106.956, .5)

# Compare with vs. without analytic gradients ####
mxOption(NULL,"Analytic gradients","Yes"); mxOption(NULL,"Analytic RAM derivatives","Yes")
foo <- mxRun(threeFactorOrthogonal)
mxOption(NULL,"Analytic gradients","No"); mxOption(NULL,"Analytic RAM derivatives","No")
bar <- mxRun(threeFactorOrthogonal)
omxCheckCloseEnough(foo$output$fit, bar$output$fit, .5)
if(0){
	omxCheckTrue(summary(foo)$wallTime < summary(bar)$wallTime)
	omxCheckTrue(foo$output$iterations <= bar$output$iterations)
}
summary(foo)$wallTime; summary(bar)$wallTime
foo$output$iterations; bar$output$iterations
omxCheckTrue(foo$output$evaluations < bar$output$evaluations)
foo$output$evaluations; bar$output$evaluations

mxOption(reset=TRUE)
