/*
 *  Copyright 2021-2021 by the individuals mentioned in the source code history
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "omxDefines.h"
#include "Compute.h"
#include "penalty.h"
#include "EnableWarnings.h"

SmoothPenalty::SmoothPenalty(S4 _obj, omxMatrix *mat) : matrix(mat)
{
  robj = _obj;
  params = robj.slot("params");
  epsilon = robj.slot("epsilon");
  scale = robj.slot("scale");
  smoothing = as<double>(robj.slot("smoothing"));
}

SmoothPenalty::~SmoothPenalty() {}

const char *SmoothPenalty::name() const
{ return matrix->name(); }

int SmoothPenalty::countNumZero(FitContext *fc) const
{
  int count = 0;
  for (int px = 0; px < params.size(); ++px) {
    if (fabs(fc->est[params[px]] / scale[px % scale.size()]) <=
        epsilon[px % epsilon.size()]) ++count;
  }
  return count;
}

void SmoothPenalty::copyFrom(const SmoothPenalty *pen)
{
  params = pen->params;
  epsilon = pen->epsilon;
  scale = pen->scale;
  smoothing = pen->smoothing;
}

double SmoothPenalty::getValue() const { return matrix->data[0]; }

double SmoothPenalty::getHP(FitContext *fc, int xx)
{
  if (!hpCache.size()) {
    IntegerVector pv = robj.slot("hyperparameters");
    int numHP = pv.size() / 3;
    if (3*numHP != pv.size()) mxThrow("%s: hyperparameters specified incorrectly", name());
    for (int p1=0; p1 < numHP; ++p1) {
      omxState *state = fc->state;
      hpCache.emplace_back(hp{state->matrixList[pv[p1 * 3]],
                           pv[1 + p1 * 3], pv[2 + p1 * 3]});
    }
  }
  auto &hp1 = hpCache[xx];
  return omxMatrixElement(hp1.m, hp1.r, hp1.c);
}

std::unique_ptr<SmoothPenalty> SmoothLassoPenalty::clone(omxMatrix *mat) const
{
  auto pen = std::make_unique<SmoothLassoPenalty>(robj, mat);
  pen->copyFrom(this);
  return pen;
}

void SmoothLassoPenalty::compute(int want, FitContext *fc)
{
  double lambda = getHP(fc, 0);
  if (want & FF_COMPUTE_FIT) {
    double tmp = 0;
    for (int px = 0; px < params.size(); ++px) {
      tmp += std::sqrt(std::pow(
        fc->est[ params[px] ], 2
      ) + smoothing)  / scale[px % scale.size()];
      // sqrt(p^2 + e) is always differentiable if e > 0
    }
    matrix->data[0] = tmp * lambda;
  }
  if (want & FF_COMPUTE_GRADIENT) {
    for (int px = 0; px < params.size(); ++px) {
      double par = fabs(fc->est[ params[px] ] / scale[px % scale.size()]);
      fc->gradZ[ params[px] ] += 
        lambda * fc->est[ params[px] ] /( scale[px % scale.size()] *
        std::sqrt( std::pow(fc->est[ params[px] ],2) + smoothing )  );
    }
  }
}

std::unique_ptr<SmoothPenalty> SmoothRidgePenalty::clone(omxMatrix *mat) const
{
  auto pen = std::make_unique<SmoothRidgePenalty>(robj, mat);
  pen->copyFrom(this);
  return pen;
}

// ridge penalty is already smooth
void SmoothRidgePenalty::compute(int want, FitContext *fc)
{
  double lambda = getHP(fc, 0);
  if (want & FF_COMPUTE_FIT) {
    double tmp = 0;
    for (int px = 0; px < params.size(); ++px) {
      double p1 = fabs(fc->est[ params[px] ] / scale[px % scale.size()]);
      tmp += (fc->est[ params[px] ] / scale[px % scale.size()]) * 
        (fc->est[ params[px] ] / scale[px % scale.size()]); // the previous implementation
      // also used the square of the scale in the denominator. I am not sure if this
      // was intentionally, but I've followed this approach above
    }
    matrix->data[0] = tmp * lambda;
  }
  if (want & FF_COMPUTE_GRADIENT) {
    
    for (int px = 0; px < params.size(); ++px) {
      fc->gradZ[ params[px] ] += 2*lambda * (fc->est[ params[px] ] / scale[px % scale.size()]);
    }
  }
}

std::unique_ptr<SmoothPenalty> SmoothElasticNetPenalty::clone(omxMatrix *mat) const
{
  auto pen = std::make_unique<SmoothElasticNetPenalty>(robj, mat);
  pen->copyFrom(this);
  return pen;
}

void SmoothElasticNetPenalty::compute(int want, FitContext *fc)
{
  double alpha = getHP(fc, 0);
  double lambda = getHP(fc, 1);
  if (want & FF_COMPUTE_FIT) {
    double lasso = 0;
    double ridge = 0;
    for (int px = 0; px < params.size(); ++px) {
      
      lasso += std::sqrt(std::pow(
        fc->est[ params[px] ], 2
      ) + smoothing)  / scale[px % scale.size()];
      // sqrt(p^2 + e) is always differentiable if e > 0
      
      ridge += (fc->est[ params[px] ] / scale[px % scale.size()]) * 
        (fc->est[ params[px] ] / scale[px % scale.size()]); 
      
    }
    matrix->data[0] = lambda * ((1-alpha) * ridge + alpha * lasso);
  }
  if (want & FF_COMPUTE_GRADIENT) {
    for (int px = 0; px < params.size(); ++px) {
      double lassoGradient = alpha* lambda * (
        fc->est[ params[px] ] /
          ( scale[px % scale.size()] *
            std::sqrt( std::pow(fc->est[ params[px] ],2) + smoothing )));
      double ridgeGradient = (1-alpha) * 2 * lambda * (fc->est[ params[px] ] / scale[px % scale.size()]);
      fc->gradZ[ params[px] ] += lassoGradient + ridgeGradient;
    }
  }
}
