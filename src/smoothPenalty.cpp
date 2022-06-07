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

std::unique_ptr<Penalty> SmoothLassoPenalty::clone(omxMatrix *mat) const
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
      fc->gradZ[ params[px] ] += 
        lambda * fc->est[ params[px] ] /( scale[px % scale.size()] *
        std::sqrt( std::pow(fc->est[ params[px] ],2) + smoothing )  );
    }
  }
}

std::unique_ptr<Penalty> SmoothRidgePenalty::clone(omxMatrix *mat) const
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

std::unique_ptr<Penalty> SmoothElasticNetPenalty::clone(omxMatrix *mat) const
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
