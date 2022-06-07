/*
 *  Copyright 2021-2021 by the individuals mentioned in the source code history
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef u_PENALTY_H_
#define u_PENALTY_H_

class omxMatrix;

// the new definitions of the penalty functions
// differs from the previous one as we will need different
// tuning parameters. To account for this, I created a new class
class SmoothPenalty{
  struct hp {
    omxMatrix *m;
    int r, c;
  };
  std::vector<hp> hpCache;
protected:
  S4 robj;
  omxMatrix *matrix;
  IntegerVector params;
  NumericVector epsilon;
  NumericVector scale;
  double smoothing; // this is the new tuning paramter which 
  // will be used to smooth the penalty function
  void copyFrom(const SmoothPenalty *pen);
  
public:
  SmoothPenalty(S4 _obj, omxMatrix *_mat);
  virtual ~SmoothPenalty();
  double penaltyStrength(double absPar, int px) const;
  int countNumZero(FitContext *fc) const;
  virtual void compute(int ffcompute, FitContext *fc)=0;
  const char *name() const;
  virtual std::unique_ptr<SmoothPenalty> clone(omxMatrix *mat) const = 0;
  double getValue() const;
  double getHP(FitContext *fc, int xx); 
  
};


class SmoothLassoPenalty : public SmoothPenalty {
  typedef SmoothPenalty super;
public:
  SmoothLassoPenalty(S4 _obj, omxMatrix *_mat) : SmoothPenalty(_obj, _mat) {}
  virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<SmoothPenalty> clone(omxMatrix *mat) const override;
};

// The ridge penalty is already smooth. However, because the implementation
// differs from that used below, I will also create a separate ridge function
class SmoothRidgePenalty : public SmoothPenalty {
  typedef SmoothPenalty super;
public:
  RidgePenalty(S4 _obj, omxMatrix *_mat) : SmoothPenalty(_obj, _mat) {}
  virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<SmoothPenalty> clone(omxMatrix *mat) const override;
};

class SmoothElasticNetPenalty : public SmoothPenalty {
  typedef Penalty super;
public:
  SmoothElasticNetPenalty(S4 _obj, omxMatrix *_mat) : SmoothPenalty(_obj, _mat) {}
  virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<SmoothPenalty> clone(omxMatrix *mat) const override;
};


// The following are the old penalty functions:
class Penalty {
  struct hp {
    omxMatrix *m;
    int r, c;
  };
  std::vector<hp> hpCache;
protected:
  S4 robj;
	omxMatrix *matrix;
  IntegerVector params;
  NumericVector epsilon;
  NumericVector scale;
  double smoothProportion;
  void copyFrom(const Penalty *pen);

 public:
  Penalty(S4 _obj, omxMatrix *_mat);
  virtual ~Penalty();
  double penaltyStrength(double absPar, int px) const;
  int countNumZero(FitContext *fc) const;
	virtual void compute(int ffcompute, FitContext *fc)=0;
	const char *name() const;
  virtual std::unique_ptr<Penalty> clone(omxMatrix *mat) const = 0;
  double getValue() const;
  double getHP(FitContext *fc, int xx);
};

class LassoPenalty : public Penalty {
  typedef Penalty super;
public:
  LassoPenalty(S4 _obj, omxMatrix *_mat) : Penalty(_obj, _mat) {}
	virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<Penalty> clone(omxMatrix *mat) const override;
};

// The ridge penalty is always smooth
class RidgePenalty : public Penalty {
  typedef Penalty super;
public:
  RidgePenalty(S4 _obj, omxMatrix *_mat) : Penalty(_obj, _mat) {}
	virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<Penalty> clone(omxMatrix *mat) const override;
};

class ElasticNetPenalty : public Penalty {
  typedef Penalty super;
public:
  ElasticNetPenalty(S4 _obj, omxMatrix *_mat) : Penalty(_obj, _mat) {}
	virtual void compute(int ffcompute, FitContext *fc) override;
  virtual std::unique_ptr<Penalty> clone(omxMatrix *mat) const override;
};

#endif
