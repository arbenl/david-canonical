// coder_calibration.stan
//
// Stand-alone Dawid-Skene model with gold-standard anchoring. Calibrates
// kappa_plus[m] and kappa_minus[m] for each coder m (human or LLM) using
// only items whose true label B_e is adjudicated.
//
// Inputs:
//   E_gold: items with gold B label (anchors)
//   M     : number of coders (humans + LLM seats)
//   Y[e, m]: 1 if coder m labels item e as 1
//   B_gold[e]: 1 if gold item e is truly 1
//
// Outputs:
//   kappa_plus[m]: P(Y=1 | B=1) per coder
//   kappa_minus[m]: P(Y=0 | B=0) per coder
//
// Un-gold coder consensus is intentionally excluded from the likelihood.
// Until a dependence-aware coder likelihood is audited, un-gold LLM agreement
// must not narrow calibration posteriors.

data {
  int<lower=0> E_gold;
  int<lower=1> M;
  array[E_gold, M] int<lower=0, upper=1> Y_gold;
  array[E_gold] int<lower=0, upper=1> B_gold;
}

parameters {
  vector[M] kappa_plus_raw;
  vector[M] kappa_minus_raw;
}

transformed parameters {
  vector<lower=0.5, upper=1.0>[M] kappa_plus;
  vector<lower=0.5, upper=1.0>[M] kappa_minus;
  for (m in 1:M) {
    kappa_plus[m] = 0.5 + 0.5 * inv_logit(kappa_plus_raw[m]);
    kappa_minus[m] = 0.5 + 0.5 * inv_logit(kappa_minus_raw[m]);
  }
}

model {
  // Priors
  kappa_plus_raw ~ normal(1, 0.5);
  kappa_minus_raw ~ normal(1, 0.5);

  // Gold items: fully observed B
  for (e in 1:E_gold) {
    for (m in 1:M) {
      real p;
      if (B_gold[e] == 1) {
        p = kappa_plus[m];
      } else {
        p = 1 - kappa_minus[m];
      }
      Y_gold[e, m] ~ bernoulli(p);
    }
  }
}
