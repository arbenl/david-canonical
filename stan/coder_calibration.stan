// coder_calibration.stan
//
// Stand-alone Dawid-Skene model with gold-standard anchoring. Calibrates
// kappa_plus[m] and kappa_minus[m] for each coder m (human or LLM) using
// items whose true label B_e is adjudicated.
//
// Inputs:
//   E_gold: items with gold B label (anchors)
//   E_un  : items with no gold label (use joint LLM agreement)
//   M     : number of coders (humans + LLM seats)
//   Y[e, m]: 1 if coder m labels item e as 1
//   B_gold[e]: 1 if gold item e is truly 1
//
// Outputs:
//   kappa_plus[m]: P(Y=1 | B=1) per coder
//   kappa_minus[m]: P(Y=0 | B=0) per coder
//   posterior B for ungolden items via latent indicator

data {
  int<lower=0> E_gold;
  int<lower=0> E_un;
  int<lower=1> M;
  array[E_gold, M] int<lower=0, upper=1> Y_gold;
  array[E_gold] int<lower=0, upper=1> B_gold;
  array[E_un, M] int<lower=0, upper=1> Y_un;
}

parameters {
  vector[M] kappa_plus_raw;
  vector[M] kappa_minus_raw;
  real<lower=0, upper=1> phi_un;       // prior prevalence among un-gold items
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
  phi_un ~ beta(2, 2);

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

  // Un-gold items: B is latent, marginalize
  for (e in 1:E_un) {
    real lp_one = log(phi_un);
    real lp_zero = log1m(phi_un);
    for (m in 1:M) {
      lp_one += bernoulli_lpmf(Y_un[e, m] | kappa_plus[m]);
      lp_zero += bernoulli_lpmf(Y_un[e, m] | 1 - kappa_minus[m]);
    }
    target += log_sum_exp(lp_one, lp_zero);
  }
}

generated quantities {
  vector[E_un] post_B_un;
  for (e in 1:E_un) {
    real lp_one = log(phi_un);
    real lp_zero = log1m(phi_un);
    for (m in 1:M) {
      lp_one += bernoulli_lpmf(Y_un[e, m] | kappa_plus[m]);
      lp_zero += bernoulli_lpmf(Y_un[e, m] | 1 - kappa_minus[m]);
    }
    post_B_un[e] = exp(lp_one - log_sum_exp(lp_one, lp_zero));
  }
}
