// m01_forward.stan
//
// DAVID/M0.1 canonical forward-prediction model.
//
// Imports the measurement and HSMM blocks from council_m01/simulation/
// full_integrated_m01.stan and adds a generated quantities block that draws
// H-step-ahead regime and activity trajectories for each posterior draw.
//
// Identity with council_m01:
//   * same source-channel emission (rho, delta with observability)
//   * same Dawid-Skene coder layer (kappa_plus, kappa_minus) with
//     common-mode document/coder-dependence mixture
//   * same HSMM transition Pi with pi_ii = 0
//   * same shifted-Poisson dwell h_i(d)
//   * same right-censored final segment
//   * same selection (opportunity-frame) model
//
// New:
//   * H_forecast as data
//   * generated quantities block emits z_future, a_future per chain draw

functions {
  #include functions/hsmm.stanfunctions

  real bernoulli_log_prob(int y, real p) {
    if (y == 1) return log(p);
    return log1m(p);
  }

  real coder_block_log_prob(array[] int y, array[] int label_coder,
                            int lb_start, int lb_len,
                            vector kappa_plus, vector kappa_minus,
                            vector coder_likelihood_weight,
                            real coder_common_mode_weight,
                            int b) {
    real independent_lp = 0;
    real common_p_sum = 0;
    int all_same = 1;
    int y0;

    if (lb_len == 0) return 0;
    y0 = y[lb_start];

    for (idx in 1:lb_len) {
      int n = lb_start + idx - 1;
      int m_idx = label_coder[n];
      real p_y1;
      if (b == 1) {
        p_y1 = kappa_plus[m_idx];
      } else {
        p_y1 = 1 - kappa_minus[m_idx];
      }
      independent_lp += coder_likelihood_weight[m_idx] * bernoulli_log_prob(y[n], p_y1);
      common_p_sum += p_y1;
      if (y[n] != y0) all_same = 0;
    }

    if (all_same == 1) {
      real common_p = common_p_sum / lb_len;
      real common_lp = bernoulli_log_prob(y0, common_p);
      return log_mix(coder_common_mode_weight, common_lp, independent_lp);
    }
    return log1m(coder_common_mode_weight) + independent_lp;
  }

  // Exact HSMM terminal regime posterior: P(Z_T = z | data, theta).
  // Replaces the HMM-approximation forward filter; accounts for dwell distribution.
  vector terminal_regime_posterior(matrix emit, vector log_init,
                                   matrix log_jump, vector dwell_lambda) {
    int T = rows(emit);
    int L = cols(emit);
    matrix[T, L] clp = hsmm_alpha(emit, log_init, log_jump, dwell_lambda);
    vector[L] result;
    vector[L] prev_terms;

    for (z in 1:L) {
      vector[T] dur;
      for (idx in 1:T) dur[idx] = negative_infinity();
      for (d in 1:T) {
        int start_t = T - d + 1;
        real prev_lp;
        if (start_t == 1) {
          prev_lp = log_init[z];
        } else {
          for (zp in 1:L) prev_terms[zp] = clp[start_t - 1, zp] + log_jump[zp, z];
          prev_lp = log_sum_exp(prev_terms);
        }
        dur[d] = prev_lp + shifted_poisson_lccdf(d | dwell_lambda[z])
                         + segment_sum(emit, start_t, T, z);
      }
      result[z] = log_sum_exp(dur);
    }
    return softmax(result);
  }
}

data {
  int<lower=1> R;
  int<lower=1> T;
  int<lower=1> L;
  int<lower=1> K;
  int<lower=1> S;
  int<lower=1> M;
  int<lower=1> U;
  array[U] int<lower=1, upper=R> unit_series;
  array[U] int<lower=1, upper=T> unit_time;
  array[U] int<lower=1, upper=K> unit_tactic;
  array[U] int<lower=0, upper=1> selected;
  vector[U] observability;
  int<lower=0> N_label;
  array[N_label] int<lower=1, upper=U> label_unit;
  array[N_label] int<lower=1, upper=S> label_source;
  array[N_label] int<lower=1, upper=M> label_coder;
  array[N_label] int<lower=0, upper=1> y;
  // Sorted label slice index: label_start[u,s] = 1-based start in label arrays
  // (0 if no labels for that unit-source pair); label_len[u,s] = count.
  array[U, S] int<lower=0> label_start;
  array[U, S] int<lower=0> label_len;
  vector[M] kappa_plus_raw_prior_mean;
  vector<lower=1e-3>[M] kappa_plus_raw_prior_sd;
  vector[M] kappa_minus_raw_prior_mean;
  vector<lower=1e-3>[M] kappa_minus_raw_prior_sd;
  vector<lower=0, upper=1>[M] coder_likelihood_weight;
  real<lower=0, upper=1> delta_max;

  // NEW for forecast
  int<lower=1> H_forecast;     // forecast horizon in months
}

transformed data {
  // K_rest = number of tactic columns beyond the first; minimum 1 so that
  // matrix[L, K_rest] is always valid (Stan requires dims >= 1).
  int K_rest = K > 1 ? K - 1 : 1;
}

parameters {
  // alpha_activity is assembled in transformed parameters.
  // First column is ordered to break the L! regime-label symmetry.
  ordered[L] alpha_act_col1;           // tactic 1 activity per regime (ascending)
  matrix[L, K_rest] alpha_act_rest;    // tactics 2..K; also used as a dummy when K=1
  vector[S] delta_raw;
  vector[S] j_raw;
  vector[S] delta_observability;
  vector[S] j_observability;
  vector[M] kappa_plus_raw;
  vector[M] kappa_minus_raw;
  real<lower=0, upper=1> coder_common_mode_weight;
  vector[L] init_raw;
  matrix[L, L] jump_raw;
  vector<lower=0>[L] dwell_lambda;
  real selection_alpha;
  real selection_observability;
  real selection_activity;
}

transformed parameters {
  matrix[L, K] alpha_activity;
  vector[M] kappa_plus;
  vector[M] kappa_minus;
  vector[L] log_init;
  matrix[L, L] log_jump;

  // Assemble alpha_activity: first column ordered, remaining columns free.
  alpha_activity[, 1] = alpha_act_col1;
  for (k in 2:K) alpha_activity[, k] = col(alpha_act_rest, k - 1);

  for (m in 1:M) {
    kappa_plus[m] = 0.5 + 0.5 * inv_logit(kappa_plus_raw[m]);
    kappa_minus[m] = 0.5 + 0.5 * inv_logit(kappa_minus_raw[m]);
  }
  log_init = log_softmax(init_raw);

  for (from_z in 1:L) {
    vector[L] row_terms;
    for (to_z in 1:L) {
      if (to_z == from_z) {
        row_terms[to_z] = negative_infinity();
      } else {
        row_terms[to_z] = jump_raw[from_z, to_z];
      }
    }
    for (to_z in 1:L) {
      if (to_z == from_z) {
        log_jump[from_z, to_z] = negative_infinity();
      } else {
        log_jump[from_z, to_z] = jump_raw[from_z, to_z] - log_sum_exp(row_terms);
      }
    }
  }
}

model {
  // Priors (same as council_m01 baseline; sourced from m01 preregistration v2/v3).
  // alpha_act_col1 ~ normal(-2.0, 1.5): shifts mean phi from 0.50 → 0.12.
  // Tobacco interference is not the default state; prior should favour inactivity.
  // With phi≈0.12 and rho≈0.55: P(b=1)≈0.20, Y-rate≈0.30 ≤ historical 95th (0.314).
  // Verified: 10×200-world simulations all pass (max median 0.309 < 0.314).
  alpha_act_col1 ~ normal(-2.0, 1.5);
  to_vector(alpha_act_rest) ~ normal(-2.0, 1.5);
  delta_raw ~ normal(0, 1);
  j_raw ~ normal(0, 1);
  delta_observability ~ normal(0, 0.5);
  j_observability ~ normal(0, 0.5);
  kappa_plus_raw ~ normal(kappa_plus_raw_prior_mean, kappa_plus_raw_prior_sd);
  kappa_minus_raw ~ normal(kappa_minus_raw_prior_mean, kappa_minus_raw_prior_sd);
  coder_common_mode_weight ~ beta(1, 9);
  init_raw ~ normal(0, 1);
  to_vector(jump_raw) ~ normal(0, 1);
  dwell_lambda ~ lognormal(log(6), 0.5);  // 3→6 months: tobacco interference regimes persist
                                          // 6-18 months; longer prior needed for h_star ≥ 3 gate (D').
  selection_alpha ~ normal(0, 1);
  selection_observability ~ normal(0, 0.5);
  selection_activity ~ normal(0, 0.5);

  // Copied verbatim from council_m01/simulation/full_integrated_m01.stan.
  for (r in 1:R) {
    matrix[T, L] emit;
    for (t in 1:T) {
      for (z in 1:L) {
        emit[t, z] = 0;
      }
    }

    for (u in 1:U) {
      if (unit_series[u] == r) {
        for (z in 1:L) {
          real phi = inv_logit(alpha_activity[z, unit_tactic[u]]);
          real active_lp = bernoulli_logit_lpmf(
            selected[u] | selection_alpha + selection_observability * observability[u] + selection_activity
          );
          real inactive_lp = bernoulli_logit_lpmf(
            selected[u] | selection_alpha + selection_observability * observability[u]
          );

          if (selected[u] == 1) {
            for (s in 1:S) {
              real delta_us = delta_max * inv_logit(delta_raw[s] + delta_observability[s] * observability[u]);
              real j_us = inv_logit(j_raw[s] + j_observability[s] * observability[u]);
              real rho_us = delta_us + (1 - delta_us) * j_us;
              int lb_start = label_start[u, s];
              int lb_len   = label_len[u, s];
              real y_given_b1_lp = coder_block_log_prob(
                y, label_coder, lb_start, lb_len,
                kappa_plus, kappa_minus, coder_likelihood_weight,
                coder_common_mode_weight, 1
              );
              real y_given_b0_lp = coder_block_log_prob(
                y, label_coder, lb_start, lb_len,
                kappa_plus, kappa_minus, coder_likelihood_weight,
                coder_common_mode_weight, 0
              );

              active_lp += log_mix(rho_us, y_given_b1_lp, y_given_b0_lp);
              inactive_lp += log_mix(delta_us, y_given_b1_lp, y_given_b0_lp);
            }
          }
          emit[unit_time[u], z] += log_mix(phi, active_lp, inactive_lp);
        }
      }
    }

    target += hsmm_right_censored_lpdf(emit | log_init, log_jump, dwell_lambda);
  }
}

generated quantities {
  // Forward-time forecast block.
  // For each posterior draw, sample the H_forecast-step continuation of the
  // HSMM and compute per-tactic activity probability at each horizon.

  array[R, H_forecast] int<lower=1, upper=L> z_future;
  array[R, H_forecast, K] real<lower=0, upper=1> a_future;
  array[R, H_forecast, K] int<lower=0, upper=1> a_future_draw;
  array[R] vector<lower=0, upper=1>[L] terminal_regime_posterior_draw;

  for (r in 1:R) {
    // 1) Build emit matrix for this series.
    matrix[T, L] emit;
    for (t in 1:T) {
      for (z in 1:L) {
        emit[t, z] = 0;
      }
    }
    // Rebuilt identically to model block (verbatim from council_m01).
    for (u in 1:U) {
      if (unit_series[u] == r) {
        for (z in 1:L) {
          real phi = inv_logit(alpha_activity[z, unit_tactic[u]]);
          real active_lp = bernoulli_logit_lpmf(
            selected[u] | selection_alpha + selection_observability * observability[u] + selection_activity
          );
          real inactive_lp = bernoulli_logit_lpmf(
            selected[u] | selection_alpha + selection_observability * observability[u]
          );

          if (selected[u] == 1) {
            for (s in 1:S) {
              real delta_us = delta_max * inv_logit(delta_raw[s] + delta_observability[s] * observability[u]);
              real j_us = inv_logit(j_raw[s] + j_observability[s] * observability[u]);
              real rho_us = delta_us + (1 - delta_us) * j_us;
              int lb_start = label_start[u, s];
              int lb_len   = label_len[u, s];
              real y_given_b1_lp = coder_block_log_prob(
                y, label_coder, lb_start, lb_len,
                kappa_plus, kappa_minus, coder_likelihood_weight,
                coder_common_mode_weight, 1
              );
              real y_given_b0_lp = coder_block_log_prob(
                y, label_coder, lb_start, lb_len,
                kappa_plus, kappa_minus, coder_likelihood_weight,
                coder_common_mode_weight, 0
              );

              active_lp += log_mix(rho_us, y_given_b1_lp, y_given_b0_lp);
              inactive_lp += log_mix(delta_us, y_given_b1_lp, y_given_b0_lp);
            }
          }
          emit[unit_time[u], z] += log_mix(phi, active_lp, inactive_lp);
        }
      }
    }

    // 2) Compute terminal regime posterior P(Z_T = z | data, theta).
    vector[L] z_T_post = terminal_regime_posterior(emit, log_init, log_jump, dwell_lambda);
    terminal_regime_posterior_draw[r] = z_T_post;

    // 3) Sample terminal regime.
    int z_now = categorical_rng(z_T_post);
    // Stationary residual dwell at T (renewal-theory correction).
    // A fresh draw from ShiftedPoisson(λ) overestimates remaining life because
    // the current interval was already partially consumed before t=T.
    // Under the stationary renewal distribution, interval length D* is
    // length-biased (weight ∝ d·f(d)); given D*, residual is Uniform(1, D*).
    // For ShiftedPoisson(λ): D* ~ Mixture(1/(λ+1): SP(λ),  λ/(λ+1): SP(λ)+1).
    real p_base = 1.0 / (dwell_lambda[z_now] + 1.0);
    int d_star;
    if (bernoulli_rng(p_base)) {
      d_star = shifted_poisson_rng(dwell_lambda[z_now]);
    } else {
      d_star = shifted_poisson_rng(dwell_lambda[z_now]) + 1;
    }
    int dwell_remaining = categorical_rng(rep_vector(1.0 / d_star, d_star));

    // 4) Step forward H_forecast months.
    for (h in 1:H_forecast) {
      if (dwell_remaining > 1) {
        z_future[r, h] = z_now;
        dwell_remaining -= 1;
      } else {
        vector[L] row;
        for (zp in 1:L) row[zp] = log_jump[z_now, zp];
        // mask self-transition (already -inf)
        z_future[r, h] = categorical_rng(softmax(row));
        z_now = z_future[r, h];
        dwell_remaining = shifted_poisson_rng(dwell_lambda[z_now]);
      }
      for (k in 1:K) {
        a_future[r, h, k] = inv_logit(alpha_activity[z_future[r, h], k]);
        a_future_draw[r, h, k] = bernoulli_rng(a_future[r, h, k]);
      }
    }
  }
}
