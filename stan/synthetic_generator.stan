// synthetic_generator.stan
//
// DAVID/M0.1 twin-parity prior generator (tracker B-4).
//
// Draws one sample from EVERY prior of m01_forward.stan per iteration, using
// the generated quantities block under fixed_param sampling. The Python
// generative twin (david/simulator/synthetic_world.py) must produce
// statistically indistinguishable prior samples; tests/test_twin_parity.py
// compares the two with per-parameter KS and moment tests.
//
// Every prior below MUST stay in lockstep with the model block of
// m01_forward.stan and with synthetic_world.sample_world:
//   * Pi via row-softmax of N(0,1) jump_raw with the diagonal masked to -inf
//   * alpha_act_col1 via order statistics of L iid N(-2.0, 1.5)
//   * dwell_lambda ~ lognormal(log(6), 0.5); dwell duration = 1 + Poisson(lambda)
//   * delta = delta_max * inv_logit(delta_raw + delta_observability * O)
//     j     = inv_logit(j_raw + j_observability * O)
//     rho   = delta + (1 - delta) * j
//   * kappa = 0.5 + 0.5 * inv_logit(N(1, 0.5))
//   * init via softmax of N(0,1)
//
// If you edit a prior in m01_forward.stan or synthetic_world.py, edit it here
// in the same change set — the twin-parity battery fails closed on drift.

functions {
  #include functions/hsmm.stanfunctions
}

data {
  int<lower=2> L;
  int<lower=1> K;
  int<lower=1> S;
  int<lower=1> M;
  real<lower=0, upper=1> delta_max;
  int<lower=1> G;        // observability grid size for the link comparison
  vector[G] obs_grid;    // pre-registered grid points in [0, 1]
}

transformed data {
  // Mirrors m01_forward.stan: alpha_activity column 1 is ordered; the
  // remaining K-1 columns are free. K_rest >= 1 keeps dims valid when K = 1.
  int K_rest = K > 1 ? K - 1 : 1;
}

generated quantities {
  // Raw parameters, one prior draw per iteration.
  vector[L] alpha_act_col1;
  matrix[L, K_rest] alpha_act_rest;
  vector[S] delta_raw;
  vector[S] j_raw;
  vector[S] delta_observability;
  vector[S] j_observability;
  vector[M] kappa_plus_raw;
  vector[M] kappa_minus_raw;
  vector[L] init_raw;
  matrix[L, L] jump_raw;
  vector[L] dwell_lambda;
  real selection_alpha;
  real selection_observability;
  real selection_activity;

  // Transformed parameters, mirroring m01_forward.stan transformed parameters.
  matrix[L, K] alpha_activity;
  vector[M] kappa_plus;
  vector[M] kappa_minus;
  vector[L] init;
  matrix[L, L] Pi;

  // Detection links evaluated on the observability grid (B-4 link parity).
  matrix[S, G] delta_link;
  matrix[S, G] j_link;
  matrix[S, G] rho_link;

  // Forward draws through the same mechanism synthetic_world uses at t = 0.
  int z0;
  int dwell_draw;
  int a_draw;

  // ---- raw draws (priors copied from the m01_forward.stan model block) ----
  {
    vector[L] col1_iid;
    for (l in 1:L) col1_iid[l] = normal_rng(-2.0, 1.5);
    // ordered[L] with iid normal increments of density == order statistics
    // of L iid N(-2.0, 1.5) draws; matches Python's np.sort of iid draws.
    alpha_act_col1 = sort_asc(col1_iid);
  }
  for (l in 1:L) {
    for (k in 1:K_rest) alpha_act_rest[l, k] = normal_rng(-2.0, 1.5);
  }
  for (s in 1:S) {
    delta_raw[s] = normal_rng(0, 1);
    j_raw[s] = normal_rng(0, 1);
    delta_observability[s] = normal_rng(0, 0.5);
    j_observability[s] = normal_rng(0, 0.5);
  }
  for (m in 1:M) {
    kappa_plus_raw[m] = normal_rng(1, 0.5);
    kappa_minus_raw[m] = normal_rng(1, 0.5);
  }
  for (l in 1:L) init_raw[l] = normal_rng(0, 1);
  for (i in 1:L) {
    for (j in 1:L) jump_raw[i, j] = normal_rng(0, 1);
  }
  for (l in 1:L) dwell_lambda[l] = lognormal_rng(log(6), 0.5);
  selection_alpha = normal_rng(0, 1);
  selection_observability = normal_rng(0, 0.5);
  selection_activity = normal_rng(0, 0.5);

  // ---- transforms (copied from m01_forward.stan transformed parameters) ----
  alpha_activity[, 1] = alpha_act_col1;
  for (k in 2:K) alpha_activity[, k] = col(alpha_act_rest, k - 1);

  for (m in 1:M) {
    kappa_plus[m] = 0.5 + 0.5 * inv_logit(kappa_plus_raw[m]);
    kappa_minus[m] = 0.5 + 0.5 * inv_logit(kappa_minus_raw[m]);
  }
  init = softmax(init_raw);

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
        Pi[from_z, to_z] = 0;
      } else {
        Pi[from_z, to_z] = exp(jump_raw[from_z, to_z] - log_sum_exp(row_terms));
      }
    }
  }

  // ---- detection links (same algebra as the m01_forward.stan likelihood) ----
  for (s in 1:S) {
    for (g in 1:G) {
      delta_link[s, g] = delta_max
        * inv_logit(delta_raw[s] + delta_observability[s] * obs_grid[g]);
      j_link[s, g] = inv_logit(j_raw[s] + j_observability[s] * obs_grid[g]);
      rho_link[s, g] = delta_link[s, g] + (1 - delta_link[s, g]) * j_link[s, g];
    }
  }

  // ---- forward draws mirroring synthetic_world.sample_world at t = 0 ----
  z0 = categorical_rng(init);
  dwell_draw = shifted_poisson_rng(dwell_lambda[z0]);
  a_draw = bernoulli_rng(inv_logit(alpha_activity[z0, 1]));
}
