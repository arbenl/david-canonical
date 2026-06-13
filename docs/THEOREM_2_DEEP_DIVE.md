# Deep Dive: Theorem B′ — Bayes Error \& Asymptotic Information Scaling

> [!TIP]
> **Status:** Pedagogical deep-dive synchronized to the active theorem packet.
> The formal source of truth is `docs/thesis_mathematical_core.tex`.

This document provides a detailed, step-by-step pedagogical explanation of **Theorem B′ (Bayes Error and Asymptotic Information Scaling)**. It is designed to help you thoroughly understand the math, the proof, how it behaves in practice, and why it is resilient to real-world challenges.

---

## 1. The Core Conceptual Problem

In predicting target Tobacco Industry (TI) actions, we rely on noisy, imperfect data channels. If a data source (e.g. an LLM coder or an RSS feed) is barely better than a random coin toss, how much can we trust its decisions? And how much data ($N$) do we need to collect to get a precise estimate of the hidden state prevalence ($\phi$)?

Theorem B′ establishes the absolute information-theoretic bounds for such channels:
* **Theorem B′.1 (Bayes Error):** States the absolute limit of classification accuracy from a single observation.
* **Theorem B′.2 (Information Scaling):** Establishes that the information we gain collapses **quadratically** ($I^2$) as a source's informativeness vanishes.

---

## 2. Breaking Down the Parameters

| Symbol / Parameter | Domain | Operational Interpretation |
| :--- | :---: | :--- |
| **$A \in \{0, 1\}$** | Latent state space | True presence of a TI tactic (1 = Active, 0 = Inactive). |
| **$Y \in \{0, 1\}$** | Observed state space | Binary label emitted by a noisy source. |
| **$\phi \in [0, 1]$** | Parameter | True underlying prevalence: $P(A = 1)$. |
| **$I = |\rho - \delta|$** | Informativeness | The channel signal strength (Sensitivity minus False Positive rate). |
| **$\varepsilon^\star \in [0, 0.5]$** | Error bound | Minimum possible probability of classification error. |
| **$I_F(\phi)$** | Fisher Information | The information parameter $\phi$ carries about the latent state. |

---

## 3. Theorem B′.1: Single-Observation Bayes Error

### The Claim
Under equal prior odds ($P(A=1) = P(A=0) = 0.5$), any classifier mapping $Y \to A$ must suffer an error rate of at least:
$$\varepsilon^\star = \frac{1}{2}(1 - I)$$

### The Proof
For binary classification under equal priors, the minimum probability of error (the Bayes error) is determined by the **Total Variation (TV)** distance between the conditional distributions of $Y$:
$$\varepsilon^\star = \frac{1}{2} \left( 1 - \text{TV}(P_{Y \mid A=1}, P_{Y \mid A=0}) \right)$$

Since $Y$ is binary, $P_{Y \mid A=1} \sim \text{Bernoulli}(\rho)$ and $P_{Y \mid A=0} \sim \text{Bernoulli}(\delta)$. The Total Variation distance is:
$$\text{TV} = \frac{1}{2} \sum_{y \in \{0, 1\}} |P(Y=y \mid A=1) - P(Y=y \mid A=0)|$$
$$\text{TV} = \frac{1}{2} \left( |\rho - \delta| + |(1-\rho) - (1-\delta)| \right) = |\rho - \delta| = I$$

Substituting $\text{TV} = I$ into the Bayes error bound yields:
$$\varepsilon^\star = \frac{1}{2}(1 - I) \quad \blacksquare$$

* **Intuition:** If $I = 0$ (random coin flip), your minimum error is $50\%$ (pure guessing). If $I = 1$ (perfect channel), the minimum error is $0$.

---

## 4. Theorem B′.2: Asymptotic Information Scaling ($N \cdot I^2$)

### The Claim
For $N$ conditionally independent observations, the Fisher information $I_F(\phi)$ scales quadratically with informativeness $I$:
$$I_F(\phi) = N \frac{I^2}{p(1 - p)}$$
where $p = \delta + I\phi$ is the marginal probability of a positive observation.

### The Proof
The probability of a positive label is $p = P(Y=1) = \delta + I\phi$ (assuming $\rho > \delta$).
The log-likelihood function of a single observation $y \sim \text{Bernoulli}(p)$ is:
$$\ln f(y \mid \phi) = y \ln(p) + (1 - y) \ln(1 - p)$$

Taking the derivative w.r.t $\phi$ gives the score function:
$$\frac{\partial}{\partial \phi} \ln f(y \mid \phi) = I \frac{y - p}{p(1 - p)}$$

The Fisher information of a single observation is the variance of this score:
$$I_{F, 1}(\phi) = E\left[ \left( \frac{\partial}{\partial \phi} \ln f(Y \mid \phi) \right)^2 \right] = \frac{I^2}{p^2 (1 - p)^2} E[(Y - p)^2]$$

Since $E[(Y - p)^2] = \text{Var}(Y) = p(1-p)$:
$$I_{F, 1}(\phi) = \frac{I^2}{p(1 - p)}$$

Summing over $N$ independent observations yields the total Fisher information:
$$I_F(\phi) = N \frac{I^2}{p(1 - p)} \quad \blacksquare$$

---

## 5. The Asymptotic Implication: Quadratic Collapse

Because $p(1 - p) \le 0.25$, we establish the absolute bound:
$$I_F(\phi) \ge 4 N \cdot I^2$$

This means the information scales as **$N \cdot I^2$**. The variance of any prevalence estimator is bounded by the Cramér-Rao Lower Bound:
$$\text{Var}(\hat{\phi}) \ge \frac{1}{I_F(\phi)} \approx \frac{p(1 - p)}{N \cdot I^2}$$

To maintain a constant variance (estimation accuracy), the required sample size $N$ scales as:
$$N \propto \frac{1}{I^2}$$

* **The Challenge:** If a scraper's informativeness drops from $0.50$ to $0.05$ (a 10-fold drop), you cannot compensate with a 10-fold increase in observations. You need **100 times more data** ($10^2$). Weak sources are asymptotically useless in moderate sample sizes.

---

## 6. Operational Gate FG3

The DAVID engine guards against this collapse by computing $I$ from the MCMC posterior draws:
1. It extracts the sensitivity $\rho_s$ and false positive rate $\delta_s$ per draw.
2. It evaluates $I = |\rho_s - \delta_s|$ and checks the lower 95% credible bound.
3. If $I_{\text{lower 95\%}} < 0.10$, **Gate FG3 fails**. The cell is routed as `prior_dominated`, and the UI hides the conditional point prediction, forcing a safe uninformative interval.
