# Deep Dive: Theorem A′ — Source-Channel Latent-Class Identifiability

> [!TIP]
> **Publication-Ready Deep-Dive PDF Available:** 
> * **Compiled PDF Paper:** [theorem_1_deep_dive.pdf](file:///Users/arbenlila/development/david/canonical/docs/theorem_1_deep_dive.pdf) (compiled via Tectonic)
> * **LaTeX Source:** [theorem_1_deep_dive.tex](file:///Users/arbenlila/development/david/canonical/docs/theorem_1_deep_dive.tex)

This document provides a detailed, step-by-step pedagogical explanation of **Theorem A′ (Source-Channel Latent-Class Identifiability)**. It is designed to help you thoroughly understand the math, the proof, how it behaves in practice, and why it is resilient to real-world challenges.

---

## 1. The Core Conceptual Problem

Imagine you are tracking whether the Tobacco Industry (TI) is actively obstructing smoke-free policies in Kosovo. The active presence of this obstruction is a **latent variable**—it exists in reality, but it cannot be directly measured with $100\%$ certainty. We denote this true latent state as:
* $A_g = 1$: Obstruction is active in stratum $g$ (Kosovo × smoke-free policy).
* $A_g = 0$: Obstruction is inactive in stratum $g$.

To observe this hidden state, you deploy **three noisy sources**:
1. **Source 1 ($Y_1$):** A local news RSS scraper.
2. **Source 2 ($Y_2$):** A legislative monitor api.
3. **Source 3 ($Y_3$):** An LLM classifier analyzing scientific publication abstracts.

Each source is imperfect. Sometimes they miss actual obstruction (false negatives), and sometimes they flag normal public relations as obstruction (false positives).

### The Circularity Paradox
This presents a classic chicken-and-egg problem:
* To know how **reliable** a source is, you need to know the **ground-truth** state ($A_g$).
* To estimate the **ground-truth** state ($A_g$), you need to know the **reliability** of your sources.

If you don't know either, can you ever estimate the true probability of obstruction $\phi_g = P(A_g=1)$ and the error rates of the sources at the same time? Or are there infinite different combinations of source error rates and activity probabilities that could explain the exact same data?

**Theorem A′ solves this paradox.** It mathematically guarantees that as long as you have **at least 3 independent sources**, there is **exactly one unique solution** to the problem. You can simultaneously recover the true prevalence of obstruction and the individual error rates of every source.

---

## 2. Breaking Down the Parameters

To understand the mathematical proof, let's first define each item in the model clearly:

| Parameter / Variable | Mathematical Meaning | Real-World Translation (Kosovo Example) |
| :--- | :--- | :--- |
| **$A_g \in \{0, 1\}$** | The latent (hidden) state. | Is there actual, hidden TI obstruction happening in Kosovo? (Yes = 1, No = 0) |
| **$\phi_g \in [0, 1]$** | Prevalence parameter: $P(A_g = 1)$ | The true probability that the tobacco industry is actively obstructing the policy. |
| **$Y_s \in \{0, 1\}$** | Observed output of Source $s$. | Did the RSS feed ($Y_1$), Parliamentary API ($Y_2$), or LLM ($Y_3$) flag an incident? |
| **$\rho_s \in [0, 1]$** | Sensitivity: $P(Y_s = 1 \mid A_g = 1)$ | True Positive Rate: The probability that Source $s$ flags an incident when TI obstruction is *genuinely* occurring. |
| **$\delta_s \in [0, 1]$** | False Positive Rate: $P(Y_s = 1 \mid A_g = 0)$ | Noise Rate: The probability that Source $s$ flags an incident even though the industry is *not* obstructing. |

---

## 3. Step-by-Step Mathematical Proof

We observe only the joint outcomes of our three sources. We can count how often we see the combinations:
* $(Y_1=0, Y_2=0, Y_3=0)$
* $(Y_1=1, Y_2=0, Y_3=0)$
* ... up to $(Y_1=1, Y_2=1, Y_3=1)$

This joint probability distribution forms a $2 \times 2 \times 2$ grid of numbers, which mathematicians call a **3-way tensor**, denoted as $T_{ijk} = P(Y_1=i, Y_2=j, Y_3=k)$.

### Step 3.1: Writing the Joint Probability as a Tensor Decomposition
Under the assumption of **conditional independence** (the sources do not coordinate with each other; their outputs depend *only* on the true underlying state $A_g$), we can write the joint probability for any combination $(i, j, k) \in \{0, 1\}^3$ as:
$$T_{ijk} = P(Y_1=i, Y_2=j, Y_3=k) = \sum_{a \in \{0, 1\}} P(A_g = a) P(Y_1=i \mid A_g=a) P(Y_2=j \mid A_g=a) P(Y_3=k \mid A_g=a)$$

Let's write this in matrix terms. For each source $s$, we define a conditional probability matrix $M^{(s)}$ of size $2 \times 2$:
$$M^{(s)} = \begin{pmatrix} 
P(Y_s=0 \mid A=0) & P(Y_s=0 \mid A=1) \\ 
P(Y_s=1 \mid A=0) & P(Y_s=1 \mid A=1) 
\end{pmatrix} = \begin{pmatrix} 
1 - \delta_s & 1 - \rho_s \\ 
\delta_s & \rho_s 
\end{pmatrix}$$
And we define the latent prior distribution vector:
$$\boldsymbol{\pi} = \begin{pmatrix} 1 - \phi \\ \phi \end{pmatrix}$$

Now, the tensor $T_{ijk}$ can be written as:
$$T_{ijk} = \sum_{a=0}^{1} \pi_a M^{(1)}_{ia} M^{(2)}_{ja} M^{(3)}_{ka}$$
This is the **Canonical Polyadic (CP) tensor decomposition** of a tensor $T$ into a sum of $R=2$ rank-1 tensors (since there are 2 latent classes).

---

### Step 3.2: Understanding Kruskal's Theorem (1977)
Kruskal proved a fundamental theorem about when a tensor CP decomposition is unique. 

Unlike matrices, where a rank-2 matrix can be factored into infinite different matrix combinations, **tensors of order 3 or higher have unique factorizations under very mild conditions.**

Kruskal defined the **Kruskal rank** (denoted $k(M)$) of a matrix $M$:
> **Definition:** The Kruskal rank $k(M)$ is the largest integer $k$ such that **any** set of $k$ columns of $M$ is linearly independent.

Since our factor matrices $M^{(s)}$ have size $2 \times 2$, their maximum possible Kruskal rank is $2$.
* When are the two columns of $M^{(s)}$ linearly independent?
* Answer: When they are not multiples of each other, meaning the determinant of $M^{(s)}$ is non-zero.

Let's compute the determinant of $M^{(s)}$:
$$\det(M^{(s)}) = (1 - \delta_s)\rho_s - (1 - \rho_s)\delta_s = \rho_s - \delta_s$$

Therefore, as long as **$\rho_s \neq \delta_s$** (the source's true positive rate is not identical to its false positive rate—i.e., it is not a random coin toss), the columns are linearly independent.
This means:
$$k(M^{(s)}) = 2 \quad \text{for } s \in \{1, 2, 3\}$$

---

### Step 3.3: Satisfying the Uniqueness Bound
Kruskal's Theorem states that the CP decomposition of $T$ into $R$ rank-1 components is unique up to permutation and scaling if:
$$k(M^{(1)}) + k(M^{(2)}) + k(M^{(3)}) \ge 2R + 2$$

Let's substitute our values:
* The sum of our Kruskal ranks is: $2 + 2 + 2 = 6$.
* The threshold we must meet is: $2(2) + 2 = 6$.

Since **$6 \ge 6$**, the decomposition is **mathematically unique**. 

By the extension of this theorem to latent class analysis (**Allman-Matias-Rhodes, 2009**), this uniqueness guarantees that the parameters $(\phi, \{\rho_s, \dots\}, \dots)$ are uniquely identifiable up to a simple swap of the labels (which class we call "1" and which we call "0"). $\blacksquare$

---

## 4. Concrete Example: Why 2 Sources Fail and 3 Sources Succeed

Let's look at the degrees of freedom (independent equations) to understand why this works.

### Scenario A: Only 2 Sources ($Y_1, Y_2$)
With 2 binary sources, you observe 4 joint states: $(0,0), (0,1), (1,0), (1,1)$. 
Because these probabilities must sum to 1, you have **3 independent equations** (degrees of freedom) in your observed data.

However, how many parameters are you trying to find?
* True activity prior: $\phi$ (1 parameter)
* Source 1 parameters: $\rho_1, \delta_1$ (2 parameters)
* Source 2 parameters: $\rho_2, \delta_2$ (2 parameters)
* **Total parameters = 5**

You cannot solve a system of equations with **3 equations and 5 unknowns**. There are infinite different combinations of priors and error rates that yield the exact same observed data. The model is **unidentifiable**.

### Scenario B: 3 Sources ($Y_1, Y_2, Y_3$)
With 3 binary sources, you observe 8 joint states. Since they must sum to 1, you have **7 independent equations** in your observed data.

How many parameters are you trying to find?
* True activity prior: $\phi$ (1 parameter)
* Source 1: $\rho_1, \delta_1$ (2 parameters)
* Source 2: $\rho_2, \delta_2$ (2 parameters)
* Source 3: $\rho_3, \delta_3$ (2 parameters)
* **Total parameters = 7**

You now have **7 equations and 7 unknowns**. The mathematical system is completely determined! 

---

## 5. Resilience to Challenges: How the Engine Holds Up

In the real world, mathematical assumptions are challenged by noisy or biased environments. Here is how Theorem A′ handles these attacks, and how the DAVID engine enforces them.

### Challenge 1: The "Collusion" Attack (Conditional Dependence)
* **The Threat:** What if Source 1 (RSS feed) and Source 2 (Legislative Monitor) are not independent? For instance, what if the legislative monitor simply copy-pastes news reports from the RSS feed?
* **The Math Impact:** If the sources are dependent, the joint probability is no longer a simple product: $P(Y_1, Y_2 \mid A) \neq P(Y_1 \mid A)P(Y_2 \mid A)$. The tensor equation breaks down, and Kruskal's theorem no longer applies. The model will overestimate confidence and produce biased estimates.
* **How DAVID defends:** The engine maintains a strictly reviewed **Source Independence Ledger** (`config/source_independence.json`). Pairwise independence weights are set manually based on structural audits (e.g., common ownership or shared feed APIs). If two sources are highly dependent, they are combined or penalized, and the adversarial check `F11` (conditional independence) triggers a gate failure if dependence is detected.

### Challenge 2: The "Noise" Attack (Non-Informative Sources)
* **The Threat:** What if Source 3 is completely useless (e.g., it flags everything randomly, meaning $\rho_3 \approx \delta_s$)?
* **The Math Impact:** If $\rho_3 = \delta_3$, the columns of $M^{(3)}$ are identical, making them linearly dependent. The Kruskal rank of $M^{(3)}$ collapses from 2 to 1.
  The sum of Kruskal ranks becomes:
  $$k(M^{(1)}) + k(M^{(2)}) + k(M^{(3)}) = 2 + 2 + 1 = 5$$
  The Kruskal bound is violated ($5 < 6$). Uniqueness is lost, and the parameters cannot be recovered.
* **How DAVID defends:** The engine checks this dynamically via the **Practical Identification Distance** $d(\theta)$ for every stratum fit. If a source's sensitivity and false-positive rates drift within $0.05$ of each other, the distance $d(\theta)$ drops below the required `ID_DISTANCE_FLOOR = 0.05`, and the stratum is automatically blocked via **Gate FG2**, withholding the forecast to prevent publishing noise.

```
                  ┌─────────────────────────────────────┐
                  │ Compute d(θ) over all MCMC draws    │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │    Is median d(θ) >= 0.05?           │
                  └────────┬───────────────────┬────────┘
                           │                   │
                     (Yes) │                   │ (No)
                           ▼                   ▼
                ┌─────────────────────┐   ┌──────────────────────────────┐
                │ Pass Gate FG2       │   │ Fail Gate FG2: Block stratum │
                │ Route: Certified    │   │ Route: evidence_gap          │
                └─────────────────────┘   └──────────────────────────────┘
```

### Challenge 3: The "Label-Switching" Ambiguity
* **The Threat:** Since the tensor decomposition is unique up to permutation, the model cannot distinguish between:
  1. $A_g=1$ means "Obstruction is active" and $A_g=0$ means "Obstruction is inactive".
  2. $A_g=1$ means "Obstruction is inactive" and $A_g=0$ means "Obstruction is active".
  This is called the "label-switching" problem, where the model might flip the definition of 0 and 1.
* **How DAVID defends:** We break this symmetry by applying an **order constraint** in both the generative model and the Stan model parameters:
  * We enforce that the activity parameters in the first tactic class are ordered ascending: $\alpha_{\text{activity}}[:, 0]$ is sorted.
  * This breaks the mathematical symmetry of the latent classes, anchoring State 1 and State 0 to their correct biological/operational definitions, ensuring the model never flips your predictions.
