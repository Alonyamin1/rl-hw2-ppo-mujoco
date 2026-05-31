# Reinforcement Learning HW2 — PPO Implementation Report

## 1. Environments

We trained on two MuJoCo environments using the Gymnasium API (version v4):

- **Hopper-v4**: A one-legged robot that must learn to hop forward without falling.
  Observation space: 11-dimensional. Action space: 3-dimensional (continuous joint torques).
- **HalfCheetah-v4**: A two-legged cheetah robot that must learn to run forward as fast as possible.
  Observation space: 17-dimensional. Action space: 6-dimensional (continuous joint torques).

Each environment was trained independently as a separate run.

---

## 2. Algorithm: PPO with Clipped Surrogate Objective

We implemented Proximal Policy Optimization (PPO) from scratch in PyTorch, following
Algorithm 1 and Section 3 of Schulman et al. (2017).

### Overview

PPO is an on-policy actor-critic algorithm. Each iteration proceeds as:
1. Collect T timesteps of experience using the current policy.
2. Compute Generalized Advantage Estimates (GAE) over the collected batch.
3. Optimize the clipped surrogate objective for K epochs using minibatch gradient descent.
4. Discard the collected data and repeat.

---

## 3. Network Architecture

Both the actor and critic are separate feedforward networks (no shared parameters),
following the setup from Section 6.1 of the paper.

**Actor (Policy network)**
- Input: observation vector (11-dim for Hopper, 17-dim for HalfCheetah)
- 2 hidden layers of 64 units each, tanh activations
- Output: mean vector of a Gaussian distribution (3-dim or 6-dim)
- A separate learnable parameter vector `log_std` (one value per action dimension,
  not a network output) represents the log standard deviation of the policy.
  This is state-independent, as used in the paper.

**Critic (Value network)**
- Same architecture: input → 64 → 64 → scalar V(s)

**Weight initialization**: Orthogonal initialization with gain sqrt(2) for hidden layers,
gain 0.01 for the actor output layer (encourages small initial actions), gain 1.0 for the
critic output layer.

---

## 4. Policy and Action Selection

During training, actions are sampled from a multivariate Gaussian:

    a ~ N(μ(s), σ)

where μ(s) is the actor network output and σ = exp(log_std).

Actions are clipped to the environment's valid action range before being passed to env.step(),
but the unclipped sampled action is used for computing log probabilities to avoid gradient issues.

**During evaluation**, actions are deterministic: we use the mean μ(s) directly without sampling.

---

## 5. Advantage Estimation: GAE

We use Generalized Advantage Estimation (GAE, Schulman et al. 2015) with the following formula:

    δ_t  = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    Â_t  = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...

Computed backwards through the rollout buffer. The `(1 − done)` terms ensure that
advantage estimates do not bootstrap across episode boundaries.

After computing all advantages, we normalize them to zero mean and unit variance
across the entire batch. This keeps gradient magnitudes consistent regardless of
environment reward scale (important since Hopper and HalfCheetah have very different reward ranges).

---

## 6. PPO Loss Function

The combined objective (equation 9 from the paper) is:

    L = L_CLIP − c₁ · L_VF + c₂ · S[π]

We minimize the negation of this in PyTorch:

    loss = policy_loss + c₁ · value_loss − c₂ · entropy

**Policy loss (clipped surrogate)**:

    r_t(θ)  = exp( log π_new(a|s) − log π_old(a|s) )
    L_CLIP  = mean( min(r_t · Â_t,  clip(r_t, 1−ε, 1+ε) · Â_t) )

The ratio r_t is computed in log space (subtraction then exp) rather than directly
dividing probabilities. This is done for numerical stability: raw action probabilities
under a Gaussian can be extremely small (e.g. 10⁻⁸), and dividing such values causes
floating-point instability. Working in log space keeps values in a numerically
well-behaved range.

**Value loss**:

    L_VF = MSE( V(s_t), R_t )

where R_t = Â_t + V(s_t) are the computed returns (targets for the critic).

**Entropy bonus**:

    S[π] = mean( entropy of Normal distribution at each state )
         = mean( sum over dimensions of: 0.5 · log(2πe · σ²) )

---

## 7. Hyperparameters

| Parameter              | Value       | Notes |
|------------------------|-------------|-------|
| Horizon T              | 4096        | Steps collected per PPO iteration |
| PPO epochs K           | 10          | Gradient passes over each batch |
| Minibatch size M       | 128         | Scaled with horizon (T/M = 32, same ratio as paper) |
| Discount γ             | 0.99        | Standard for MuJoCo |
| GAE λ                  | 0.95        | Bias-variance tradeoff |
| Initial learning rate  | 3 × 10⁻⁴   | Linearly annealed to 0 |
| Clip epsilon ε         | 0.2         | PPO clipping range |
| VF coefficient c₁      | 0.5         | Value loss weight |
| Entropy coefficient c₂ | 0.01        | Entropy bonus weight |
| Gradient clip norm     | 0.5         | Max gradient norm |
| Total training steps   | 10,000,000  | Per run (3 runs per environment) |
| Evaluation frequency   | 10,000 steps | As required |
| Evaluation episodes    | 10          | Per evaluation checkpoint |
| Final evaluation       | 100 episodes | Reported as final reward |

---

## 8. Differences from the Original Paper

The paper's Table 3 hyperparameters are used as a starting point. The following
deliberate modifications were made:

1. **Larger horizon and minibatch (T=4096, M=128)**: The paper uses T=2048, M=64 for a
   1M-step budget. We train for 3M steps per run, so doubling the horizon keeps the
   number of PPO iterations reasonable (≈2441 vs ≈4882) while providing smoother gradient
   estimates per update. The minibatch is scaled proportionally to keep the T/M ratio
   constant at 32.

2. **Linear learning rate annealing**: The paper uses a fixed lr=3×10⁻⁴ for MuJoCo.
   We anneal it linearly from 3×10⁻⁴ to 0 over the full training run.
   Motivation: later in training, the policy is already good — large gradient steps
   can destabilize it. Annealing allows fine-grained adjustments near convergence.
   (The paper itself uses this for Atari, Table 5.)

2. **Entropy bonus c₂=0.01**: The paper explicitly states for MuJoCo: "we don't use an
   entropy bonus." We add a small one to encourage continued exploration, especially
   useful in the early stages of Hopper training where the robot falls over immediately
   and the policy can get stuck in local minima.

3. **Gradient clipping (max_norm=0.5)**: Not mentioned in the paper for MuJoCo.
   Added as a safety measure against rare large gradient steps, particularly during
   early training when the value function estimate is poor.

4. **VF coefficient c₁=0.5**: The paper uses separate networks and states c₁ is
   "irrelevant." Since we also use separate networks, this only scales the value
   loss contribution in the combined backward pass. Set to 0.5 as is standard.

---

## 9. Training Procedure

- 3 independent training runs per environment, each with a different random seed
  (seeds 1, 2, 3 correspond to run_id 1, 2, 3).
- 10,000,000 training steps per run (maximum allowed by the assignment).
- Evaluation every 10,000 steps using 10 deterministic episodes (mean action),
  producing 1,000 evaluation points per run.
- Checkpoint saved after every evaluation for resume capability.
- Final reported reward: mean of 100 deterministic evaluation episodes at end of training.

---

## 10. Results

*(To be filled after training)*

### Hopper-v4

| Run | Final Reward (100 eps) |
|-----|------------------------|
| 1   | TBD                    |
| 2   | TBD                    |
| 3   | TBD                    |
| **Average** | **TBD**       |

### HalfCheetah-v4

| Run | Final Reward (100 eps) |
|-----|------------------------|
| 1   | TBD                    |
| 2   | TBD                    |
| 3   | TBD                    |
| **Average** | **TBD**       |

*(Learning curve plots to be inserted here — one per run per environment)*

---

## 11. What Worked

*(To be filled after training)*

---

## 12. What Did Not Work

*(To be filled after training)*

---

## References

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov.
"Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.

J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel.
"High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438, 2015.
