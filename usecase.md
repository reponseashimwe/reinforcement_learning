# Dermatology Triage Clinic - Use Case Documentation

## Overview

This document describes the custom mission-specific environment developed for reinforcement learning research: a **Dermatology Triage Clinic** simulator. The environment combines spatial resource management, patient scheduling, and medical decision-making under uncertainty.

---

## Problem Statement

### Clinical Context

A small dermatology clinic receives patients with various skin complaints throughout the day. The clinic has:

-   **Limited resources**: Exam rooms, dermatologists, and nurse practitioners
-   **Variable patient severity**: From minor rashes to potentially critical lesions
-   **Time constraints**: Patients waiting increase dissatisfaction and risk
-   **Diagnostic uncertainty**: Initial symptoms don't always reveal true severity

### Challenge

The agent must learn to:

1. **Correctly triage patients** to appropriate care levels based on observable symptoms
2. **Manage clinic resources** efficiently (open/close exam rooms dynamically)
3. **Minimize patient wait times** while maintaining high diagnostic accuracy
4. **Balance exploration vs. exploitation** to learn optimal policies

### Why This Is Non-Generic

Unlike simple grid-world or game environments, this use case integrates:

-   **Clinical decision-making**: Symptom-based severity assessment (domain-specific)
-   **Resource allocation**: Dynamic room management with operational costs
-   **Multi-objective optimization**: Accuracy, throughput, and wait time tradeoffs
-   **Partial observability**: True severity is hidden; agent sees noisy symptom features
-   **Sequential decision-making**: Each action affects queue dynamics and future states

---

## Environment Specification

### Patient Model (Ground Truth)

Each patient is generated with a **hidden severity level** that determines the correct triage:

| Severity Level   | Description                   | Prevalence | Examples                                                |
| ---------------- | ----------------------------- | ---------- | ------------------------------------------------------- |
| **Mild (0)**     | Minor skin complaints         | 40%        | Minor rash, dry skin, cosmetic concerns                 |
| **Moderate (1)** | Requires physical examination | 35%        | Persistent dermatitis, suspicious moles, moderate acne  |
| **Severe (2)**   | Needs specialist attention    | 20%        | Infected lesions, rapidly changing moles, severe burns  |
| **Critical (3)** | Urgent medical attention      | 5%         | Suspected melanoma, severe infections, anaphylaxis risk |

### Symptom Feature Generation

Patients present with **observable features** that correlate with (but don't perfectly reveal) their severity:

#### 1. **Demographic Features**

-   **Age (normalized)**: `age_norm ∈ [0, 1]`
    -   Generated from normal distribution, clipped to [0, 1]
    -   Older patients may have higher-risk lesions

#### 2. **Clinical History**

-   **Duration (normalized)**: `duration_norm ∈ [0, 1]`
    -   How long symptoms have persisted
    -   Generated from exponential distribution (most cases recent, some chronic)
    -   Longer duration may indicate progression

#### 3. **Vital Sign Flags (Binary)**

-   **Fever Flag**: `fever_flag ∈ {0, 1}`

    -   Probability increases with severity: P(fever) = 0.05 + 0.20 × severity
    -   Indicates possible systemic infection

-   **Infection Flag**: `infection_flag ∈ {0, 1}`
    -   Probability increases with severity: P(infection) = 0.05 + 0.25 × severity
    -   Signs of local or systemic infection

#### 4. **Symptom Embedding (8-dimensional vector)**

-   **Vector**: `symptom_embed ∈ [0, 1]^8`
-   Represents quantified clinical observations:
    -   Redness intensity
    -   Border irregularity
    -   Size/diameter
    -   Itching/pain level
    -   Texture changes
    -   Color variation
    -   Asymmetry
    -   Evolution speed

**Generation Process**:

```python
if severity == 0:  # Mild
    base = 0.2
elif severity == 1:  # Moderate
    base = 0.45
elif severity == 2:  # Severe
    base = 0.70
else:  # Critical
    base = 0.95

symptom_embed = Normal(mean=base, std=0.08) + noise
```

This creates a **noisy mapping** from severity to symptoms, forcing the agent to learn patterns under uncertainty.

---

## Observation Space

The agent receives a **15-dimensional continuous observation vector**:

```
[0]     age_norm           (0.0 - 1.0)
[1]     duration_norm      (0.0 - 1.0)
[2]     fever_flag         (0.0 or 1.0)
[3]     infection_flag     (0.0 or 1.0)
[4-11]  symptom_embed[0-7] (0.0 - 1.0 each)
[12]    room_avail         (0.0 or 1.0)
[13]    queue_len_norm     (0.0 - 1.0)
[14]    time_of_day_norm   (0.0 - 1.0)
```

**Key Properties**:

-   **Continuous observations**: Requires function approximation (neural networks)
-   **Partial observability**: True severity is hidden
-   **Contextual information**: Queue and resource state inform decisions
-   **Temporal dimension**: Episode progress affects optimal strategy

---

## Action Space

The agent selects from **8 discrete actions**:

| Action ID | Action Name         | Description                 | Use Case                                     |
| --------- | ------------------- | --------------------------- | -------------------------------------------- |
| **0**     | `send_doctor`       | Route to dermatologist      | Severe/critical cases requiring specialist   |
| **1**     | `send_nurse`        | Route to nurse practitioner | Moderate cases, follow-ups                   |
| **2**     | `remote_advice`     | Telemedicine consultation   | Mild cases, reassurance, non-urgent          |
| **3**     | `escalate_priority` | Mark urgent + doctor        | Critical cases needing immediate attention   |
| **4**     | `defer_patient`     | Postpone to end of queue    | Ambiguous cases, need more observation time  |
| **5**     | `idle`              | Wait/observe                | Strategic waiting when resources constrained |
| **6**     | `open_room`         | Add exam room capacity      | High queue load, increase throughput         |
| **7**     | `close_room`        | Reduce room capacity        | Low demand, reduce operational costs         |

### Action Space Justification

This action set is **exhaustive and mission-specific** because it covers:

1. **Triage decisions** (actions 0-3): All possible care pathways
2. **Queue management** (action 4): Handling uncertainty
3. **Operational strategy** (action 5): Patience/observation
4. **Resource control** (actions 6-7): Dynamic capacity management

### Correct Triage Mapping (Ground Truth)

| Patient Severity | Optimal Action(s)       | Reasoning                                    |
| ---------------- | ----------------------- | -------------------------------------------- |
| Mild (0)         | `remote_advice` (2)     | Cost-effective, safe for low-risk cases      |
| Moderate (1)     | `send_nurse` (1)        | Physical exam needed, nurse qualified        |
| Severe (2)       | `send_doctor` (0)       | Specialist assessment required               |
| Critical (3)     | `escalate_priority` (3) | Immediate specialist attention with priority |

---

## Reward Structure

The agent receives rewards based on **three objectives**:

### 1. Triage Accuracy Rewards

**Correct triage**:

-   Mild → Remote: **+1.0**
-   Moderate → Nurse: **+1.0**
-   Severe → Doctor: **+2.0**
-   Critical → Escalate (wait < 5 steps): **+3.0**
-   Critical → Escalate (wait ≥ 5 steps): **+2.0**

**Incorrect triage**: **-1.5** (penalty for misdiagnosis)

### 2. Wait Time Penalty

```
penalty = -0.01 × queue_size (per step)
```

-   Penalizes long queues
-   Encourages throughput
-   Scales with number of waiting patients

### 3. Resource Cost

```
penalty = -0.05 × num_open_rooms (per step)
```

-   Penalizes excessive room usage
-   Encourages efficient resource allocation
-   Balances capacity vs. operational cost

### Reward Function (Complete)

```
R(s, a) = R_triage + R_wait + R_resource

R_triage = correct_triage_reward(action, true_severity)
R_wait   = -0.01 × queue_length
R_resource = -0.05 × num_open_rooms
```

### Multi-Objective Tradeoffs

The agent must learn to balance:

-   **Accuracy vs. Speed**: Taking time to observe vs. quick triage
-   **Throughput vs. Quality**: Processing many patients vs. careful assessment
-   **Capacity vs. Cost**: Opening rooms for throughput vs. minimizing costs

This creates a **rich policy space** where different strategies can emerge.

---

## State Dynamics

### Episode Flow

1. **Initialization**:

    - Queue starts with 3 patients (warm start)
    - 1 exam room open
    - Current patient spawned

2. **Step Transition**:

    - Agent observes current patient features
    - Agent selects action
    - Reward computed based on correctness
    - If patient "treated" (actions 0-3, 5-7): spawn next patient from queue
    - If patient deferred (action 4): patient moved to end of queue
    - All queued patients' wait times increment
    - New observation formed

3. **Termination**:
    - Episode ends after **max_steps** (default 500)
    - No early termination (ensures consistent episode lengths)

### Patient Arrival Process

-   New patients arrive when queue empties (deterministic generation)
-   Each patient is **independently sampled** from severity distribution
-   No inter-patient dependencies (simplifying assumption)

---

## Why This Environment Is Challenging

### 1. Partial Observability

-   Agent never sees `true_severity` directly
-   Must learn probabilistic mapping: symptoms → severity → action
-   Noisy features require robust pattern recognition

### 2. Multi-Objective Optimization

-   No single reward component dominates
-   Pareto-optimal policies may exist
-   Requires careful credit assignment

### 3. Resource Management

-   Opening rooms increases throughput but costs resources
-   Optimal room count depends on queue dynamics
-   Strategic timing of capacity changes

### 4. Long-Term Credit Assignment

-   Incorrect triage has immediate negative reward
-   But wait time penalties accumulate over episode
-   Deferred patients affect future states

### 5. Exploration Challenge

-   8 actions × continuous observation space
-   Rare critical patients (5%) require special handling
-   Must explore edge cases during training

---

## Evaluation Metrics

### Primary Metrics

1. **Mean Episode Reward**: Overall performance measure
2. **Triage Accuracy (%)**: Percentage of correct triage decisions
3. **Average Wait Time**: Mean patient wait time per episode
4. **Episode Length**: Steps to complete episode (always max_steps)

### Secondary Metrics

5. **Correct Triages by Severity**: Breakdown by patient type
6. **Action Distribution**: How often each action is used
7. **Resource Efficiency**: Average rooms used vs. queue length
8. **Convergence Speed**: Episodes to reach stable performance

### Evaluation Protocol

-   **Training**: Seeds 0-4 (5 seeds)
-   **Validation**: Seeds 5-9 (5 seeds)
-   **Testing**: Seeds 10-19 (10 seeds)
-   **Episodes per evaluation**: 50-100
-   **Deterministic policy** during evaluation

---

## Clinical Relevance

While simplified, this environment captures real challenges in medical triage:

### Real-World Parallels

1. **Tele-dermatology**: Remote vs. in-person decisions
2. **Emergency departments**: Triage nurses assessing severity
3. **Resource-limited settings**: Optimizing specialist time
4. **Pandemic response**: Dynamic capacity management

### Limitations (Simplifications)

-   No patient heterogeneity beyond severity
-   Deterministic treatment outcomes
-   No provider skill differences
-   No temporal arrival patterns (e.g., rush hours)
-   No patient preferences or satisfaction beyond wait time

### Extensions for Future Work

-   Add patient-provider matching
-   Include diagnostic uncertainty with tests
-   Model time-of-day arrival patterns
-   Add provider fatigue dynamics
-   Multi-clinic coordination

## References

-   Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. _Machine Learning_.
-   Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. _Nature_.
-   Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv_.
-   Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. _ICML_.

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Author**: Reponse Ashimwe
