# Meta-PINN
Continuously updating
## Overview

**Meta-PINN (Physics-Informed Meta-Learning Neural Network)** combines physical constraints with meta-learning to model UAV dynamics and achieve adaptive control under complex wind disturbances. 

Key features:
- Residual force modeling with **Physics-Informed Neural Networks (PINNs)**.
- **Conditional Nominal Modulation (CNM)** to capture environment dependence.
- **K-shot meta-adaptation** for rapid transfer to unseen wind conditions.
- Online **adaptive robust control** for stable UAV flight in challenging environments.
---

## Quick Start

### 1. Offline Training

```bash
python train_offline_meta_pinn.py 
```

### 2. Online Adaptation

```bash
python pinn_online_adaptive.py
```


## Experiments

* **Tasks**: figure-8 trajectory, circle, random path.
* **Wind profiles**: constant (0/5/10/12/15 m/s), sinusoidal, gusts, OU turbulence.
* **Baselines**: PID, Adaptive, NN+Adaptive, Meta-PINN+Adaptive.

Meta-PINN achieves significantly lower tracking error and faster adaptation under unseen wind conditions compared to baselines.
