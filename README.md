# Neuro-Fisher

This repository contains the implementation for the paper:

**Quantifying Signal-to-Noise Ratio in Neural Latent Trajectories via Fisher Information**. European Signal Processing Conference. EUSIPCO, Lyon, France.
*Hyungju Jeon, Il Memming Park*
[arXiv:2408.08752](https://arxiv.org/abs/2408.08752)

## Abstract

Spike train signals recorded from a large population of neurons often exhibit low-dimensional spatio-temporal structure and modeled as conditional Poisson observations. The low-dimensional signals that capture internal brain states are useful for building brain machine interfaces and understanding the neural computation underlying meaningful behavior. We derive a practical upper bound to the signal-to-noise ratio (SNR) of inferred neural latent trajectories using Fisher information. We show that the SNR bound is proportional to the overdispersion factor and the Fisher information per neuron. Further numerical experiments show that inference methods that exploit the temporal regularities can achieve higher SNRs that are proportional to the bound. Our results provide insights for fitting models to data, simulating neural responses, and design of experiments.

---

## Overview and Core Features
![Overview](figs/numerical_analysis.png)
- Generate Poisson spike train observations and log-linear observation model parameters with controllable signal-to-noise ratio (SNR)
- Compute Instantaneous Observed Fisher Information 
- (To be implemented) Estimate Cramér-Rao lower bounds on estimation accuracy for the latent trajectory
- (To be implemented) Evaluate and compare inference results with theoretical uncertainty

---

## Installation

Clone and run from source:

```bash
git clone https://github.com/hyungju-jeon/neuro-fisher.git
cd neuro-fisher
```

```bash
pip install -r requirements.txt 
```
or 
```bash
conda env create -f environment.yml
```

---

## Usage

Generate Poisson spike train observations and log-linear observation model parameters with target signal-to-noise ratio (SNR) given a latent trajectory.

```python
observations, loading_matrix, bias, firing_rate_per_bin, snr = gen_poisson(
    x=latent_trajectory,
    C=None,
    d_neurons=num_neurons,
    tgt_rate=target_rate,
    p_coh=p_coherence,
    p_sparse=p_sparse,
    tgt_snr=target_snr,
)
```
- If `C` is not provided, it will be generated with target coherence and sparsity.
- If `C` is provided, it will be used as the loading matrix but scaled to match the target signal-to-noise ratio (SNR).


Check out the demo folder for complete examples.

Demo for generating log-linear Poisson observations from GP latent trajectory given Fisher Information SNR bound:
```bash
python demo/simulate_observation_gp.py
```
and from 2D Ring latent trajectory:
```bash
python demo/simulate_observation_ring.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Jeon2024b,
  author    = {Hyungju Jeon and Il Memming Park},
  booktitle = {European Signal Processing Conference},
  title     = {Quantifying signal-to-noise ratio in neural latent trajectories via {F}isher information},
  month     =  aug,
  year      = {2024},
  archivePrefix = "arXiv",
  primaryClass  = "q-bio.NC",
  eprint        = "2408.08752"
}
```
