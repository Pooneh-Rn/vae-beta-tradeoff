# Generative Modeling and Latent-Space Exploration (beta-VAE Trade-off Study)

Training a variational autoencoder on MNIST at several KL weights (beta) to study the trade-off
between reconstruction quality and latent-space organization. Built as part of an Advanced Data
Mining course project (Spring 2026).

## Summary

- **Data**: MNIST (55,000 train / 5,000 val / 10,000 test).
- **Model**: a fixed convolutional VAE architecture (16-dim latent space), trained five times at
  beta = 0.1, 1, 5, 20, 50 — everything else held identical.
- **Original contribution**: quantifying the reconstruction-vs-generation trade-off directly,
  including a classifier-based interpolation-quality metric (not just eyeballed samples) and PCA
  visualizations of the latent space at low vs. high beta.

| beta | Recon loss | KL loss | Interpolation confidence |
|---|---|---|---|
| 0.1 | 72.72 | 49.14 | 0.879 |
| 1 | 81.19 | 24.71 | **0.890** |
| 5 | 120.57 | 7.79 | 0.809 |
| 20 | 195.84 | 0.44 | 0.596 |
| 50 | 206.37 | 0.00 | 0.563 |

Reconstruction loss rises and KL loss falls monotonically with beta, as expected. Interpolation
quality peaks at beta=1, not at the lowest beta tested — too little KL pressure leaves gaps in
the latent space (garbled early interpolation frames), while too much pressure causes posterior
collapse (KL loss hits ~0, the latent space loses all class structure, and the decoder starts
outputting a generic blurry digit regardless of input). Full writeup, including the PCA and
interpolation-grid evidence for both failure modes, is in [`report.md`](./report.md).

## Running it

Self-contained, runs top to bottom in Google Colab with a GPU runtime. Downloads MNIST
automatically — no manual data setup needed.

1. Open `notebook.ipynb` in Colab.
2. Set **Runtime > Change runtime type > GPU**.
3. **Runtime > Run all**.

An optional interactive demo (Gradio app) is included at the end of the notebook — pick a beta
and two digit classes, then drag a slider to interpolate between them in latent space live. Run
the two demo cells after the main training section to try it.

## Dependencies

Installed automatically by the first cell: `torch`, `torchvision`, `numpy`, `matplotlib`,
`scikit-learn`, `gradio` (only needed for the optional demo). No local setup needed if running in
Colab.

## Reproducibility

Random seed fixed at 42 throughout. All five VAE runs share the same architecture, data, and
seed — beta is the only variable that changes between them.
