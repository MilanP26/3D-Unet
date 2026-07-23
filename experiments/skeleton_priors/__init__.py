"""Structural-prior experiments that reuse the production pipeline's outputs.

Kept separate from src/seeded_unet on purpose (see experiments/README.md): this
package only *reads* what the trained affinity model produces, so nothing here
requires retraining the model or touching the GPU.
"""
