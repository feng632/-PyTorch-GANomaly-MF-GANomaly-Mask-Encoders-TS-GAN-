# Reproduction Results

## GANomaly baseline on MVTec AD grid

| Preprocessing | Aggregation | Seed | Image AUROC |
|---|---|---:|---:|
| Whole image resized to 64x64 | latent score | 42 | 0.519632 |
| Four deterministic quadrants | maximum patch score | 42 | 0.681704 |
| Four deterministic quadrants | mean patch score | 42 | 0.702590 |

The thesis reports 0.658 for GANomaly on `grid`. The exact patch-to-image aggregation is not specified, so both maximum and mean aggregation are recorded. Results above are from one seed and should not be treated as a final statistical comparison.

## Reproduction differences

- Thesis: PyTorch 1.8.0, CUDA 11.4, RTX 2060.
- This implementation: Python 3.11, PyTorch 2.11.0+cu128, RTX 5070 Laptop.
- Model code was reconstructed from the thesis description because author code was not found.
- Formal sliding-window PSNR anomaly maps are not implemented yet.
