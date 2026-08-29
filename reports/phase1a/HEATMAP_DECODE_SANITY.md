# Phase 1A synthetic heatmap/DSNT sanity

This report uses synthetic three-channel geometry only. It does not inspect real images and does not establish the cause of the saved B0 endpoint.

H is the raw heatmap used for MSE. P is its spatial Softmax and sums to one. DSNT is the coordinate expectation under P.

| case | decoder | T | raw MSE | normalized entropy | MRE px | AoP valid | penalized AoP score |
|---|---|---:|---:|---:|---:|---:|---:|
| gaussian_argmax | argmax | 0.05 | 0.00000000 | 0.246110 | 0.000 | yes | 0.000 |
| gaussian_dsnt_t1 | dsnt | 1 | 0.00000000 | 0.999945 | 104.044 | yes | 0.001 |
| gaussian_dsnt_t0.05 | dsnt | 0.05 | 0.00000000 | 0.246110 | 0.002 | yes | 0.000 |
| gaussian_amplitude_0.1_dsnt_t0.05 | dsnt | 0.05 | 0.00062126 | 0.999629 | 103.669 | yes | 0.001 |
| gaussian_amplitude_0.01_dsnt_t0.05 | dsnt | 0.05 | 0.00075173 | 0.999998 | 104.221 | yes | 0.010 |
| zero_heatmaps_dsnt_t0.05 | dsnt | 0.05 | 0.00076699 | 1.000000 | 104.255 | no | 180.000 |
| flat_heatmaps_dsnt_t0.05 | dsnt | 0.05 | 0.06250000 | 1.000000 | 104.255 | no | 180.000 |

Raw heatmap MSE is computed before softmax. Amplitude and temperature are reported as a fixed diagnostic matrix, not searched for model selection.
