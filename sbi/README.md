# SBI inputs

`make_summaries.py` runs the trained CNNs over every simulated cluster and A2744 and
saves what the simulation-based inference needs:

| file | contents |
|---|---|
| `summaries/params_<domain>.npz` | per cluster: `cross_section`, `log10_mass`, `log10_mass_750`, `ncomponents`, `catalogue_mass`, `filename`, `image_idx`; per model: `held_out`, `seeds` |
| `summaries/softmax_<domain>.npy` | P(SIDM), shape (n_models, n_noise, N) |
| `summaries/latents_<domain>.npy` | 512-d latents, shape (n_models, n_noise, N, 512). Not in git (~1 GB), so regenerate it |
| `summaries/obs.npz` | A2744 `softmax` (n_models, n_obs) and `latents` (n_models, n_obs, 512); index 0 is the data, the rest are random-rotation noise maps |
| `summaries/info.json` | models, options and input-file dates |

Rows are the same clusters in the same order for every model and every file, so
`softmax[m, r]`, `latents[m, r]` and the parameters line up directly.

    cd ml_check
    astro-env/bin/python sbi/make_summaries.py                       # all 30 models
    astro-env/bin/python sbi/make_summaries.py --legacy-dir ../Dark_Matter_DA/sbi/inputs

`--legacy-dir` also writes `latents.pkl`, `theta.pkl`, `obs_latents.pkl`,
`latent_softmax.pkl` and `latent_meta_fixed.pkl` in the format read by the old
`Dark_Matter_DA/sbi` scripts (their paths need pointing at that directory).
