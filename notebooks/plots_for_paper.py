#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_check modules

from get_model_probabilities import *
from add_shear_to_data import combine_catalogues, get_source_redshift, return_error_in_mean, get_model_names, SHEAR_CUTS, KAPPA_THRESH, SL_KAPPA_MAP
from sidm_inference_on_data_and_models import infer_sidm
import scienceplots
plt.style.use(["science","grid"])
'''
Paper Plots
-----------
In various notebooks i have carried out some tests, etc, in this notebook, i 
bring together all the plots that i have done and put the code here for 
the paper. These include= 
- Fig1. Examples of the training and test data
- Fig2. Main figure with the outputs and the CDM comparison
- Fig3. Sensitivity maps for analogues and the data
- Fig4. Correlation of sensitivity and convergence
- Fig5. SIDM inference
- Fig6. Final plot with literature
- Appendixes
    - FigA1. Redshift distributions and various tests on data
    - FigA2. How correlated are the models
    - FigA3. Latent space overlap
    - FigA4. Systematics in the data
    - FigA5. Systematics in the model
'''
if not os.path.isdir("plots"):
    os.system("mkdir plots")

SBI_DIR = "../sbi"
FIDUCIAL = "wb1"                                        # wb1: bahamas_1 kept in the fine-tuning (nob1: "spin2cx")
PAPER_TAG = "final"                                     # ba_final: the 30 fine-tuned seeds, final checkpoints

def kappa_on_stamp(path, cat, downsample=2):
    '''
    A convergence map (FITS with a celestial WCS) in the figure-1 pixel frame (stamp pixel
    index, 0-99 with origin lower), via an affine map from (RA, Dec) to stamp position
    fitted on the catalogue itself (cat['x'], cat['y'] already in stamp coordinates).
    Returns X, Y, kappa for ax.contour, and kappa interpolated at the galaxies.
    '''
    from astropy.wcs import WCS
    with fits.open(path) as h:
        kappa = h[0].data.astype(float)
        wcs = WCS(h[0].header)
    ra0, dec0 = np.median(cat['RA']), np.median(cat['DEC'])
    def design(ra, dec):
        return np.c_[(np.asarray(ra) - ra0) * np.cos(np.deg2rad(dec0)), np.asarray(dec) - dec0, np.ones(np.size(ra))]
    A = design(cat['RA'], cat['DEC'])
    cx = np.linalg.lstsq(A, cat['x'], rcond=None)[0]
    cy = np.linalg.lstsq(A, cat['y'], rcond=None)[0]
    rms = np.sqrt(np.mean((A @ cx - cat['x'])**2 + (A @ cy - cat['y'])**2))
    print(f"sky -> stamp affine fit: rms residual {rms:.3f} stamp pixels")
    jj, ii = np.mgrid[0:kappa.shape[0]:downsample, 0:kappa.shape[1]:downsample]
    ra, dec = wcs.pixel_to_world_values(ii, jj)
    D = design(ra.ravel(), dec.ravel())
    X = (D @ cx).reshape(ra.shape) + 49.5
    Y = (D @ cy).reshape(ra.shape) + 49.5
    gi, gj = wcs.world_to_pixel_values(np.asarray(cat['RA']), np.asarray(cat['DEC']))
    inside = (gi >= 0) & (gj >= 0) & (gi <= kappa.shape[1] - 1) & (gj <= kappa.shape[0] - 1)
    k_gal = np.full(len(gi), np.nan)
    k_gal[inside] = kappa[np.round(gj[inside]).astype(int), np.round(gi[inside]).astype(int)]
    return X, Y, kappa[::downsample, ::downsample], k_gal


def figure1(sl_kappa=None, kappa_level=KAPPA_THRESH):
    '''
    Postage stamp examples (plots/data_examples.png). With sl_kappa (a strong-lensing
    convergence map) the kappa = kappa_level (default the strong-lensing cut) contour is drawn on the A2744 panels and the
    figure is written to plots/data_examples_kappa<level>.png instead.
    '''
    # ## Postage stamp examples

    # ### Get the ideal data



    args = get_temp_args()
    args.source_domain = 'bahamas'
    args.target_domain = 'darkskies'
    args.apply_intrinsic_ell=0
    ideal_data = prepare_dataloaders(args)


    for i in ideal_data.keys():
        ideal_data[i][0].dataset.dataset.transform.transforms[2].apply = 0
        if 'source' in i:
            ideal_src = ideal_data[i][0].dataset.dataset[0]
        if 'target' in i:
            ideal_tgt = ideal_data[i][0].dataset.dataset[0]


    # ### Now get the obs sampled data


    idx=0


    args.source_domain = 'bahamas_obs'
    args.target_domain = 'darkskies_obs'
    obs_sampled_data = prepare_dataloaders(args)

    for i in obs_sampled_data.keys():
        obs_sampled_data[i][0].dataset.dataset.transform.transforms[3].apply = 0
        if 'source' in i:
            obs_sampled_src = obs_sampled_data[i][0].dataset.dataset[idx]
        if 'target' in i:
            obs_sampled_tgt = obs_sampled_data[i][0].dataset.dataset[idx]


    for i in obs_sampled_data.keys():
        obs_sampled_data[i][0].dataset.dataset.transform.transforms[3].apply = 1.
        if 'source' in i:
            obs_noisy_src = obs_sampled_data[i][0].dataset.dataset[idx]
        if 'target' in i:
            obs_noisy_tgt = obs_sampled_data[i][0].dataset.dataset[idx]


    # same catalogue cuts as the CNN input map (add_shear_to_data.make_obs_maps)
    obs_cat = get_obs_data( 'concat', data_dir='../data/100/a2744/', cuts=SHEAR_CUTS, drop_strong_lensing=True)
    obs_data = bin_obs_data( obs_cat )
    obs_cat['x'] = obs_data['delta_ra']
    obs_cat['y'] = obs_data['delta_dec']

    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_cat, smooth=1.5, correct_for_ngal=False, extent=[
                        -args.image_size//2,args.image_size//2,
                        -args.image_size//2,args.image_size//2
                    ]
    )


    fig, ax = plt.subplots(2, 5, figsize=(16,6))
    fig.subplots_adjust(hspace=0.1,wspace=0.)
    cmap='inferno'

    vmin = -1
    vmax = 1
    ax[0,2].imshow(ideal_src[0][0]-ideal_src[0][0].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[0,2].text( 5, 90, 'Ideal source domain ($\gamma_1$)', color='white', fontweight='bold')

    im = ax[1,2].imshow(ideal_tgt[0][1]-ideal_tgt[0][1].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[1,2].text( 5, 90, 'Ideal target domain ($\gamma_2$)', color='white', fontweight='bold')

    cax = inset_axes(
            ax[1,2],
            width="100%",      # same width as axes
            height="5%",       # thickness
            loc="lower center",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax[1,2].transAxes,
            borderpad=0,
        )
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")

    ax[0,3].imshow(obs_sampled_src[0][0]-obs_sampled_src[0][0].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[0,3].text( 5, 90, 'Masked source domain ($\gamma_1$)', color='white', fontweight='bold')

    ax[1,3].imshow(obs_sampled_tgt[0][1]-obs_sampled_tgt[0][1].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[1,3].text( 5, 90, 'Masked target domain ($\gamma_2$)', color='white', fontweight='bold')

    cax = inset_axes(
            ax[1,2],
            width="100%",      # same width as axes
            height="5%",       # thickness
            loc="lower center",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax[1,3].transAxes,
            borderpad=0,
        )
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")

    ax[0,4].imshow(obs_noisy_src[0][0]-obs_noisy_src[0][0].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[0,4].text( 5, 90, 'Noisy source domain ($\gamma_1$)', color='white', fontweight='bold')

    im = ax[1,4].imshow(obs_noisy_tgt[0][1]-obs_noisy_tgt[0][1].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[1,4].text( 5, 90, 'Noisy target domain ($\gamma_2$)', color='white', fontweight='bold')

    cax = inset_axes(
            ax[1,2],
            width="100%",      # same width as axes
            height="5%",       # thickness
            loc="lower center",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax[1,4].transAxes,
            borderpad=0,
        )
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")
    ax[0,0].imshow(obs_data['e1']-obs_data['e1'].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[0,0].text( 5, 90, 'A2744 UNCOVER ($\gamma_1$)', color='white', fontweight='bold')

    im = ax[1,0].imshow(obs_data['e2']-obs_data['e2'].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    ax[1,0].text( 5, 90, 'A2744 UNCOVER ($\gamma_2$)', color='white', fontweight='bold')

    cax = inset_axes(
            ax[1,2],
            width="100%",      # same width as axes
            height="5%",       # thickness
            loc="lower center",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax[1,0].transAxes,
            borderpad=0,
        )
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")

    if np.max(obs_cat['x']) > 1000:
        ra,dec = ra_dec_to_simulation_image_pos( obs_cat)
        obs_cat['x'] = ra
        obs_cat['y'] = dec
    image_size  = 100
    kappa_e, kappa_b = get_kappa(
        obs_cat, smooth=1.5, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )

    im = ax[0,1].imshow(kappa_e, cmap=cmap, origin='lower', vmin=-0.2,vmax=0.8)
    ax[0,1].text( 5, 90, 'A2744 $\\kappa_E$', color='white', fontweight='bold')

    ax[1,1].imshow(kappa_b, cmap=cmap, origin='lower', vmin=-0.2,vmax=0.8)
    ax[1,1].text( 5, 90, 'A2744 $\kappa_B$', color='white', fontweight='bold')

    if sl_kappa is not None:
        X, Y, k_sl, k_gal = kappa_on_stamp(sl_kappa, obs_cat)
        n_in = np.sum(k_gal >= kappa_level)
        print(f"strong-lensing kappa >= {kappa_level}: {n_in} of {len(k_gal)} catalogue galaxies "
              f"({n_in / len(k_gal):.1%}); {np.sum((k_gal >= kappa_level) & (np.abs(obs_cat['x']) < 50) & (np.abs(obs_cat['y']) < 50))} in the stamp")
        for iax in [ax[0,0], ax[1,0], ax[0,1], ax[1,1]]:
            iax.contour(X, Y, k_sl, levels=[kappa_level], colors='cyan', linewidths=1.3)
            iax.set_xlim(-0.5, 99.5)
            iax.set_ylim(-0.5, 99.5)
        ax[0,0].text( 5, 82, f'cyan: SL model $\\kappa={kappa_level:g}$', color='cyan', fontsize=8)

    cax = inset_axes(
            ax[1,2],
            width="100%",      # same width as axes
            height="5%",       # thickness
            loc="lower center",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax[1,1].transAxes,
            borderpad=0,
        )
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")


    for iax in ax.flatten():
        iax.set_xticklabels([])
        iax.set_yticklabels([])
        iax.errorbar( 15, 5,xerr=5, capsize=2, color='white')       # 10 pixels = 200 kpc
        iax.text( 15, 5, '200kpc', ha='center', va='bottom', color='white')
        iax.plot([ 80, 90], [5,5], '-', color='white')
        iax.plot([90, 90],[5, 15], '-', color='white')
        #iax.text( 90, 5, 'E', ha='left', va='center', color='white')
        #iax.text( 72, 10, 'N', ha='left', va='center', color='white')
        iax.set_xticks([])
        iax.set_yticks([])

    filename = "plots/data_examples.png" if sl_kappa is None else f"plots/data_examples_kappa{kappa_level:g}.png"
    plt.savefig(filename)

def figure2():
    # # Main figure

    cdm = 3e-3

    fig, axarr = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)

    ifilter = 'concat'


    postive_mass, mass_cut = get_mass_cut( ifilter, nsigma=2, study='harvey' )


    mass_cut = [ i for i in mass_cut]


    this_domain = [    {
            'tgt':'darkskies_obs',
            'src':'bahamas_obs'
        },
        {
            'tgt':'flamingo_obs',
            'src':'tng_obs'
        }

    ]

    fiducial = f"pickles/all_models_{ifilter}_nz_alignbest_results.pkl"

    colors = {
        'flamingo':'cyan', 
        'bahamas':'r', 
        'darkskies':'b',
        'tng':'green'}

    cdm_vals = []
    cdm_val_err = []
    npts = 10000
    xpdf = np.linspace(0.35,0.7, npts)
    marginalised_pdf = np.zeros(npts)

    for ifx, results_file in enumerate( [ fiducial ]):
        domain = this_domain[ifx]

        all_results= pkl.load(open(results_file,'rb'))

        for itgt, target in enumerate(['src','tgt']):
            all_thresholds = []

            domain_name = domain[target].split('_')[0]

            if domain_name == 'tng':
                domain_name = 'tng-cluster'

            for imodel in all_results.keys():

                seed = float(imodel.split('_')[1])                

                mass_weights = get_massfunction_weights(
                    domain[target].split('_')[0]
                )


                tgt = get_threshold_for_cross( 
                    all_results[imodel][target], 
                    mass_cut=mass_cut, 
                    integrated_mass=True,
                    function=np.mean,
                    mass_weights=mass_weights,
                    dataset=domain[target].split('_')[0],
                    ncomponents=1,
                    quiet=False)

                all_thresholds.append(tgt['thresholds'])

            all_thresholds = np.array(all_thresholds)
            means = np.mean(all_thresholds,axis=0)
            errors = return_error_in_mean( all_thresholds )

            cdm_vals.append(1-means[0])
            cdm_val_err.append(errors[0])
            if len(tgt['cross_sections']) > 1:
                tgt['cross_sections'][0] += cdm
                axarr[0].errorbar( tgt['cross_sections'], 
                           1-means,
                            errors,
                            fmt='-o', 
                            capsize=2,
                            markersize=8, color=colour_scheme[domain_name.lower()])
                axarr[0].plot( tgt['cross_sections'], 
                           1-means, 'o', label=f"{domain_name.upper()}", 
                            markersize=8, color=colour_scheme[domain_name.lower()])

                axarr[0].errorbar( cdm, 
                           1-means[0],
                            errors[0],
                            fmt='*', 
                            capsize=2,
                            markersize=15, color=colour_scheme[domain_name.lower()])


                axarr[0].plot( cdm, 
                           1-means[0],
                            '*', 
                            markersize=15, color=colour_scheme[domain_name.lower()])


                axarr[0].set_xlabel("Self-interaction cross-section [cm$^2$g$^{-1}$]", fontsize=15)




            ypdf = norm.pdf( xpdf, *(1-means[0], errors[0]))
            marginalised_pdf += ypdf
            axarr[1].plot(  xpdf, ypdf,
                             color=colour_scheme[f"{domain[target].split('_')[0].lower()}"])

            axarr[1].fill_between( xpdf,
                            ypdf,
                            np.zeros(npts),
                             color=colour_scheme[f"{domain[target].split('_')[0].lower()}"],
                           alpha=0.2, label=domain_name.upper())


            axarr[1].set_xlabel(r"$P(\mathrm{SIDM})$", fontsize=15)
            axarr[1].set_ylabel("Probability distribution", fontsize=15)

            axarr[1].set_xlim(0.475, 0.65)

    marginalised_pdf /= np.sum(marginalised_pdf)*(xpdf[1]-xpdf[0])

    axarr[1].plot(  xpdf, marginalised_pdf, lw=2, ls='--',
                             color='k', label='Marginalised CDM')

    axarr[1].fill_between( xpdf,
                    marginalised_pdf,
                    np.zeros(npts),
                     color='k', 
                   alpha=0.2)



    axarr[0].set_xscale('log')

    axarr[0].set_xlabel("Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    axarr[0].set_ylabel(r"$P(\mathrm{SIDM})$", fontsize=15)

    axarr[0].set_xlim(1e-3,1.5)
    axarr[0].set_xscale('log')


    plot_observations( "pickles/model_on_data.pkl", ifilter, ax=axarr[0], 
                      plot_args={'lw':2, 'color':'k'}, fill_args={'alpha':0.2, 'color':'k'},
                      uncertainty=[68,95])

    axarr[0].legend(loc=2,ncols=2)

    plot_observations( "pickles/model_on_data.pkl", ifilter, ax=axarr[1], 
                      plot_args={'lw':2, 'color':'k'}, fill_args={'alpha':0.1, 'color':'k'}, plotpdf=True)

    edges = [np.min(cdm_vals)-0.02, np.max(cdm_vals)+0.02]
    curly_brace(axarr[1], edges[0], edges[1], 22, 3, upward=True)
    axarr[1].text(np.mean(edges), 25, "Collisionless \n Dark Matter", ha='center', fontsize=12)       

    text_center =  1- np.mean(pkl.load(open( "pickles/model_on_data.pkl",'rb'))[1]['concat'])
    axarr[1].text(text_center, 20, "Observation \n (A2744 UNCOVER)", ha='center', fontsize=12)      

    axarr[1].grid(False)
    axarr[1].set_ylim(0, 40)
    axarr[1].set_xlim(0.4, 0.7)

    axarr[1].legend(ncols=2)

    fig.align_xlabels()

    filename = "plots/final_model_weighting_with_data.pdf"
    plt.savefig(filename)



    # ## Estimate the significance
    models, prob, noise = pkl.load(open("pickles/model_on_data.pkl","rb"))
    obsprob = 1-prob['concat']
    obsstd = return_error_in_mean( obsprob )
    margingliased_sigma = np.sqrt(np.sum( (xpdf - xpdf[ np.argmax(marginalised_pdf) ])**2*marginalised_pdf)/np.sum(marginalised_pdf))
    total_sigma = np.sqrt( margingliased_sigma**2 + obsstd**2)
    significance = (np.mean(obsprob) - xpdf[ np.argmax(marginalised_pdf) ])/total_sigma
    print(f"Signficance is {significance}")
    cumsum = 1-norm.cdf( xpdf, xpdf[ np.argmax(marginalised_pdf) ], total_sigma)
    ot_prob = cumsum[np.argmin(np.abs(xpdf-(1-np.mean(prob['concat']))))]
    print(f"One tailed prob is {100-100*ot_prob}" )
    sigmas = np.linspace(1,6,1000)
    significance = sigmas[np.argmin(np.abs(norm.sf(sigmas) - ot_prob))]
    print(f"Signficance is {significance}")
    print(f"Inter CDM variance of variance: {np.std(cdm_vals)}, and mean intra CDM variance {np.mean(cdm_val_err)}")
    print(f"Discrepancy of {obsprob.mean()-xpdf[ np.argmax(marginalised_pdf) ]}")

def summary_curve(summaries, domain, mu, sig, min_ncomp=2):
    '''
    Per model, the A2744-mass-weighted mean P(SIDM) at each simulated cross-section,
    over every cluster with at least min_ncomp components and every shape-noise draw
    (sbi/make_summaries.py output). Returns (cross_sections, (n_models, n_cross)).
    '''
    P = np.load(f"{summaries}/params_{domain}.npz")
    soft = np.load(f"{summaries}/softmax_{domain}.npy")                 # (n_models, n_noise, N)
    cs, mass, ncomp = P["cross_section"], P["log10_mass_750"], P["ncomponents"]
    sigmas = np.unique(cs)
    out = np.zeros((soft.shape[0], len(sigmas)))
    for j, s in enumerate(sigmas):
        rows = (cs == s) & (ncomp >= min_ncomp) & np.isfinite(mass)
        w = gaussian_mass_weights(mass[rows], mu, sig)
        out[:, j] = (soft[:, :, rows] * w).sum(-1).mean(1) / w.sum()
    return sigmas, out


def _pooled_tail_panel(ax, curves, mu, sig, min_ncomp, p_obs, pooled_tag, means=None):
    '''Figure 2 right panel: mass-weighted P(SIDM) of each CDM cluster (n_comp >= min_ncomp, every noise draw),
    per simulation and pooled with equal weight per simulation; shaded: the pooled fraction above A2744.'''
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sbi"))
    from cdm_pooled_tail import load as load_cdm
    x = np.linspace(0.15, 0.9, 800)
    sims = [d for d in curves]
    pooled, P, W, kmax = np.zeros_like(x), [], [], 0
    for sim in sims:
        p, w, _, _ = load_cdm(pooled_tag, sim, mu, sig, min_ncomp)
        k = gaussian_kde(p, weights=w)(x)
        pooled += k / len(sims)
        kmax = max(kmax, k.max())
        P.append(p); W.append(w / len(sims))
        ax.plot(x, k, color=COLOUR_SCHEME[sim], lw=1.2)
        ax.fill_between(x, k, color=COLOUR_SCHEME[sim], alpha=0.15, lw=0, label=sim.upper())
        if means is not None:                         # ensemble mean and its (model-correlated) error, above the curve
            m_sim, e_sim = means[sim]
            ax.errorbar(m_sim, 1.06 * k.max(), xerr=e_sim, fmt="o", color=COLOUR_SCHEME[sim], ms=5, capsize=3, lw=1.5)
    P, W = np.concatenate(P), np.concatenate(W)
    f = W[P >= p_obs].sum()
    ax.plot(x, pooled, color="k", lw=2, ls="--", label="Marginalised CDM")
    m = x >= p_obs
    ax.fill_between(x[m], pooled[m], color="k", alpha=0.35, lw=0, label=rf"$P(\mathrm{{SIDM}}) \geq$ A2744: {100 * f:.1f}\% ({norm.isf(f):.2f}$\sigma$)")
    ax.axvline(p_obs, color="k", lw=2)
    ax.text(p_obs + 0.01, 0.55, "A2744", transform=ax.get_xaxis_transform(), ha="left", va="center", fontsize=13)
    ax.set_xlabel(r"$P(\mathrm{SIDM})$ of each CDM cluster", fontsize=15)
    ax.set_ylabel("Probability distribution", fontsize=15)
    ax.set_xlim(0.15, 0.9)
    ax.set_ylim(0, 1.45 * kmax)
    ax.legend(ncols=2, loc="upper left", fontsize=9)
    print(f"pooled CDM above A2744 ({p_obs:.3f}): {100 * f:.2f}% ({norm.isf(f):.2f} sigma)")


def figure2_summaries(summaries="../sbi/summaries_ba_best_n5",
                      extra={"flamingo": "../sbi/summaries_ba_best_n5_flamingo"},
                      min_ncomp=2, tag="ba_best", show_obs=True, discrepancy=True, pooled_tag="final", sig_scale=1.0):
    '''
    Figure 2 from the current pipeline (sbi/make_summaries.py): the same CNNs, five
    shape-noise draws per cluster, clusters with >= min_ncomp components weighted to
    the A2744 mass (mass_weighting.gaussian_mass_weights), and A2744 from obs.npz.
    extra adds CDM-only simulations (e.g. FLAMINGO) to both panels and to the
    marginalised CDM distribution.
    '''
    from mass_weighting import a2744_mass_gaussian

    cdm = 3e-3
    mu, sig = a2744_mass_gaussian()
    sig *= sig_scale                                    # widen the mass weighting (e.g. 2x the measurement error)
    curves = {d: summary_curve(summaries, d, mu, sig, min_ncomp) for d in ["bahamas", "darkskies"]}
    curves.update({d: summary_curve(path, d, mu, sig, min_ncomp) for d, path in extra.items()})
    obsprob = np.load(f"{summaries}/obs.npz")["softmax"][:, 0]           # (n_models,), P(SIDM)
    obs_mean, obs_err = obsprob.mean(), return_error_in_mean(obsprob)

    fig, axarr = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    cdm_vals, cdm_val_err = [], []
    for domain, (sigmas, per_model) in curves.items():
        means, errors = per_model.mean(0), return_error_in_mean(per_model)
        colour = COLOUR_SCHEME[domain]
        cdm_vals.append(means[0])
        cdm_val_err.append(errors[0])
        x = sigmas.copy()
        x[0] += cdm
        if len(x) > 1:
            axarr[0].errorbar(x, means, errors, fmt="-o", capsize=2, markersize=8, color=colour)
            axarr[0].plot(x, means, "o", label=domain.upper(), markersize=8, color=colour)
        axarr[0].errorbar(cdm, means[0], errors[0], fmt="*", capsize=2, markersize=15, color=colour,
                          label=domain.upper() if len(x) == 1 else None)
        print(f"{domain}: CDM {means[0]:.4f} +/- {errors[0]:.4f}; "
              + ", ".join(f"{s:g}: {m:.4f}" for s, m in zip(sigmas[1:], means[1:])))

    lo = min(cdm_vals) - 0.06
    hi = max(max(cdm_vals) + 0.06, obs_mean + 3.5 * obs_err) if show_obs else max(cdm_vals) + 0.06
    # right panel: with discrepancy=True the per-cluster CDM distributions pooled over the simulations and the
    # fraction above A2744 (sbi/cdm_pooled_tail.py); the Gaussian-of-means version is then drawn off-figure
    ax_r = plt.figure().add_subplot() if discrepancy else axarr[1]
    npts = 10000
    xpdf = np.linspace(lo, hi, npts)
    marginalised_pdf = np.zeros(npts)
    for domain, m, e in zip(curves, cdm_vals, cdm_val_err):
        ypdf = norm.pdf(xpdf, m, e)
        marginalised_pdf += ypdf
        colour = COLOUR_SCHEME[domain]
        ax_r.plot(xpdf, ypdf, color=colour)
        ax_r.fill_between(xpdf, ypdf, np.zeros(npts), color=colour, alpha=0.2, label=domain.upper())
    marginalised_pdf /= np.sum(marginalised_pdf) * (xpdf[1] - xpdf[0])
    ax_r.plot(xpdf, marginalised_pdf, lw=2, ls="--", color="k", label="Marginalised CDM")
    ax_r.fill_between(xpdf, marginalised_pdf, np.zeros(npts), color="k", alpha=0.2)

    # A2744: band in the left panel, pdf in the right
    obs_pdf = norm.pdf(xpdf, obs_mean, obs_err) if show_obs else np.zeros(npts)
    xr = np.logspace(-3, 1, 100)
    if show_obs:
        axarr[0].plot(xr, np.full(100, obs_mean), lw=2, color="k")
        for ns, level in [(1, 68), (2, 95)]:
            axarr[0].fill_between(xr, obs_mean - ns * obs_err, obs_mean + ns * obs_err, alpha=0.2, color="k", lw=0)
            axarr[0].plot(xr, np.full(100, obs_mean - ns * obs_err), "--", color="k")
            axarr[0].plot(xr, np.full(100, obs_mean + ns * obs_err), "--", color="k")
        axarr[0].text(0.0012, obs_mean + 2.2 * obs_err, "A2744 UNCOVER DATA (68, 95\\%)", ha="left", va="bottom", fontsize=11)
        ax_r.plot(xpdf, obs_pdf, lw=2, color="k")
        ax_r.fill_between(xpdf, obs_pdf, np.zeros(npts), alpha=0.1, color="k")

    axarr[0].set_xscale("log")
    axarr[0].set_xlabel(r"Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    axarr[0].set_ylabel(r"$P(\mathrm{SIDM})$", fontsize=15)
    axarr[0].set_xlim(1e-3, 1.5)
    ytop = obs_mean + 2 * obs_err + 0.045 if show_obs else max(c[1].mean(0).max() for c in curves.values()) + 0.03
    axarr[0].set_ylim(min(cdm_vals) - 0.04, ytop)
    axarr[0].legend(loc=4, ncols=2)

    peak = max(np.max(marginalised_pdf), np.max(obs_pdf), max(norm.pdf(0, 0, e) for e in cdm_val_err))
    top = 1.35 * peak if show_obs else 1.8 * peak       # without A2744: room for the legend above the brace
    edges = [np.min(cdm_vals) - 0.02, np.max(cdm_vals) + 0.02]
    yb, yt = (0.78 * top, 0.87 * top) if show_obs else (1.08 * peak, 1.17 * peak)
    curly_brace(ax_r, edges[0], edges[1], yb, 0.06 * top, upward=True)
    ax_r.text(np.mean(edges), yt, "Collisionless\nDark Matter", ha="center", fontsize=12)
    if show_obs:
        ax_r.text(obs_mean, 1.05 * norm.pdf(0, 0, obs_err), "Observation\n(A2744 UNCOVER)", ha="center", fontsize=12)
    ax_r.set_xlabel(r"$P(\mathrm{SIDM})$", fontsize=15)
    ax_r.set_ylabel("Probability distribution", fontsize=15)
    ax_r.set_xlim(lo, hi)
    ax_r.set_ylim(0, top)
    ax_r.grid(False)
    ax_r.legend(ncols=2, loc=1, fontsize=8)
    if discrepancy:
        plt.close(ax_r.figure)
        _pooled_tail_panel(axarr[1], curves, mu, sig, min_ncomp, obsprob.mean(), pooled_tag,
                           means={d: (m, e) for d, m, e in zip(curves, cdm_vals, cdm_val_err)})
    fig.align_xlabels()

    filename = f"plots/final_model_weighting_with_data_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")

    # Significance, as in figure2: A2744 against the marginalised CDM distribution
    peak = xpdf[np.argmax(marginalised_pdf)]
    marg_sigma = np.sqrt(np.sum((xpdf - peak) ** 2 * marginalised_pdf) / np.sum(marginalised_pdf))
    total_sigma = np.sqrt(marg_sigma ** 2 + obs_err ** 2)
    ot_prob = norm.sf(obs_mean, peak, total_sigma)
    print(f"A2744 {obs_mean:.4f} +/- {obs_err:.4f}; marginalised CDM peak {peak:.4f}, width {marg_sigma:.4f}")
    print(f"Discrepancy {obs_mean - peak:.4f}: one-tailed p = {ot_prob:.3g} ({norm.isf(ot_prob):.2f} sigma)")
    print(f"Inter-CDM scatter {np.std(cdm_vals):.4f}, mean intra-CDM error {np.mean(cdm_val_err):.4f}")


def figure3and4():
    # # Get sensitivity plots

    obs_cat = get_obs_data( 'concat', data_dir="../data/100/a2744/")
    binned_data = bin_obs_data(obs_cat)

    obs_cat['x'] = binned_data['delta_ra']
    obs_cat['y'] = binned_data['delta_dec']
    image_size  = 100
    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_cat, smooth=2, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )

    segment_size = 4
    sim_fiducial, sim_probabilities = pkl.load(open(f"pickles/sim_senstivity_{segment_size}_select_moving_av.pkl","rb"))      
    cmap_contour = 'YlOrRd'
    cmap_image = 'viridis'
    kappa_bins = np.linspace(0.0,0.6,7)

    fig, axarr=plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.1)

    kappas = []
    contrast=3
    vmin = 1

    vmax =  contrast
    ref = None
    dmlabel = ['Collisionless','SIDM0.2','Observation']

    h=0.7
    a2744_analogue = pkl.load(open("pickles/a2744_analogue.pkl","rb"))


    for idx in range(sim_probabilities.shape[0]):
        if idx ==0:
            kappa_data = a2744_analogue['cdm']['kappa']
        else:
            kappa_data = a2744_analogue['sidm']['kappa']


        relative_prob = np.mean(sim_fiducial[:,idx,0] - sim_probabilities[idx], axis=-1)

        relative_prob /= relative_prob.std()

        ax = axarr[idx]



        snr = ax.imshow(relative_prob, 
                         origin='lower', extent=[-50,50,-50,50],
                         vmin=vmin, vmax=vmax,
                         cmap=cmap_image)
        #

        ax.contour(gaussian_filter(kappa_data,1.5), origin='lower', 
                       extent=[-50,50,-50,50], 
                   cmap=cmap_contour, 
                   levels=np.linspace(0.,1.0,20))

        divider = make_axes_locatable(ax)


        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
        ax.errorbar( 15-50, 5-50,xerr=10, capsize=2, color='white', lw=2)
        ax.text( 15-50, 5-50, '200kpc', ha='center', va='bottom', color='white', weight='bold')
        ax.plot([ 80-50, 90-50], [85-50,85-50], '-', color='white', lw=2)
        ax.plot([90-50, 90-50],[95-50, 85-50], '-', color='white', lw=2)
        ax.text( 85-50, 84-50, 'E', ha='center', va='top', color='white', weight='bold')
        ax.text( 91-50, 90-50, 'N', ha='left', va='center', color='white', weight='bold')
        ax.text( 5-50, 40, dmlabel[idx], color='white', weight='bold', fontsize=20)


    fiducial, probabilities = pkl.load(open(f"pickles/senstivity_{segment_size}_moving_av.pkl","rb"))

    obs_probs = np.mean((1-fiducial) - probabilities ,axis=-1)
    obs_probs /= obs_probs.std()

    ax = axarr[-1]
    prob = ax.imshow(obs_probs, 
                     origin='lower', extent=[-50,50,-50,50],
                     vmin=vmin,vmax=vmax,cmap=cmap_image)
    #
    kappa = ax.contourf(zoom(obs_kappa_e,2), origin='lower', 
                   extent=[-50,50,-50,50], lw=3, cmap=cmap_contour, 
               levels=kappa_bins, alpha=0.01)

    ax.contour(zoom(obs_kappa_e,2), origin='lower', 
                   extent=[-50,50,-50,50], lw=5, cmap=cmap_contour, 
               levels=kappa_bins)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.errorbar( 15-50, 5-50,xerr=10, capsize=2, color='white', lw=2, capthick=2)
    ax.text( 15-50, 5-50, '200kpc', ha='center', va='bottom', color='white', weight='bold')
    ax.plot([ 80-50, 90-50], [85-50,85-50], '-', color='white', lw=2)
    ax.plot([90-50, 90-50],[95-50, 85-50], '-', color='white', lw=2)
    ax.text( 85-50, 84-50, 'E', ha='center', va='top', color='white', weight='bold')
    ax.text( 91-50, 90-50, 'N', ha='left', va='center', color='white', weight='bold')
    #fig.patch.set_facecolor('black')
    ax.text( 5-50, 40, 'Observation', color='white', weight='bold', fontsize=20)

    fraction = 0.015
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = fig.colorbar(
        prob, 
        ax=axarr, 
        orientation='vertical',
        fraction=fraction*0.96,
        pad=0.025)
    cbar.ax.set_title('$S$')


    #cax = divider.append_axes('right', size='5%', pad=0.4)
    cbar = fig.colorbar(
        kappa, 
        ax=axarr, 
        orientation='vertical',
        fraction=fraction,
        pad=0.01)

    cbar.ax.set_title('$\kappa$')
    cbar.solids.set_alpha(1)
    fname = "plots/sensitivity_map.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")

    def sidm_sen( y ):
        return np.mean(y>0)
    def sidm_sen_std(y):
        return np.sqrt( np.mean(y>0))


    obs_data = get_obs_data('concat', data_dir="../data/100/a2744/")

    obs_cat = bin_obs_data(
        obs_data
    )
    obs_data['x'] = obs_cat['delta_ra']
    obs_data['y'] = obs_cat['delta_dec']
    image_size  = 100
    smooth=1.5
    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_data, smooth=smooth, correct_for_ngal=False, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )



    segment_size = 1
    nbins=15
    bins = np.linspace(0.,0.5,nbins)

    corr = {}
    std = {}
    ncomp = 0
    for dataset in tqdm(["bahamas_cdm","bahamas_0.1","bahamas_0.3","bahamas_1"]):
        meta, data = pkl.load(open(f"../data/100/obs/concat/{dataset}.pkl","rb"))
        meta, shear = pkl.load(open(f"../data/100/shear/{dataset}.pkl","rb"))


        fiducial, sim_probabilities = pkl.load(open(f"pickles/sim_senstivity_{segment_size}_{dataset}.pkl","rb"))    
        new_fiducial = np.median(np.median(sim_probabilities,axis=1),axis=1)

        rel_prob = np.array( [ np.mean(new_fiducial[i,:] - sim_probabilities[i], axis=-1) for i in range(sim_probabilities.shape[0]) if meta['ncomponents'][i] > ncomp])

        indexes = np.array([ i for i in  range(sim_probabilities.shape[0]) if meta['ncomponents'][i] > ncomp])

        rel_prob /= rel_prob.std()

        rel_prob = np.array([
            zoom(i, 100/rel_prob.shape[1]) for i in rel_prob])

        meta, data = pkl.load(open(f"../data/100/shear/{dataset}.pkl","rb"))

        x_bin_me = data[indexes,2,:,:].flatten()
        y_bin_me = rel_prob.flatten()

        y_bin_me = y_bin_me/np.abs(y_bin_me)
        remove = np.isfinite(y_bin_me)

        y_bin_me = y_bin_me[remove]
        x_bin_me = x_bin_me[remove]

        ycdm, x, n = binned_statistic( 
            x_bin_me, y_bin_me, bins=bins, statistic=sidm_sen)

        ycdmstd, x, n = binned_statistic( 
            x_bin_me, y_bin_me, bins=bins, statistic=sidm_sen_std)

        count, x, n = binned_statistic( 
            x_bin_me, y_bin_me, bins=bins, statistic="count")

        xc_sim = (x[1:]+x[:-1])/2.
        corr[dataset] = ycdm
        std[dataset] = ycdmstd/np.sqrt(count)

    obsfiducial, obsprobabilities = pkl.load(open(f"pickles/senstivity_2.pkl","rb"))

    obs_probs = np.mean((1-obsfiducial) - obsprobabilities ,axis=-1)
    obs_probs /= obs_probs.std()

    obs_probs = zoom(obs_probs, 100/obs_probs.shape[0])

    x_bin_me = obs_kappa_e.flatten()
    y_bin_me = obs_probs.flatten()



    y_bin_me = y_bin_me/np.abs(y_bin_me)
    keep = np.isfinite(y_bin_me) 

    y_bin_me = y_bin_me[keep]
    x_bin_me = x_bin_me[keep]

    nbins=20
    bins = np.linspace(0.,0.5,nbins)

    yobs, x, n = binned_statistic(
        x_bin_me, y_bin_me, bins=bins, statistic=sidm_sen)
    yobsstd, x, n = binned_statistic(
        x_bin_me,y_bin_me, bins=bins, statistic=sidm_sen_std)
    yobsn, x, n = binned_statistic(
        x_bin_me,y_bin_me, bins=bins, statistic='count')
    #yobs +=0.5
    yobsstd /= np.sqrt(yobsn)
    xc_obs = (x[1:]+x[:-1])/2.
    yobsstd_monte = []
    for imonte in range(1000):
        np.random.shuffle(x_bin_me)
        yobsstd_monte.append( binned_statistic(
        x_bin_me, y_bin_me, bins=bins, statistic=np.mean)[0])
    yobsstd = np.std(np.array(yobsstd_monte), axis=0)

    fig = plt.figure(figsize=(6,4))
    xc = (x[1:] + x[:-1]) / 2.
    ax = plt.gca()

    for dataset in corr.keys():
        ax.errorbar(xc_sim, corr[dataset],
                    std[dataset], fmt='o-', capsize=2,
                lw=2, label=' '.join(dataset.split('_')).upper())

    ax.errorbar( xc_obs, yobs, yobsstd, fmt='o', color='k', label="Observation", capsize=2)
    leg = ax.legend()
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1.0)
    ax.set_xlabel(r'Weak lensing convergence, $\kappa$')

    ax.fill_between(
        [0.0,1.0],
        [0.5]*2, [1]*2,
        color='c', alpha=0.2)

    ax.fill_between(
        [0.0,1.0],
        [0]*2, [0.5]*2,
        color='k', alpha=0.2)

    ax.text(0.05, 0.9, "Regions sensitive to SIDM", transform=ax.transAxes, fontsize=15)
    ax.text(0.4, 0.1, "Regions sensitive to CDM", transform=ax.transAxes, fontsize=15)
    ax.set_ylabel("Fraction of pixels that are sensitive to SIDM")
    ax.set_xlim(0,0.5)
    ax.set_ylim(0.,1)
    fname = "plots/Sensitivty_relation.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")



def figure5():
    # Particle physics inference

    ifilter = 'concat'
    models, probs = pkl.load(open("pickles/probs_for_cross_concat_nob1.pkl","rb"))
    models, probabilities, probabilities_noise =pkl.load(open("pickles/model_on_data.pkl","rb"))
    nmodels = probabilities[ifilter].shape[0]

    nmodels = len(probs['0.00'])
    all_thresholds = []
    for imodel in range(nmodels):
        #Get the X and y values for a given model
        all_thresholds.append(1.-np.array([ np.mean(probs[i][imodel]) for i in probs.keys()]))

    all_thresholds = np.array(all_thresholds)
    thresholds = np.mean(all_thresholds,axis=0)
    err = return_error_in_mean(all_thresholds)

    cross = np.array([float(i) for i in probs.keys() ])

    est = infer_sidm(
        probabilities['concat'],
        { icross:thresholds[icx] for icx, icross in enumerate(cross)},
        return_all=True)



    gp = est['gp']

    fig, axarr = plt.subplots(1,2,figsize=(8,3))

    fig.subplots_adjust(wspace=0.3)
    #####################
    handles = []
    ###PLOT 1

    #####################
    ax = axarr[0]
    handles.append(ax.errorbar(np.log10(cross),thresholds,err,fmt='ko',capsize=2,label='Simulations'))
    #ax.plot( cdm_cross[-1], cdm_thresh[-1], 'k*', ms=10)

    xplot=np.linspace(-5.4,1,100)
    gp_prediction, std = gp.predict(xplot.reshape(-1, 1), return_std=True)

    ax.plot( xplot, gp_prediction,'k--')
    handles.append(ax.fill_between( xplot, gp_prediction-std, gp_prediction+std, alpha=0.1, color='k',label='Gaussian Process'))
    ax.plot( xplot, gp_prediction-std, color='k')
    ax.plot( xplot, gp_prediction+std, color='k')

    ax.set_ylim(0.45,0.75)
    ax.set_xlim(-2.5,0.2)
    ax.set_xlabel(
        r'$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$', fontsize=15
    )
    ax.set_ylabel(r"Mean $P(\mathrm{SIDM})$", fontsize=15)

    #####################

    ###PLOT 2

    ####################””#
    ax = axarr[1]

    xpdf = np.linspace(0.4,0.8,1000)
    thresh = 1-probabilities['concat']

    ax = axarr[0].twiny()

    ypdf = norm.pdf(xpdf, 1-np.mean(probabilities['concat']), return_error_in_mean(probabilities['concat']))
    ypdf /= np.sum(ypdf)*(xpdf[1]-xpdf[0])

    ax.plot(ypdf[::-1], xpdf[::-1], 'k-')
    handles.append(ax.fill_between(ypdf, xpdf, alpha=0.4, label=f'A2744 UNCOVER', color='b'))
    ax.set_xlim(0, ypdf.max()*1.1)
    ax.set_ylim(0.45, 0.75)
    ax.set_xticklabels([])
    ax.grid(False)
    ax.legend(handles=handles,loc=2)
    #####################


    ###PLOT 3

    #####################



    xpdf = np.linspace(-10,10,10000)
    ypdf = norm.pdf(
        xpdf, 
        np.mean(est['estimates']), 
        return_error_in_mean(np.array(est['estimates']))
    )

    log_max_like = np.sum(ypdf*xpdf)/np.sum(ypdf)
    log_max_like = xpdf[np.argmax(ypdf)]

    cdf = cumulative_trapezoid(
        ypdf,
        xpdf,
        initial=0
    )

    cdf /= cdf[-1]


    for ivx, value in enumerate([0.34, 0.475,0.495]):
        low_value = cdf[ np.argmax(ypdf)]-value #cdf[ np.argmax(ypdf)] - 0.34
        hig_value = cdf[ np.argmax(ypdf)]+value #cdf[ np.argmax(ypdf)] + 0.34


        low = np.interp(
            low_value,
            cdf,
            xpdf
        )
        high = np.interp(
            hig_value,
            cdf,
            xpdf
        )
        ax = axarr[1]

        ax.fill_between(
            xpdf[ (xpdf>low) & (xpdf<high)], 
            ypdf[ (xpdf>low) & (xpdf<high)], 
            alpha=0.4, color='blue')

        log_error = np.array([log_max_like - low, high - log_max_like])
        if ivx ==0:
            log_like_one_sigma = log_error.copy()
        if ivx == 0:
            max_like = 10**log_max_like

            error = np.array([ 
                max_like - 10**(log_max_like - log_error[0]),
                10**(log_max_like + log_error[1]) - max_like,  
            ])
            ax.plot([log_max_like]*2,[0,np.max(ypdf)],'k-')


        ax.plot([low]*2,[0,ypdf[ np.argmin(np.abs(xpdf-low))] ],'k--')
        ax.plot([high]*2,[0,ypdf[ np.argmin(np.abs(xpdf-high))]],'k--')


    ax.plot(xpdf, ypdf,'k-',label=f"$\sigma_{{\\rm DM}}/m={max_like:0.2f}^{{+{error[1]:0.2f}}}_{{-{error[0]:0.2f}}}$cm$^2$g$^{{-1}}$")

    ax.set_xlabel(
        r'$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$', fontsize=15
    )
    ax.set_ylabel("Posterior likelihood", fontsize=15)
    ax.legend(loc=1)
    ax.set_xlim(-2.5, 1.5)
    ax.set_xticks(np.linspace(-2.5,1,6))
    ax.set_ylim(0, 1.2)

    fig.align_ylabels()
    filename = "plots/output_to_model.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))  
    print(f"${max_like:.2f}^{{+{error[1]:.2f}}}_{{-{error[0]:.2f}}}")

    
    return log_max_like, log_like_one_sigma

def figure6_summaries(tag=PAPER_TAG, min_ncomp=2, merger=True):
    '''Figure 6 (particle physics) at the maximum likelihood of the paper GP posterior with its 16-84% interval.'''
    from mass_weighting import a2744_mass_gaussian
    mu, sig = a2744_mass_gaussian()
    paths = paper_paths(tag)
    sigmas, curve = combined_curve(paths["summaries"], mu, sig, min_ncomp)
    p_obs = np.load(f"{paths['summaries']}/obs.npz")["softmax"][:, 0]
    gp, x, post, (med, lo, hi) = sigma_posterior(sigmas, curve, p_obs)
    ml = x[np.argmax(post)]
    figure6(ml, np.array([ml - lo, hi - ml]), merger=merger)


def figure6( log_max_like, log_error, merger=True):
    
    max_like = 10**log_max_like

    error = np.array([ 
        max_like - 10**(log_max_like - log_error[0]),
        10**(log_max_like + log_error[1]) - max_like,  
    ])
            
    markersize = 12

    fig = plt.figure(figsize=(7,4.5))
    ax = plt.gca()


    x = np.logspace(0,5)

    ax.plot( x, sigma_vd( x, 120,40),  lw=2, color='k', label='Nadler et al 2023')
    ax.plot( x, sigma_vd( x, 3,500),  '--', lw=2, color='k', label='BAHAMAS-VD')


    ax.set_ylabel('$\sigma_{\\rm DM}/m$ $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$', fontsize=15, wrap=True)
    ax.set_xlabel('Relative velocity of dark matter particles in a halo [km/s]', fontsize=15, labelpad=-2)

    ax.loglog()
    ax.set_ylim(1e-2,4e2)
    ax.set_xlim(10,1e4)

    def v_to_m(v):
        m = mass_from_velocity(v * units.km / units.s, redshift=0.3)
        return m

    def m_to_v(m):
        # inverse of mass_from_velocity (using dummy v³ relation)
        v = v_from_m(m)
        return v

    bottom_ticks = ax.get_xticks()                   # e.g., [1, 10, 100, ...]

    def sci_notation(x, pos):
        if x == 0:
            return "0"
        exp = np.log10(x)
        #pre = 10**(np.log10(x) - exp)

        return rf"$10^{{{exp:.01f}}}$"  # e.g., 10^10, 10^11


    ax.tick_params(axis='both', labelsize=15)

    plot_constraints( ax=ax, select_these=['Dwarf_correa21','cluster_harvey19','sagunski'],
                    labels={'Dwarf_correa21':'Correa (2021)','cluster_harvey19':'Harvey et al (2019)', 'sagunski':'Sagunski et al (2020)'})



    velocity_disp = [522, 633]

    velocity_merg = [2000,2000]
    ax.plot( np.mean(velocity_disp), max_like, '*',
                markersize=20, color='red', lw=2, label='This Work (Velocity Disp)')

    if merger:
        ax.plot( np.mean(velocity_merg), max_like, '*',
                markersize=20, color='orange', lw=2, label='This Work (Merger Velocity)')

    ax.errorbar( np.mean(velocity_disp), max_like, yerr=error[:,None], 
                markersize=20, color='red', fmt='*', capsize=4, lw=2)

    if merger:
        ax.errorbar( np.mean(velocity_merg), max_like, yerr=error[:,None], 
                markersize=20, color='orange', fmt='*', capsize=4, lw=2)
    print(f"$\sigma_{{\\rm DM}}/m={max_like.real:0.2f}_{{-{error[0].real:0.2f}}}^{{+{error[1].real:0.2f}}}$cm$^2$g^{{-1}}")
    # get handles
    ax.legend(loc=1, fontsize=9, markerscale=0.6, handlelength=1.6, labelspacing=0.3, borderpad=0.4)   # small enough to clear the upper error bars
    fname = "plots/particle_physics.pdf"
    plt.savefig(fname)
    plt.savefig(fname.replace(".pdf", ".png"), dpi=200)

# # Appendix plots

def bmode_check(nmonte=200, seed=0, image_size=100):
    '''
    For the B-mode panel of Figure A1 (as notebooks/prepare_uncovers_data.ipynb, which saved
    pickles/bmode_check.pkl): the Kaiser-Squires B-mode of the A2744 catalogue (SHEAR_CUTS, strong-lensing
    galaxies dropped, as the CNN map) and the E-mode of nmonte copies with every galaxy's orientation
    randomised (the noise floor). Returns both flattened.
    '''
    cat = get_obs_data('concat', data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=True)
    binned = bin_obs_data(cat)
    cat['x'], cat['y'] = binned['delta_ra'], binned['delta_dec']
    extent = [-image_size//2, image_size//2, -image_size//2, image_size//2]
    _, kappa_b = get_kappa(cat, smooth=0, correct_for_ngal=False, extent=extent)
    amp = np.hypot(cat['gamma1'], cat['gamma2'])
    rng = np.random.default_rng(seed)
    rotated = []
    for _ in tqdm(range(nmonte), desc="rotated B-mode noise"):
        theta = rng.uniform(0, np.pi, len(amp))
        new = cat.copy()
        new['gamma1'], new['gamma2'] = amp*np.cos(2*theta), amp*np.sin(2*theta)
        rotated.append(get_kappa(new, smooth=0, correct_for_ngal=False, extent=extent)[0])
    return np.array(rotated).ravel(), np.asarray(kappa_b).ravel()


def figureA1(profile_npz="../sbi/results/main_halo_profile/main_halo_sims_jauzac_core.npz"):  # matched to the Jauzac main-halo (core) mass
    
    nrow = 2 
    ncol = 3
    ifilter = 'concat'
    fig =plt.figure(figsize=(15,10))
    nbins=50
    
    ####################################
    ##### PLOT 1 Intrinsic ell #####
    ####################################
    ax = plt.subplot(nrow,ncol,1)

    obs_data = get_obs_data(
            ifilter, data_dir="../data/100/a2744/"
    )

    x = np.linspace(0,1,nbins)
    data = np.sqrt(obs_data['e1']**2+obs_data['e2']**2)

    ydata, x = np.histogram( data, bins=x,density=True)



    ax.stairs(
        ydata,
        x,
        facecolor='g',
        alpha=0.3,
        fill=True,
        linewidth=2,
        color="g",
        label='Data'
    )
    gaussian_assumption = np.sqrt( 
        norm.rvs( *norm.fit(obs_data['e1']), 2*obs_data['e1'].shape[0])**2 +
        norm.rvs( *norm.fit(obs_data['e2']), 2*obs_data['e1'].shape[0])**2 
    )

    y, x = np.histogram(gaussian_assumption,bins=x,density=True)

    xc = (x[1:] + x[:-1])/2.

    ervs = chi.rvs( *chi.fit(data),  int(data.shape[0]))

    theta = np.random.uniform(0,np.pi, int(data.shape[0]))
    e1 = ervs*np.cos(2.*theta)
    e2 = ervs*np.sin(2.*theta)
    g = np.sqrt(e1**2+e2**2)
    y, x = np.histogram( e1*2/(1+g**2), density=True, bins=np.linspace(-1,1,nbins) )

    obsshear = np.sqrt(obs_data['e1']**2+obs_data['e2']**2)

    ax.plot( xc, chi.pdf(xc, *chi.fit(data)), color='r',label='Chi',lw=2)
    ax.plot( xc, norm.pdf(xc, *norm.fit(data)), color='b',label='Gaussian',lw=2)



    ax.set_ylabel("Probability distribution",fontsize=15)


    ax.set_xlabel("Absolute ellipticity",fontsize=15)
    ax.set_xlim(0,1.2)
    ax.legend(fontsize=15)

    ####################################
    ##### PLOT 3 Source redshift #####
    ####################################


    ax = plt.subplot(nrow,ncol,2)
    ax.set_ylabel("Probability distribution",fontsize=15)

    obs_data = get_obs_data(
                ifilter, data_dir="../data/100/a2744/", photoz=True, remove_members=True)

    UNCOVER_ERROR=0.06
    zall = np.asarray(obs_data['redshift'])
    vals, counts = np.unique(zall, return_counts=True)
    zpad = vals[np.argmax(counts)]                            # the value given to every galaxy without a photo-z
    print(f"n(z): {np.sum(zall == zpad)} of {len(zall)} galaxies padded at z = {zpad:.4f}, left out")
    kde = gaussian_kde(zall[zall != zpad], UNCOVER_ERROR)
    xpdf = np.linspace(0,10,1000)
    ypdf =  kde.pdf(xpdf)



    ax.plot(xpdf,
            ypdf, color='k', label='n(z)'
            )
    ax.fill_between(
        xpdf, ypdf, alpha=0.6, color='k'
    )
    ax.plot(
        [args.default_zs]*2,
        [0, 1.05 * ypdf.max()],
        'k--', label=f"$\langle z\\rangle={args.default_zs:0.2f}$")
    ax.legend(fontsize=15)
    ax.set_xlabel("Galaxy source redshift", fontsize=15)
    ax.set_ylim(0, 1.05 * ypdf.max())
    ax.set_xlim(0,10)

    ####################################
    ##### PLOT 4 Susceptibility tensor #####
    ####################################

    ax = plt.subplot(nrow,ncol,3)


    def std_err( x ):
        return np.std(x)/np.sqrt(x.shape[0])

    from shear_susceptibility import fit_g1
    from scipy.stats import binned_statistic_2d as _bs2
    fid_cut = {'size_cut': [2, 200], 'signal_noise_cut': 5, 'stat_type': 'snr', 'mag_cut': [0, 30], 'verbose': False,
               'RS_info': False}
    S, R, G, M = [], [], [], []
    for ifilter in ['f115w', 'f150w']:            # both filters, each with its own G1(size, S/N) fit
        cat = fits.open(f"../data/100/a2744/abell2744clu-grizli-v5.4-{ifilter}-clear_drc_sci_clean.shears")[1].data
        calc_shear(cat, "test.fits", **{**fid_cut, **SHEAR_CUTS[ifilter]})
        data = fits.open("test.fits")[1].data
        model, _ = fit_g1(data, verbose=False)
        ok = np.isfinite(data['g1_gal'])
        S.append(data['snr'][ok]); R.append(data['gal_size'][ok]); G.append(data['g1_gal'][ok])
        M.append(model(data['gal_size'][ok], data['snr'][ok]))
    S, R, G, M = (np.concatenate(v) for v in (S, R, G, M))
    xb = np.logspace(np.log10(np.percentile(S, 0.5)), np.log10(np.percentile(S, 99.5)), 16)
    yb = np.logspace(np.log10(np.percentile(R, 0.5)), np.log10(np.percentile(R, 99.5)), 13)
    g_data = _bs2(S, R, G, 'mean', bins=[xb, yb])[0]
    n_data = _bs2(S, R, G, 'count', bins=[xb, yb])[0]
    g_mod = _bs2(S, R, M, 'mean', bins=[xb, yb])[0]
    g_data[n_data < 20] = np.nan
    im = ax.pcolormesh(xb, yb, g_data.T, cmap='viridis', vmin=1.1, vmax=1.5, shading='flat')
    xc, yc = np.sqrt(xb[1:] * xb[:-1]), np.sqrt(yb[1:] * yb[:-1])
    cs = ax.contour(xc, yc, np.where(n_data >= 20, g_mod, np.nan).T, levels=np.arange(1.15, 1.5, 0.05), colors='w', linewidths=1.2)
    ax.clabel(cs, fmt='%.2f', fontsize=9)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Galaxy signal-to-noise', fontsize=15)
    ax.set_ylabel('Galaxy size (pixels)', fontsize=15)
    plt.colorbar(im, ax=ax, label=r'Lensing susceptibility, $G_1$')


    ####################################
    ##### PLOT 5 B MODES #####
    ####################################

    ax = plt.subplot(nrow,ncol,4)


    stacked_rotated_modes, obs_kappa_b = bmode_check()
    bins = np.linspace(-1,1,50)

    y_noise_floor, x =np.histogram(stacked_rotated_modes,bins=bins, density=True)
    y_b_modes, x =np.histogram(obs_kappa_b,bins=bins, density=True)

    xc = (x[1:] + x[:-1])/2.

    ax.stairs(
        y_noise_floor,
        x,
        fill=False,
        linewidth=2,
        color="k",
        ls='--',
        label='Sys-free B-Modes'
    )

    ax.stairs(
        y_b_modes,
        x,
        fill='red',
        alpha=0.3,
        linewidth=2,
        color="k",
         label='Measured B-Modes'
    )

    residual_b = np.std(stacked_rotated_modes.flatten()) - np.std(obs_kappa_b.flatten())
    ax.text(0.1, 0.69, f"Residual B-mode={residual_b:0.3f}", 
            bbox=dict(
            facecolor="white",
            edgecolor="lightgrey",
            boxstyle="square,pad=0.2",   # or "round,pad=0.3"
            linewidth=1,
        ),
            transform=ax.transAxes, ha='left', fontsize=15)
    ax.set_xlabel(r"Convergence, $\kappa$",fontsize=15)
    ax.legend(loc=1,fontsize=15)
    ax.set_ylim(0,6)
    ax.set_ylabel("Probability distribution",fontsize=15)

    ####################################
    ##### PLOT 5 CLUSTER-MEMBER LEAK #####
    ####################################
    # photo-z member leak against the weak-lensing (KS) convergence (sbi/leakage_vs_kappa.py)
    ax = plt.subplot(nrow,ncol,5)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sbi"))
    from leakage_vs_kappa import leakage, draw_leakage
    leak_cat = get_obs_data('concat', data_dir="../data/100/a2744/", photoz=True, cuts=SHEAR_CUTS)
    L = leakage(leak_cat, "ks", nbins=12)
    print(f"member leak: f_cl = ({L['c']:.3f} +- {L['c_err']:.3f}) kappa")
    draw_leakage(ax, L, fontsize=15, legend_size=9)

    ####################################
    ##### PLOT 6 MAIN-HALO PROFILE #####
    ####################################
    # from sbi/main_halo_lenstool.py --no-lenstool --sims (mass-matched BAHAMAS, n_comp >= 2)
    ax = plt.subplot(nrow,ncol,6)
    prof = np.load(profile_npz)
    ax.axhline(0, color="k", lw=1.6)
    for key, col, lab in (("sim_0", "k", "BAHAMAS CDM"), ("sim_1", "#D55E00", r"BAHAMAS $\sigma/m=1$ cm$^2$g$^{-1}$")):
        if key in prof:
            q16, q50, q84 = prof[key]
            ax.fill_between(prof["mid"], q16, q84, color=col, alpha=0.12, lw=0)
            ax.plot(prof["mid"], q50, "-", color=col, lw=2, label=lab)
    ax.errorbar(prof["rc"], prof["t"], prof["te"], fmt="o", color="#0072B2", capsize=2, label=r"$\langle\chi_t\rangle$")
    ax.errorbar(prof["rc"] * 1.03, prof["xx"], prof["xe"], fmt="s", color="#CC79A7", capsize=2, ms=4, mfc="none",
                label=r"$\langle\chi_\times\rangle$")
    ax.set_xscale("log")
    rmax = float(prof["rmax"])
    ax.set_xlim(60, rmax)
    ticks = [t for t in (60, 100, 200, 400, 800) if t <= rmax]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Radius [kpc]", fontsize=15)
    ax.set_ylabel(r"Mean galaxy distortion, $\chi$", fontsize=15)
    ax.legend(fontsize=11)

    filename = "plots/intrinsic_ell.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=150)

def figureA2():
    estimates = pkl.load(open("pickles/how_correlated.pkl","rb"))
    y_true = np.zeros(estimates[0].shape[1])+np.mean(estimates[0][:,:,0])
    predictions = estimates[0][:,:,0]

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------
    # predictions.shape = (n_models, n_test_samples)
    # y_true.shape      = (n_test_samples,)

    # Example:
    # predictions = np.load("predictions.npy")
    # y_true = np.load("y_true.npy")

    n_models, n_test = predictions.shape

    # ------------------------------------------------------------
    # Calculate errors
    # ------------------------------------------------------------
    errors = predictions - y_true[None, :]

    # ------------------------------------------------------------
    # Measure ensemble error as a function of N
    # ------------------------------------------------------------
    #
    # Here we take the first N models.
    # Better: randomly select subsets of N models and average
    # over many subsets (see below).
    # ------------------------------------------------------------

    Ns = np.arange(1, n_models + 1)

    rmse = []

    for N in Ns:
        nran = []
        for i in range(1000):
            rnadin = np.random.choice(np.arange(n_models), N, replace=False)
            ensemble_prediction = np.mean(predictions[rnadin], axis=0)
            ensemble_error = (ensemble_prediction - y_true)/np.std(errors,axis=0)
            nran.append(np.sqrt(np.mean(ensemble_error**2)))
        rmse.append(np.mean(nran))

    rmse = np.array(rmse)

    # ------------------------------------------------------------
    # Fit a power law RMSE ~ N^alpha
    # ------------------------------------------------------------

    def correlated_rmse(N, sigma, rho):
        """
        Expected ensemble RMSE for models with
        individual error scale sigma and
        pairwise error correlation rho.
        """
        return sigma * np.sqrt((1 - rho) / N + rho)

    def fit_power( x, a, b ):
        return x**b

    # ------------------------------------------------------------
    # Your measured ensemble RMSE
    # ------------------------------------------------------------

    # Ns          = array of ensemble sizes
    # mean_rmse   = measured mean RMSE for each N
    # std_rmse    = uncertainty / scatter in RMSE measurements


    # ------------------------------------------------------------
    # Fit correlated-error model
    # ------------------------------------------------------------

    popt_power, pcov_power = curve_fit(
        fit_power,
        Ns,
        rmse,
    )

    A, alpha = popt_power

    A_err, alpa_err = np.sqrt(np.diag(pcov_power))

    print(f"A = {A:.4f} +/- {A_err:.4f}")
    print(f"alpha   = {alpha:.4f} +/- {alpa_err:.4f}")

    print(f"\nMeasured scaling:")
    print(f"RMSE ~ {A:.4f} * N^{alpha:.3f}")

    popt_corr, pcov_corr = curve_fit(
        correlated_rmse,
        Ns,
        rmse,
    )

    sigma_fit, rho_fit  = popt_corr
    sigma_err, rho_err,  = np.sqrt(np.diag(pcov_corr))

    print(f"rho = {rho_fit:.4f} +/- {rho_err:.4f}")
    print(f"sigma = {sigma_fit:.4f} +/- {sigma_err:.4f}")


    # ------------------------------------------------------------
    # Plot measured scaling
    # ------------------------------------------------------------

    plt.figure(figsize=(5, 4))

    plt.plot(Ns, rmse, "o", label="Measured")

    plt.plot(Ns, fit_power(Ns, *popt_power), "-",  lw=2,
             label=f"Fitted Power law ($N^{{{alpha:0.2f}}}$)")
    plt.plot(Ns, correlated_rmse(Ns, *popt_corr), "-", lw=2, 
             label=f"Fitted correlation law $\\rho={rho_fit:0.2f}$")


    # Independent-model prediction for comparison
    independent_rmse = sigma_fit / np.sqrt(Ns)

    plt.plot(
        Ns,
        independent_rmse,
        "--",
        label=r"Independent models ($N^{-0.5}$)"
    )

    plt.xlabel("Number of models N",fontsize=15)
    plt.ylabel("Ensemble RMSE / standard deviation",fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/correlated_models.pdf")



def figureA3():


    latent = pkl.load(open("pickles/latent.pkl","rb"))
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    nmodels = len(latent['latent_space']) - 1
    nshow = 4

    sidm_list = [0.3]

    fig = plt.figure(figsize=(7,5))
    fig.subplots_adjust(hspace=0.01,wspace=0.01)

    for isx, sidm in enumerate(sidm_list):
        for imx, imodel in enumerate(range(nshow)):
            ax = plt.subplot(2,2,imx+1)

            A_latent =  latent['latent_space'][imodel+1][1]

            B_latent =  latent['latent_space'][imodel+1][0]
  
            if len(A_latent) !=0:
                fit = reducer.fit( A_latent )
                embedding_a = fit.transform(A_latent)

                center = [
                    np.mean(embedding_a[:, 0]),
                    np.mean(embedding_a[:, 1])
                ]

                density_a =  get_density(
                    embedding_a[:, 0]-center[0], embedding_a[:, 1]-center[1]

                )
                ax.contour(
                    density_a[0], density_a[1], density_a[2], cmap='Oranges', label='DARKSKIES'
                )  
            else:
                fit = reducer.fit( B_latent )

            if len(B_latent) !=0:


                embedding_b = fit.transform(B_latent)

                density_b =  get_density(
                    embedding_b[:, 0]-center[0], embedding_b[:, 1]-center[1]

                )
                ax.contour(
                    density_b[0], density_b[1], density_b[2], cmap='Blues', label='BAHAMAS'
                )  


            embedding = fit.transform(latent['latent_space'][0][imodel].detach().numpy()[None,:])
            ax.plot(
                embedding[:, 0]-center[0],
                embedding[:, 1]-center[1],
                'y*',markersize=10
            )


            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if (imx == 0) or (imx == 2):
                ax.set_ylabel("UMAP-2")

            ax.set_xlabel("UMAP-1")
            ax.text(0.1,0.9,f"Model {1+imx}", transform=ax.transAxes, fontsize=15)
    fname="plots/latent_overlap.pdf"
    plt.savefig(fname)

    ########################################
    ##### Latent space summary distance
    ########################################
    

    latent_space_dist_name = "pickles/obs_latent_distance.pkl"

    obs_distance = pkl.load(open(latent_space_dist_name,'rb'))
    nmodels = len(obs_distance[1][0.0])
    plt.figure(figsize=(5,4))

    ax = plt.gca()

    obs_distance = pkl.load(open(latent_space_dist_name,"rb"))

    color=['blue','orange']
    domain = ['BAHAMAS','DARKSKIES']

    for tgt, iobs_dist in enumerate(obs_distance):

        cross = np.array(list(iobs_dist.keys()))
        cross[ cross == 0] = 1e-3

        av_distances = np.array([np.median(iobs_dist[i]) for i in iobs_dist.keys()])
        err = np.array([np.quantile(iobs_dist[i],[0.84,0.16]) for i in iobs_dist.keys()])/nmodels**(0.5)
        ax.errorbar(cross,
                    av_distances ,
                    err.T,
                    color=color[tgt], 
                    fmt='-o', capsize=2,
                    label=domain[tgt]
                   )


    ax.legend()
    ax.plot([5e-4,2],[1,1],'k--')
    ax.text(0.001,1.0, 'Simulation Mean', ha='left', va='bottom', fontsize=15)
    ax.set_xlim(5e-4,2)
    ax.set_xscale('log')
    ax.set_ylabel("Normalised latent space distance", fontsize=15)
    ax.set_xlabel("Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    fname="plots/goodness_of_fit.pdf"
    plt.savefig(fname)


def figureA4(log_max_like,log_error):

    alpha =0.7
    text_ypt=1
    ylim=[-4,2]
    limits = [1.0,0.95,0.9,0.8,0.7]
    fid, ngal_results    = pkl.load(open("pickles/all_models_concat_ngal_dep_results.pkl", "rb"))

    models, probabilities, probabilities_noise, ngalaxies = pkl.load( open("pickles/model_on_data_galaxy_selection.pkl","rb"))
    ngalkeys = np.array(list(ngal_results.keys()))

    all_mag_low_cuts = np.linspace(21,24,11)
    all_mag_cuts = np.linspace(28,26,11)
    all_siz_cuts = np.linspace(2,3,11)

    fig, ax = plt.subplots(1,4,figsize=(16,3))
    fig.subplots_adjust(wspace=0.1)
    fmt = {'capsize':2,'color':'k','fmt':'o-'}

    central_cross_value = log_max_like
    central_cross_error = log_error[:,None] #np.array([0.58,0.69])[:,None]


    #### FIRST PLOT OF MAGNITUDE LIMISIT ####
    cross = []
    error = []
    ngal = ngalaxies/ngalaxies[0]


    for itest in range(11):

        ngalkey= ngalkeys[ np.argmin(np.abs(ngalkeys.astype(float)-ngal[itest]))]

        cross_section, results_mean, results_error = get_probs_from_results( ngal_results[ngalkey] )

        probs = { cross_section[i]:j for i,j in enumerate(results_mean) }

        data = probabilities_noise['concat'][:,itest]

        bias_sidm, bias_error = infer_sidm(
            data,
            probs,
        )

        cross.append(bias_sidm)
        error.append(bias_error)



    ax[0].errorbar( all_mag_low_cuts, cross-cross[0]+central_cross_value, np.array(error).T, **fmt)

    for limit in limits:
        if np.sum( ngal[:11] < limit) == 0:
            continue

        dx = (all_mag_low_cuts[1] - all_mag_low_cuts[0])/2.
        x = np.linspace( 
            all_mag_low_cuts[ ngal[:11] <= limit][0]-dx,
            all_mag_low_cuts[ ngal[:11] <= limit][-1]+dx*3,
            100
        )
        y_low =np.full_like(x, ylim[0])
        y_hi = np.full_like(x, ylim[1])

        ax[0].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[0].text(
            all_mag_low_cuts[ ngal[:11]  < limit][0]-dx, text_ypt, f"{limit*100}\%", va='center', ha='left', rotation=90)

        ax[0].set_xlim(
            all_mag_low_cuts[0]-dx,
            all_mag_low_cuts[-1] +dx/2.
        )



    cross = []
    error = []
    for itest in range(11,22):

        ngalkey= ngalkeys[ np.argmin(np.abs(ngalkeys.astype(float)-ngal[itest]))]

        cross_section, results_mean, results_error = get_probs_from_results( ngal_results[ngalkey] )

        probs = { cross_section[i]:j for i,j in enumerate(results_mean) }

        data = probabilities_noise['concat'][:,itest]

        bias_sidm, bias_error = infer_sidm(
            data,
            probs,
        )

        cross.append(bias_sidm)
        error.append(bias_error)
    ax[1].errorbar( all_mag_cuts, cross-cross[0]+central_cross_value, np.array(error).T, **fmt)


    for limit in limits:
        if np.sum( ngal[11:22] < limit) == 0:
            continue

        dx = (all_mag_cuts[1] - all_mag_cuts[0])/2.
        x = np.linspace( 
            all_mag_cuts[ ngal[11:22] <= limit][0]-dx,
            all_mag_cuts[ ngal[11:22] <= limit][-1]+dx*3,
            100
        )
        y_low =np.full_like(x, ylim[0])
        y_hi = np.full_like(x, ylim[1])


        ax[1].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[1].text(
            all_mag_cuts[ ngal[11:22] < limit][0]-dx/2., text_ypt, f"{limit*100}\%", va='center', ha='left', rotation=90)

        ax[1].set_xlim(
            all_mag_cuts[0]-dx,
            all_mag_cuts[-1] +dx/2.
        )


    ###### FINAL PLOT
    cross = []
    error = []

    for itest in range(22,33):

        ngalkey= ngalkeys[ np.argmin(np.abs(ngalkeys.astype(float)-ngal[itest]))]

        cross_section, results_mean, results_error = get_probs_from_results( ngal_results[ngalkey] )

        probs = { cross_section[i]:j for i,j in enumerate(results_mean) }

        data = probabilities_noise['concat'][:,itest]

        bias_sidm, bias_error = infer_sidm(
            data,
            probs,
        )

        cross.append(bias_sidm)
        error.append(bias_error)

    ax[2].errorbar( all_siz_cuts, cross-cross[0]+central_cross_value, np.array(error).T, **fmt)

    for limit in limits:
        if np.sum( ngal[22:] < limit) == 0:
            continue

        dx = (all_siz_cuts[1] - all_siz_cuts[0])/2.
        x = np.linspace( 
            all_siz_cuts[ ngal[22:] <= limit][0]-dx,
            all_siz_cuts[ ngal[22:] <= limit][-1]+dx*3,
            100
        )
        y_low =np.full_like(x, ylim[0])
        y_hi = np.full_like(x, ylim[1])

        ax[2].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[2].text(
            all_siz_cuts[ ngal[22:] < limit][0]-dx, text_ypt, f"{limit*100}\%", va='center', ha='left', rotation=90)

        ax[2].set_xlim(
            2.25,
            all_siz_cuts[-1] +dx
        )
        ax[2].set_xticks(np.linspace(2.3,3,8))

    ax[0].set_xlabel("Lower magnitude cut", fontsize=15)

    ax[1].set_xlabel("Upper magnitude cut", fontsize=15)

    ax[2].set_xlabel("Size cut", fontsize=15)


    ax[0].set_ylabel(r'$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$', fontsize=15
    )
    for iax in ax:
        iax.set_ylim(ylim[0],ylim[1])
        iax.plot( [iax.get_xlim()[0], iax.get_xlim()[1]], [-3,-3],'k--')
        iax.text( 0.05, 2/8, "CDM Limit", va="center", transform=iax.transAxes, fontsize=15)
    ##########################
    #########################D
    # DATA PIXELISATION
    #########################
    output_name = f"pickles/resolution_dep_results.pkl"

    fiducial, resolution_results = pkl.load(open(output_name,'rb'))

    models, probabilities, probabilities_noise = pkl.load( open("pickles/model_on_data_pixelisation.pkl","rb") )

    cross = []
    error = []

    cross_sections, results_mean, results_error = get_probs_from_results( fiducial )
    fid_probs = { cross_sections[i]:j for i,j in enumerate(results_mean) }  
    
    data = probabilities_noise['concat'][:,0]

    fid_sidm, bias_error = infer_sidm(
            data,
            fid_probs,
        )

    for ipx, ipixel in enumerate(resolution_results.keys()):

        data = probabilities_noise['concat'][:,ipx]

        cross_sections, results_mean, results_error = get_probs_from_results( resolution_results[ipixel] )

        probs = { cross_sections[i]:j-results_mean[0]+fid_probs[0.0] for i,j in enumerate(results_mean) }  

        bias_sidm, bias_error = infer_sidm(
            data,
            probs,
        )

        cross.append(bias_sidm)
        error.append(bias_error)

    ax[3] = plt.gca()
    ax[3].errorbar(resolution_results.keys(),cross, np.array(error).T, capsize=2, fmt='o-', color='k')
    ax[3].set_xlabel('Map resolution [kpc]', fontsize=15)

    ax[3].set_xlim(0,220)
    ax[3].plot( [ax[3] .get_xlim()[0], ax[3] .get_xlim()[1]], [-3,-3],'k--')
    fig.align_xlabels()

    fname='plots/galaxy_selection.pdf'
    plt.savefig(fname)


def figureA5(log_max_like, log_error):
    # ## Model systematics

    systematics = {
                'cluster_member_contamination': f"pickles/cluster_contamination_concat.pkl",
        'shape_measurement_bias':"pickles/shape_measurement_concat_test.pkl",
         'source_redshift_bias':"pickles/all_models_concat_nzbias_results.pkl",
            'projection_depth':"pickles/flamingo_dz_results.pkl"
    }   
    systematics_thresh= {
            'mass_dependency': "pickles/mass_data_zs_dependence.pkl",
        'merging_scenario':"pickles/component_results.pkl"
    }


    fig = plt.figure(figsize=(7,4))
    ax = plt.gca()
    corr = 2.8
    systematic_names = []

    capsize = 6

    models, probabilities, probabilities_noise = pkl.load(open("pickles/model_on_data.pkl","rb"))

    bahamas_fid = None

    central_cross_value = log_max_like
    central_cross_error = log_error[:,None]


    for isx, isystematic in enumerate(systematics.keys()):

        fiducial, all_results = pkl.load(open(systematics[isystematic],"rb"))
        ibias_keys = np.sort(list(all_results.keys()))



        colors = colourFromRange([-1,len(ibias_keys)], cmap='Reds')

        colour_norm = mpl.colors.Normalize(vmin=-2, vmax=len(all_results.keys()))
        cmap = plt.cm.Reds


        for ibx, ibias in enumerate(ibias_keys):
            bias_thresholds = []
            fid_thresholds = []
            color = cmap(colour_norm(ibx))
            cross_sections, bias_means, results_error = get_probs_from_results( all_results[ibias] )
            cross_sections, fid_means, results_error = get_probs_from_results( fiducial )

            if bahamas_fid is None:
                bahamas_fid = fid_means.copy()
                bahamas_crs = cross_sections.copy()
            if isystematic == 'projection_depth':

                bias_impact = bias_means / fid_means


                bias_means = bias_impact*bahamas_fid

                fid_means = bahamas_fid.copy()
                cross_sections = bahamas_crs.copy()

            sidm, sidm_error = infer_sidm(
                probabilities['concat'],
                { icross:bias_means[icx] for icx, icross in enumerate(cross_sections)})

            sidm_fid, sidm_error_fid = infer_sidm(
                probabilities['concat'],
                { icross:fid_means[icx] for icx, icross in enumerate(cross_sections)})

            biased_value = sidm-sidm_fid+central_cross_value

            ax.errorbar( biased_value, isx, xerr=central_cross_error, 
                        capsize=capsize, color=colors[ibx], lw=2, capthick=2)

        cax = inset_axes(
            ax,
            width="25%",      # width of colorbar
            height="50%",      # thickness
            loc="center right",
            bbox_to_anchor=(0.29, isx-0.5, 1, 1),
            bbox_transform=ax.get_yaxis_transform(),
            borderpad=0,
        )

        sm = mpl.cm.ScalarMappable(norm=colour_norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1,len(ibias_keys)])
        if isystematic =='shape_measurement_bias':
            ibias_keys = ibias_keys.astype(float)
            ibias_keys[0] = 0.02
            ibias_keys[-1] = 0.00
        if isystematic == 'source_redshift_bias':
            ibias_keys = ibias_keys.astype(float)
            ibias_keys -= 1
        cbar.set_ticklabels([ibias_keys[0], ibias_keys[-1]])
        cbar.ax.tick_params(labelsize=12)


        systematic_names.append(isystematic.replace('_',' ').capitalize())

    for isy, isystematic in enumerate(systematics_thresh.keys()):

        fiducial, results = pkl.load(open(systematics_thresh[isystematic],"rb"))

        colors = colourFromRange([-1,len(results.keys())], cmap='Reds')

        ibias_keys = np.sort(list(results.keys()))
        colors = colourFromRange([-1,len(ibias_keys)], cmap='Reds')


        colour_norm = mpl.colors.Normalize(vmin=-1, vmax=len(ibias_keys))
        cmap = plt.cm.Reds  
        for ibx, ibias in enumerate(ibias_keys):

            if 'thresholds' in list(results[ibias]['bahamas'].keys()):
                means = 1-np.mean(results[ibias]['bahamas']['thresholds'],axis=0)
                fid_means = 1-np.mean(fiducial['bahamas']['thresholds'],axis=0)
                cross_sections =  results[ibias]['bahamas']['cross_section']

            else:
                means = 1-results[ibias]['bahamas']['mean']
                fid_means = 1-fiducial['bahamas']['mean']


            sidm, sidm_error = infer_sidm(
                probabilities['concat'],
                { icross:means[icx] for icx, icross in enumerate(cross_sections)})


            sidm_fid, sidm_error_fid = infer_sidm(
                probabilities['concat'],
                { icross:fid_means[icx] for icx, icross in enumerate(cross_sections)})

            biased_value = sidm-sidm_fid+central_cross_value
            ax.errorbar(biased_value, isy+isx+1, 
                        xerr = central_cross_error, 
                        capsize=capsize, color=colors[ibx], lw=2, capthick=2)

        cax = inset_axes(
            ax,
            width="25%",      # width of colorbar
            height="50%",      # thickness
            loc="center right",
            bbox_to_anchor=(0.29, isy+isx-0.5+1, 1, 1),
            bbox_transform=ax.get_yaxis_transform(),
            borderpad=0,
        ) 
        sm = mpl.cm.ScalarMappable(norm=colour_norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1,len(ibias_keys)])
        if isystematic =='merging_scenario':
            ibias_keys = ibias_keys.astype(float)
            ibias_keys += 1

        if isystematic == 'mass_dependency':
            ibias_keys = ibias_keys.astype(str)
            ibias_keys[0] = r"$-2\sigma$"
            ibias_keys[-1] = r"$+2\sigma$"


        cbar.set_ticklabels([ibias_keys[0], ibias_keys[-1]])
        cbar.ax.tick_params(labelsize=12)


        systematic_names.append(isystematic.replace('_',' ').capitalize())
    ax.set_yticks(np.arange(len(systematic_names)))
    ax.set_yticklabels(systematic_names, fontsize=15)   



    ax.plot( [central_cross_value, central_cross_value],  [-1,len(systematic_names)+1], color='k')
    ax.plot( [central_cross_value-central_cross_error[0], central_cross_value-central_cross_error[0]],  [-1,len(systematic_names)+1], ':', color='k')
    ax.plot( [central_cross_value+central_cross_error[1], central_cross_value+central_cross_error[1]],  [-1,len(systematic_names)+1],  ':',color='k')

    ax.fill_between( 
        [central_cross_value-central_cross_error[0][0], central_cross_value+central_cross_error[1][0]],
        [-1,-1], [len(systematic_names)+1]*2, color='k', alpha=0.1
    )




    ax.set_xlim(-1.5,1)

    ax.set_ylim(-0.5,len(systematic_names)-0.5)
    ax.set_xlabel(
        r'$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$', fontsize=15
    )
    plt.savefig("plots/summarised_systematics.png")

    
# # Figures from the current pipeline (sbi/)
# The functions above read the notebooks/pickles of the original analysis. These make the same
# figures from the outputs of sbi/ (make_summaries.py, sensitivity.py, systematics.py) for one
# ensemble, PAPER_TAG; `python plots_for_paper.py` remakes all of them.


def paper_paths(tag=PAPER_TAG):
    from mass_weighting import FIDUCIAL_STUDY            # the systematics run is per A2744 mass study
    s = f"{SBI_DIR}/summaries_{FIDUCIAL}_{tag}_n5"
    return {"summaries": s, "flamingo": f"{s}_flamingo",
            "sensitivity": f"{SBI_DIR}/results/sensitivity_{FIDUCIAL}_{tag}_w4s1/sensitivity.npz",
            "systematics": f"{SBI_DIR}/results/systematics_{FIDUCIAL}_{tag}_{FIDUCIAL_STUDY}/systematics.json"}


def combined_curve(summaries, mu, sig, min_ncomp=2):
    '''
    Per model, the A2744-mass-weighted mean P(SIDM) against cross-section with BAHAMAS and DARKSKIES
    pooled as in sbi/systematics.py: each simulation's curve, the mean of the two where both have that
    cross-section. Returns (cross_sections, (n_models, n_cross)).
    '''
    curves = {d: summary_curve(summaries, d, mu, sig, min_ncomp) for d in ["bahamas", "darkskies"]}
    sigmas = np.unique(np.concatenate([c[0] for c in curves.values()]))
    out = np.zeros((next(iter(curves.values()))[1].shape[0], len(sigmas)))
    for j, s in enumerate(sigmas):
        out[:, j] = np.mean([c[1][:, np.isclose(c[0], s)][:, 0] for c in curves.values() if np.isclose(c[0], s).any()], 0)
    return sigmas, out


def sigma_posterior(sigmas, curve, p_obs, prior=(-3, 1), cdm=1e-3):
    '''
    Posterior on log10 sigma/m as sbi/systematics.posterior: the paper's GP through the ensemble curve,
    likelihood N(A2744's ensemble P(SIDM); GP mean, sqrt(GP var + error in the ensemble mean^2)), flat
    prior. Returns the fitted GP, the grid, the normalised posterior and (median, 16%, 84%).
    '''
    from sidm_inference_on_data_and_models import gp as gp_template
    from sklearn.base import clone
    x_fit = np.log10(np.where(sigmas == 0, cdm, sigmas))
    gp = clone(gp_template).fit(x_fit[:, None], curve.mean(0))
    x = np.linspace(*prior, 4001)
    mu, sd = gp.predict(x[:, None], return_std=True)
    err = float(return_error_in_mean(np.asarray(p_obs, dtype=float)))
    post = norm.pdf(float(np.mean(p_obs)), mu, np.sqrt(sd ** 2 + err ** 2))
    post /= np.trapz(post, x)
    cdf = cumulative_trapezoid(post, x, initial=0)
    return gp, x, post, tuple(np.interp([0.5, 0.16, 0.84], cdf / cdf[-1], x))


def figure5_summaries(tag=PAPER_TAG, min_ncomp=2, ml_only=False):
    '''
    Figure 5 from the current pipeline: the combined (BAHAMAS + DARKSKIES) response with its GP and
    A2744's ensemble output (left), and the posterior on log10 sigma/m (right; 68/95/99% shaded).
    Returns log10 sigma/m (median) and its 16-84% errors, for figure6.
    '''
    from mass_weighting import a2744_mass_gaussian
    paths = paper_paths(tag)
    mu, sig = a2744_mass_gaussian()
    sigmas, curve = combined_curve(paths["summaries"], mu, sig, min_ncomp)
    p_obs = np.load(f"{paths['summaries']}/obs.npz")["softmax"][:, 0]
    gp, x, post, (med, lo, hi) = sigma_posterior(sigmas, curve, p_obs)
    cdm = 1e-3
    xs = np.log10(np.where(sigmas == 0, cdm, sigmas))

    fig, axarr = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax = axarr[0]
    ax.errorbar(xs, curve.mean(0), return_error_in_mean(curve), fmt="ko", capsize=2, label="Simulations")
    xplot = np.linspace(-3.2, 0.5, 200)
    g, s = gp.predict(xplot[:, None], return_std=True)
    ax.plot(xplot, g, "k--")
    ax.fill_between(xplot, g - s, g + s, alpha=0.15, color="k", label="Gaussian process")
    p_mean, p_err = np.mean(p_obs), return_error_in_mean(p_obs)
    ax.axhspan(p_mean - p_err, p_mean + p_err, color="b", alpha=0.25, lw=0, label="A2744 UNCOVER")
    ax.axhline(p_mean, color="b", lw=1.5)
    ax.set_xticks([-3, -2, -1, 0], ["CDM", "-2", "-1", "0"])
    ax.set_xlabel(r"$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$", fontsize=13)
    ax.set_ylabel(r"Mean $P(\mathrm{SIDM})$", fontsize=13)
    ax.legend(loc=2, fontsize=9)

    ax = axarr[1]
    cdf = cumulative_trapezoid(post, x, initial=0)
    cdf /= cdf[-1]
    for q, a in ((0.005, 0.15), (0.025, 0.25), (0.16, 0.4)):
        a_lo, a_hi = np.interp([q, 1 - q], cdf, x)
        m = (x >= a_lo) & (x <= a_hi)
        ax.fill_between(x[m], post[m], color="b", alpha=a, lw=0)
    best = 10 ** med
    e_lo, e_hi = best - 10 ** lo, 10 ** hi - best
    ml = x[np.argmax(post)]
    if ml_only:      # the peak, with the same 16-84% interval
        b = 10 ** ml
        ax.plot(x, post, "k-", label=rf"$\sigma_{{\rm DM}}/m={b:.2f}^{{+{10 ** hi - b:.2f}}}_{{-{b - 10 ** lo:.2f}}}$ cm$^2$g$^{{-1}}$")
        ax.axvline(ml, color="k", lw=1)
    else:
        ax.plot(x, post, "k-", label=rf"$\sigma_{{\rm DM}}/m={best:.2f}^{{+{e_hi:.2f}}}_{{-{e_lo:.2f}}}$ cm$^2$g$^{{-1}}$")
        ax.axvline(med, color="k", lw=1)
        ax.axvline(ml, color="k", lw=1.2, ls="--", label=rf"maximum likelihood $\sigma_{{\rm DM}}/m={10 ** ml:.2f}$ cm$^2$g$^{{-1}}$")
    ax.set_xlabel(r"$\log_{10}\!\left[(\sigma_{\rm DM}/m)/(\mathrm{cm}^2\,\mathrm{g}^{-1})\right]$", fontsize=13)
    ax.set_ylabel("Posterior", fontsize=13)
    ax.set_xlim(-3, 1)
    ax.set_ylim(0, 1.25 * post.max())
    ax.legend(loc=2, fontsize=9)
    filename = f"plots/output_to_model_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    import json
    json.dump({"ml": round(float(ml), 3), "median": round(float(med), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)},
              open(f"plots/gp_posterior_ba_{tag}.json", "w"), indent=1)
    print(f"Saved {filename}: ML log10 sigma/m = {ml:.3f} ({10 ** ml:.2f}); median log10 sigma/m = {med:.2f} [{lo:.2f}, {hi:.2f}]; sigma/m = "
          f"${best:.2f}^{{+{e_hi:.2f}}}_{{-{e_lo:.2f}}}$")
    return med, np.array([med - lo, hi - med])


def sensitivity_panel(ax, delta, centres, kappa, kappa_extent, title, vmin=1, vmax=3):
    '''One occlusion-sensitivity map (Delta / its rms over windows with galaxies; > 0 pushes to SIDM)
    with convergence contours, in stamp pixels about the centre.'''
    d = np.where(delta != 0, delta, np.nan)
    d = d / np.nanstd(d)
    half = (centres[1] - centres[0]) / 2
    extent = [centres[0] - half, centres[-1] + half] * 2
    im = ax.imshow(d, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.contour(gaussian_filter(kappa, 1.5), origin="lower", extent=kappa_extent, cmap="YlOrRd",
               levels=np.linspace(0.05, 0.6, 12), linewidths=1)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor("#20203a")
    ax.errorbar(-35, -45, xerr=10, capsize=2, color="white", lw=2)
    ax.text(-35, -45, "400 kpc", ha="center", va="bottom", color="white", weight="bold")
    ax.text(-45, 40, title, color="white", weight="bold", fontsize=16)
    return im


def figure3_sensitivity(tag=PAPER_TAG, sigma_sidm=1.0):
    '''
    Figure 3 from sbi/sensitivity.py (4x4-pixel windows, stride 1): the occlusion sensitivity of the
    BAHAMAS CDM and sigma/m = sigma_sidm clusters most like A2744 (largest A2744 mass weight) and of
    A2744, each in units of its rms, positive (pushes to SIDM) only, with the convergence contours.
    '''
    z = np.load(paper_paths(tag)["sensitivity"])
    c = z["centres"]
    n = len(c)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, s, label in zip(axarr[:2], (0, sigma_sidm), ("Collisionless", f"SIDM {sigma_sidm:g}")):
        rows = np.where(np.isclose(z["bahamas_sigma"], s))[0]
        i = rows[np.argmax(z["bahamas_weight"][rows])]
        sensitivity_panel(ax, z["bahamas_delta"][i], c, z["bahamas_kappa_windows"][i].reshape(n, n),
                          [c[0], c[-1]] * 2, label)
        print(f"{label}: cluster {i}, log10 M750 {z['bahamas_log10_mass_750'][i]:.2f}, P(SIDM) {z['bahamas_p_full'][i]:.2f}")
    im = sensitivity_panel(axarr[2], z["a2744_delta"].mean(0), c, z["a2744_kappa"], [-50, 50, -50, 50], "Observation")
    fig.colorbar(im, ax=axarr, fraction=0.015, pad=0.01).ax.set_title("$S$")
    filename = f"plots/sensitivity_map_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")


def figure4_sensitivity(tag=PAPER_TAG, bins=np.linspace(0, 0.5, 11), nshuffle=1000, seed=0):
    '''
    Figure 4 from sbi/sensitivity.py: the fraction of windows (with galaxies) whose removal lowers
    P(SIDM), i.e. that push towards SIDM, against the convergence in the window, for the BAHAMAS
    clusters at each cross-section (pooled, binomial errors) and A2744 (errors from shuffling kappa).
    '''
    z = np.load(paper_paths(tag)["sensitivity"])
    c = z["centres"]
    xc = 0.5 * (bins[1:] + bins[:-1])

    def fraction(kappa, delta):
        keep = (delta != 0) & np.isfinite(kappa)
        f, _, _ = binned_statistic(kappa[keep], (delta[keep] > 0).astype(float), "mean", bins)
        n, _, _ = binned_statistic(kappa[keep], delta[keep], "count", bins)
        return f, n

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    for s in np.unique(z["bahamas_sigma"]):
        rows = np.isclose(z["bahamas_sigma"], s)
        f, n = fraction(z["bahamas_kappa_windows"][rows].ravel(), z["bahamas_delta"][rows].reshape(rows.sum(), -1).ravel())
        ax.errorbar(xc, f, np.sqrt(f * (1 - f) / np.maximum(n, 1)), fmt="o-", capsize=2, lw=2,
                    label=f"BAHAMAS {'CDM' if s == 0 else f'{s:g}'}")
    ij = np.clip(np.round(c + 49.5).astype(int), 0, 99)
    k_obs = z["a2744_kappa"][np.ix_(ij, ij)].ravel()
    d_obs = z["a2744_delta"].mean(0).ravel()
    f, _ = fraction(k_obs, d_obs)
    rng = np.random.default_rng(seed)
    null = np.array([fraction(rng.permutation(k_obs), d_obs)[0] for _ in range(nshuffle)])
    ax.errorbar(xc, f, np.nanstd(null, 0), fmt="o", color="k", capsize=2, label="Observation")
    ax.fill_between([0, 1], [0.5] * 2, [1] * 2, color="c", alpha=0.2)
    ax.fill_between([0, 1], [0] * 2, [0.5] * 2, color="k", alpha=0.2)
    ax.text(0.05, 0.9, "Regions sensitive to SIDM", transform=ax.transAxes, fontsize=13)
    ax.text(0.05, 0.05, "Regions sensitive to CDM", transform=ax.transAxes, fontsize=13)
    ax.set_xlabel(r"Weak lensing convergence, $\kappa$")
    ax.set_ylabel("Fraction of windows sensitive to SIDM")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1)
    leg = ax.legend(fontsize=8, loc=1)
    leg.get_frame().set_edgecolor("black")
    filename = f"plots/Sensitivty_relation_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")


def figureA2_summaries(tag=PAPER_TAG, nrandom=1000, seed=0):
    '''
    Figure A2 from the current summaries: how correlated the ensemble members are. The errors are
    each model's output on every (BAHAMAS CDM cluster, shape-noise draw) about that cluster's mean
    over all models and draws, in units of their scatter over models; the RMSE of the mean of N
    random models against N, fitted with sigma sqrt((1 - rho)/N + rho) and a power law.
    '''
    paths = paper_paths(tag)
    P = np.load(f"{paths['summaries']}/params_bahamas.npz")
    soft = np.load(f"{paths['summaries']}/softmax_bahamas.npy")[:, :, P["cross_section"] == 0]   # (n_models, n_noise, N)
    y_true = np.broadcast_to(soft.mean((0, 1)), soft.shape[1:]).ravel()
    pred = soft.reshape(soft.shape[0], -1)
    scale = np.std(pred - y_true, axis=0)
    n_models = pred.shape[0]
    Ns = np.arange(1, n_models + 1)
    rng = np.random.default_rng(seed)
    rmse = np.array([np.mean([np.sqrt(np.mean(((pred[rng.choice(n_models, N, replace=False)].mean(0) - y_true) / scale) ** 2))
                              for _ in range(nrandom)]) for N in Ns])

    def correlated_rmse(N, sigma, rho):
        return sigma * np.sqrt((1 - rho) / N + rho)

    def power(N, b):
        return rmse[0] * N ** b

    (alpha,), _ = curve_fit(power, Ns, rmse)
    (sigma_fit, rho_fit), pcov = curve_fit(correlated_rmse, Ns, rmse, p0=[1, 0.1])
    print(f"ensemble RMSE ~ N^{alpha:.3f}; rho = {rho_fit:.3f} +/- {np.sqrt(pcov[1, 1]):.3f}")
    plt.figure(figsize=(5, 4))
    plt.plot(Ns, rmse, "o", label="Measured")
    plt.plot(Ns, correlated_rmse(Ns, sigma_fit, rho_fit), "-", lw=2, label=f"Fitted correlation law $\\rho={rho_fit:0.2f}$")
    plt.plot(Ns, sigma_fit / np.sqrt(Ns), "--", label=r"Independent models ($N^{-0.5}$)")
    plt.xlabel("Number of models N", fontsize=15)
    plt.ylabel("Ensemble RMSE / standard deviation", fontsize=13)
    plt.legend()
    plt.tight_layout()
    filename = f"plots/correlated_models_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")


def figureA3_summaries(tag=PAPER_TAG, nshow=4, min_ncomp=2, seed=42):
    '''
    Figure A3 from summaries with latents (sbi/make_summaries.py without --no-latents, one noise draw,
    in <summaries>_latents): for the first nshow models, the UMAP of the DARKSKIES latents with the
    BAHAMAS ones projected on it (density contours) and A2744 (star); then, per cross-section, the
    distance of A2744's latent from the mean latent of the simulated clusters, in units of the pooled
    BAHAMAS + DARKSKIES CDM clusters' mean distance from their mean (1 = a typical CDM cluster; one scale
    for both domains), median and 16-84% over models / sqrt(n_models).
    Clusters with >= min_ncomp components within systematics.SUBSET_DEX = 0.45 dex of the A2744 mass.
    '''
    from mass_weighting import a2744_mass_gaussian
    d = paper_paths(tag)["summaries"] + "_latents"
    mu, _ = a2744_mass_gaussian()
    obs = np.load(f"{d}/obs.npz")["latents"][:, 0]                          # (n_models, 512)
    lat, par = {}, {}
    for dom in ("bahamas", "darkskies"):
        P = np.load(f"{d}/params_{dom}.npz")
        keep = (P["ncomponents"] >= min_ncomp) & (np.abs(P["log10_mass_750"] - mu) < 0.45)
        lat[dom] = np.load(f"{d}/latents_{dom}.npy", mmap_mode="r")[:, 0][:, keep]   # (n_models, N, 512)
        par[dom] = P["cross_section"][keep]

    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for m in range(nshow):
        ax = plt.subplot(2, 2, m + 1)
        fit = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed).fit(lat["darkskies"][m])
        e = {dom: fit.transform(lat[dom][m]) for dom in lat}
        centre = e["darkskies"].mean(0)
        for dom, cmap in (("darkskies", "Oranges"), ("bahamas", "Blues")):
            ax.contour(*get_density(*(e[dom] - centre).T), cmap=cmap)
        star = fit.transform(obs[m][None]) - centre
        ax.plot(star[:, 0], star[:, 1], "y*", markersize=12, mec="k")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if m % 2 == 0:
            ax.set_ylabel("UMAP-2")
        ax.set_xlabel("UMAP-1")
        ax.text(0.1, 0.9, f"Model {m + 1}", transform=ax.transAxes, fontsize=15)
    filename = f"plots/latent_overlap_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    n_models = obs.shape[0]
    cdm = np.concatenate([lat[dom][:, par[dom] == 0] for dom in lat], axis=1)   # pooled CDM, (n_models, n, 512)
    spread = np.linalg.norm(cdm - cdm.mean(1, keepdims=True), axis=-1).mean(1)  # (n_models,)
    for dom in lat:
        sig = np.unique(par[dom])
        ratio = np.zeros((n_models, len(sig)))
        for j, s in enumerate(sig):
            z = lat[dom][:, par[dom] == s]                                  # (n_models, n, 512)
            ratio[:, j] = np.linalg.norm(obs - z.mean(1), axis=-1) / spread
        med = np.median(ratio, 0)
        q = np.quantile(ratio, [0.16, 0.84], 0)
        x = np.where(sig == 0, 1e-3, sig)
        ax.errorbar(x, med, np.abs(q - med) / np.sqrt(n_models), color=COLOUR_SCHEME[dom], fmt="-o", capsize=2,
                    label=dom.upper())
        print(f"{dom}: " + ", ".join(f"{s:g}: {r:.3f}" for s, r in zip(sig, med)))
    ax.plot([5e-4, 2], [1, 1], "k--")
    ax.text(0.001, 1.0, "Typical CDM cluster", ha="left", va="bottom", fontsize=13)
    ax.set_xlim(5e-4, 2)
    ax.set_xscale("log")
    ax.set_ylabel("Normalised latent space distance", fontsize=15)
    ax.set_xlabel(r"Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    ax.legend()
    filename = f"plots/goodness_of_fit_ba_{tag}.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {filename}")


def figureA4A5_systematics(tag=PAPER_TAG):
    '''Figures A4 (data) and A5 (model systematics) as drawn by sbi/systematics.py, copied to plots/.'''
    import shutil
    d = os.path.dirname(paper_paths(tag)["systematics"])
    for name, out in (("systematics_data", "data_systematics"), ("systematics_model", "summarised_systematics")):
        for ext in ("pdf", "png"):
            shutil.copy(f"{d}/{name}.{ext}", f"plots/{out}_ba_{tag}.{ext}")
    print(f"Copied the systematics figures from {d}")


# ---------------------------------------------------------------------------------------------------------------------
# Figures drawn from results that sbi/ computes. The heavy computation stays in sbi/ (sensitivity maps, ratio scan,
# systematics runs); these functions only draw the paper versions, in the paper style, into plots/.
# ---------------------------------------------------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_sbi(script, *args):
    '''Run sbi/<script> as __main__ with the given arguments (each paper figure runs in its own process, see
    paper_figures.py, so its global state does not leak into other figures).'''
    import runpy
    argv, cwd = sys.argv, os.getcwd()
    sys.argv = [script] + [str(a) for a in args]
    sys.path.insert(0, os.path.join(REPO, "sbi"))
    plt.rcdefaults()                     # the scripts set their own style; not the LaTeX style this module sets
    try:
        runpy.run_path(os.path.join(REPO, "sbi", script), run_name="__main__")
    finally:
        sys.argv = argv
        os.chdir(cwd)


def figure_sensitivity_map(pairs=(7, 10, 12, 15, 16), out="plots/sensitivity_map_pairs"):
    '''Occlusion-sensitivity maps of SIDM / CDM-twin pairs next to A2744 (sbi/pairs_obs_panel.py; needs the
    catalogue for the A2744 KS map, and the pair maps from sbi/pair_sensitivity.py).'''
    _run_sbi("pairs_obs_panel.py", "--pairs", *pairs, "--out", os.path.abspath(out))


def figure_p_vs_mass(tag=PAPER_TAG, out="plots/p_vs_mass"):
    '''Mass-weighted mean P(SIDM) against log M(<750 kpc) for BAHAMAS, DARKSKIES and FLAMINGO CDM with A2744
    (sbi/p_vs_mass.py), in the Figure 2 panel shape.'''
    _run_sbi("p_vs_mass.py", "--tags", tag, "--ncomp", 2, "--no-lenstool-ks", "--harvey", "none", "--xlim", 14, 15.1,
             "--ylim", 0.4, 0.75, "--no-title", "--paper-style", "--figsize", 5.6, 5.0, "--box-aspect", 0.78,
             "--exclude", "tng", "--no-spread", "--out", os.path.abspath(f"{out}_ba_{tag}"))


def figure_shear_ratio(npz=None, out="plots/shear_ratio_scatter_50-100", recompute=False, tag=PAPER_TAG):
    '''P(SIDM) against the shear ratio <chi_t>(50-100 kpc) / <chi_t>(outer annulus) for mass-matched BAHAMAS
    clusters, outer annuli 100-300 ... 700-900 kpc, +-0.25 dex mass window. Redrawn from the saved scan (no
    catalogue); recompute=True reruns sbi/ratio_scan.py first (rebuilds the catalogue).'''
    sys.path.insert(0, os.path.join(REPO, "sbi"))
    scan_dir = os.path.abspath("plots/ratio_scan_50-100")
    npz = npz or os.path.join(scan_dir, "ratio_scan_scatter.npz")
    if recompute:
        _run_sbi("ratio_scan.py", "--summaries", os.path.join(REPO, "sbi", f"summaries_{FIDUCIAL}_{tag}_n5"),
                 "--num-kpc", 50, 100, "--den-start", 100, "--den-width", 200, "--mass-window", 0.25, "--out-dir", scan_dir)
    from ratio_plot import plot_grid
    plt.rcdefaults()                     # ratio_plot sets its own style, on sbi/sensitivity.style() (muted tick labels)
    plt.rcParams.update({"axes.edgecolor": "k", "xtick.color": "#6b6b6b", "ytick.color": "#6b6b6b",
                         "axes.labelcolor": "#222222", "legend.frameon": False, "font.size": 10})
    plot_grid(npz, out)


def figure_systematics(tag=PAPER_TAG, out_dir="plots"):
    '''Figures A4 (data) and A5 (model systematics) from the sbi/systematics.py run (paper_paths()["systematics"]).
    The model plot is centred on the maximum likelihood of the paper GP posterior with its 16-84% range as the
    statistical bar (plots/gp_posterior_ba_<tag>.json, written by figure5_summaries).'''
    import json
    from pathlib import Path
    sys.path.insert(0, os.path.join(REPO, "sbi"))
    import systematics as sy
    out = json.load(open(paper_paths(tag)["systematics"]))
    gp = json.load(open(f"plots/gp_posterior_ba_{tag}.json"))
    model = {**out, "contamination": [r for r in out.get("contamination", []) if abs(r["value"]) <= sy.CONTAMINATION_MAX + 1e-9]}
    sy.plot_ranges(model, Path(out_dir) / f"systematics_model_paper_ba_{tag}", list(sy.MODEL_GROUPS), None,
                   show_total=False, show_shifts=False, stat=(gp["lo"], gp["hi"]), centre=gp["ml"],
                   bottom=("shape_m", "shape_c"))         # the two shear biases together, last
    sy.plot_data_paper(out, Path(out_dir) / f"systematics_data_paper_ba_{tag}", skip=("galaxy_low_mag",), ncol=3)


def figure_pooled_tail(tag=PAPER_TAG, out="plots/cdm_pooled_tail"):
    '''Pooled CDM P(SIDM) distribution against A2744 (sbi/cdm_pooled_tail.py).'''
    from mass_weighting import FIDUCIAL_STUDY
    _run_sbi("cdm_pooled_tail.py", "--tag", tag, "--studies", FIDUCIAL_STUDY, "--out", os.path.abspath(f"{out}_{tag}_nc2"))


def figure_empirical_posterior(tag=PAPER_TAG, out="plots/empirical_posterior"):
    '''Empirical posterior on sigma/m from the mass-weighted simulations (sbi/empirical_posterior.py).'''
    from mass_weighting import FIDUCIAL_STUDY
    _run_sbi("empirical_posterior.py", "--summaries", os.path.join(REPO, "sbi", f"summaries_{FIDUCIAL}_{tag}_n5"),
             "--mass-study", FIDUCIAL_STUDY, "--out", os.path.abspath(f"{out}_ba_{tag}"))


def all_figures(tag=PAPER_TAG):
    '''All paper figures in one process. paper_figures.py does the same with each figure in its own process and
    collects them under the paper's file names; prefer it.'''
    os.system(f"{sys.executable} paper_figures.py --tag {tag}")


if __name__ == "__main__":
    # python plots_for_paper.py <function> : one figure in this process (e.g. figure2_summaries, figureA1)
    if sys.argv[1:2] and sys.argv[1] != "all":
        globals()[sys.argv[1]]()
        sys.exit()
    all_figures()
