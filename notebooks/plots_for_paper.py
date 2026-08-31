#!/usr/bin/env python

from get_model_probabilities import *
from add_shear_to_data import combine_catalogues, get_source_redshift, return_error_in_mean, get_model_names
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
    

def figure1():
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


    obs_cat = get_obs_data( 'concat', data_dir='../data/100/a2744/')
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

    im = ax[1,2].imshow(ideal_tgt[0][1]-ideal_tgt[0][0].max()/2., cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
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
        iax.errorbar( 15, 5,xerr=10, capsize=2, color='white')
        iax.text( 15, 5, '200kpc', ha='center', va='bottom', color='white')
        iax.plot([ 80, 90], [5,5], '-', color='white')
        iax.plot([90, 90],[5, 15], '-', color='white')
        #iax.text( 90, 5, 'E', ha='left', va='center', color='white')
        #iax.text( 72, 10, 'N', ha='left', va='center', color='white')
        iax.set_xticks([])
        iax.set_yticks([])

    filename = "plots/data_examples.png"
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


            axarr[1].set_xlabel("Model Output", fontsize=15)
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
    axarr[0].set_ylabel("Model Output", fontsize=15)

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

def figure3and4():
    # # Get sensitivity plots

    obs_cat = get_obs_data( 'concat', data_dir="../data/100/a2744/")
    binned_data = bin_obs_data(obs_cat)

    obs_cat['x'] = binned_data['delta_ra']
    obs_cat['y'] = binned_data['delta_dec']
    image_size  = 100
    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_cat, smooth=1.5, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )

    segment_size = 4
    sim_fiducial, sim_probabilities = pkl.load(open(f"pickles/sim_senstivity_{segment_size}_select_moving_av.pkl","rb"))      
    cmap_contour = 'YlOrRd'
    cmap_image = 'viridis'
    kappa_bins = np.linspace(0.05,0.65,8)

    fig, axarr=plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.1)

    kappas = []
    contrast=5
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
    ax.set_ylabel("Mean Model Output", fontsize=15)

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


    kde = gaussian_kde( est['estimates'])
    xpdf = np.linspace(-10,10,10000)
    ypdf = kde.pdf(xpdf)

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
    ax.set_ylabel("Posterior Likelihood", fontsize=15)
    ax.legend(loc=1)
    ax.set_xlim(-2.5, 1.5)
    ax.set_xticks(np.linspace(-2.5,1,6))
    ax.set_ylim(0, 0.9)

    fig.align_ylabels()
    filename = "plots/output_to_model.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))  
    print(f"${max_like:.2f}^{{+{error[1]:.2f}}}_{{-{error[0]:.2f}}}")

    
    return log_max_like, log_like_one_sigma

def figure6( log_max_like, log_error):
    
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

    ax.plot( np.mean(velocity_merg), max_like, '*',
                markersize=20, color='orange', lw=2, label='This Work (Merger Velocity)')

    ax.errorbar( np.mean(velocity_disp), max_like, yerr=error[:,None], 
                markersize=20, color='red', fmt='*', capsize=4, lw=2)

    ax.errorbar( np.mean(velocity_merg), max_like, yerr=error[:,None], 
                markersize=20, color='orange', fmt='*', capsize=4, lw=2)
    print(f"$\sigma_{{\\rm DM}}/m={max_like.real:0.2f}_{{-{error[0].real:0.2f}}}^{{+{error[1].real:0.2f}}}$cm$^2$g^{{-1}}")
    # get handles
    ax.legend(loc=1)
    fname = "plots/particle_physics.pdf"
    plt.savefig(fname)

# # Appendix plots

def figureA1():
    
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
    ##### PLOT 2 Ngalaxies per bin #####
    ####################################

    ax = plt.subplot(nrow,ncol,2)

    nmonte = 10000
    intrinsic_ell = []
    ngal_range = np.arange(20)+1
    for ngal in ngal_range:
        rdn_ells = []
        for imonte in range(nmonte):
            rdn_idx = np.random.choice(
                np.arange(obs_data['e1'].shape[0]),
                size=ngal
            )

            e1rdn = np.mean(obs_data['e1'][rdn_idx])
            e2rdn = np.mean(obs_data['e2'][rdn_idx])



            rdn_ells.append(np.sqrt(e1rdn**2+e2rdn**2))

        fit = chi.fit(rdn_ells)[2]

        intrinsic_ell.append(fit)
    intrinsic_ell = np.array(intrinsic_ell)
    ax.plot(
        ngal_range, intrinsic_ell, '-o', label='Measured'
    )
    ax.plot(
        ngal_range, intrinsic_ell[0]/np.sqrt(ngal_range), '-',label='1/sqrt(N)', lw=2
    )


    ax.set_xlabel("Number of galaxies in a bin",fontsize=15)
    ax.set_ylabel("Fitted intrinsic ell of chi fit",fontsize=15)
    ratio =  (intrinsic_ell[0]/np.sqrt(ngal_range)) / intrinsic_ell



    def g1_func( snr, a, b, c, d):

        return a + b*np.arctan( ( snr - c)/d)



    popt, pcov = curve_fit(
                    g1_func,
                    ngal_range,
                    ratio)


    ax.plot(
        ngal_range, intrinsic_ell[0]/(g1_func( ngal_range, *popt)*np.sqrt(ngal_range)), label='Fit', lw=2
    )
    ax.legend(fontsize=15)
    ax.set_ylabel("Probability distribution",fontsize=15)

    ####################################
    ##### PLOT 3 Source redshift #####
    ####################################


    ax = plt.subplot(nrow,ncol,3)
    ax.set_ylabel("Probability distribution",fontsize=15)

    obs_data = get_obs_data(
                ifilter, data_dir="../data/100/a2744/", photoz=True, remove_members=True)

    UNCOVER_ERROR=0.06
    kde = gaussian_kde(
        obs_data['redshift'][ np.abs(obs_data['redshift']-args.default_zs) > 1e-2],
        UNCOVER_ERROR
    )
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
        [0,1.0],
        'k--', label=f"$\langle z\\rangle={args.default_zs:0.2f}$")
    ax.legend(fontsize=15)
    ax.set_xlabel("Galaxy source redshift", fontsize=15)
    ax.set_ylim(0,0.7)
    ax.set_xlim(0,10)

    ####################################
    ##### PLOT 4 Susceptibility tensor #####
    ####################################

    ax = plt.subplot(nrow,ncol,4)


    def std_err( x ):
        return np.std(x)/np.sqrt(x.shape[0])

    cuts= {'f115w':{'signal_noise_cut':4, 'stat_type':'snr'}, 
           'f150w':{'signal_noise_cut':5, 'stat_type':'snr'}}
    fid_cut = {
                'size_cut':[2,200],
                'signal_noise_cut':5,
                'stat_type':'snr',
                'mag_cut':[0,30],
                'verbose':False
            }

    xbins = np.linspace(0,20,30)



    color={'f115w':'b','f150w':'r'}
    for ifilter in ['f115w','f150w']:
        obs_data = fits.open(f"../data/100/a2744/abell2744clu-grizli-v5.4-{ifilter}-clear_drc_sci_clean.shears")[1].data
        this_cut = fid_cut.copy()

        for icut in cuts[ifilter].keys():
            this_cut[icut] = cuts[ifilter][icut]

        calc_shear(
                    obs_data, 
                    "test.fits", 
                    **this_cut
        )
        data = fits.open("test.fits")[1].data

        y, x, n = binned_statistic(
            data['snr'],data['g1_gal'], 
            bins=xbins
        )
        stdy, x, n = binned_statistic(
            data['snr'],data['g1_gal'], 
            bins=xbins,
            statistic=std_err
        )
        xc = (x[1:]+x[:-1])/2.

        ax.errorbar(xc, y, stdy, capsize=2,fmt='o',label=ifilter, color=color[ifilter] )
        ax.plot(np.sort(data['snr']), data['g1_model'][np.argsort(data['snr'])],'--', color=color[ifilter])
        ax.set_xlim(3,20)
    ax.set_xlabel('Detection signal-to-noise',fontsize=15)
    ax.set_ylabel('Lensing susceptibility tensor',fontsize=15)
    ax.legend(fontsize=15)


    ####################################
    ##### PLOT 5 B MODES #####
    ####################################

    ax = plt.subplot(nrow,ncol,5)


    stacked_rotated_modes,obs_kappa_b = pkl.load(open("pickles/bmode_check.pkl","rb"))
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
    ##### PLOT 6 GALAXY DENSITY #####
    ####################################

    ax = plt.subplot(nrow,ncol,6)


    obs_cat = get_obs_data( 'concat', data_dir="../data/100/a2744/")
    binned_data = bin_obs_data(obs_cat)
    ngal = binned_data['ngal']
    if np.max(obs_cat['x']) > 1000:
        ra,dec = ra_dec_to_simulation_image_pos( obs_cat)
        obs_cat['x'] = ra
        obs_cat['y'] = dec
    image_size  = 100
    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_cat, smooth=1.5, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )

    kappa_filtered = obs_kappa_e[ ngal > 0 ].flatten()
    ngal_filtered = ngal[ ngal > 0 ].flatten()


    bins = 20
    y,x,n = binned_statistic( kappa_filtered, ngal_filtered, bins=bins, statistic=np.mean )
    stdy,x,n = binned_statistic( kappa_filtered, ngal_filtered, bins=bins, statistic=np.std )
    cy,x,n = binned_statistic( kappa_filtered, ngal_filtered, bins=bins, statistic='count')

    xc = (x[1:]+x[:-1])/2.
    fractional_change = y/np.mean(ngal_filtered)
    fraction_err = stdy/np.sqrt(cy)/np.mean(ngal_filtered)

    ax.errorbar(xc,fractional_change,fraction_err, capsize=2, fmt='o-',color='k')


    def lin_func( x, a, b):
        return x*0+a 

    popt, pope = curve_fit(
        lin_func, xc[ xc>0], fractional_change[xc>0],sigma=fraction_err[xc>0]
    )

    ax.plot( np.array([-1,1]), lin_func(np.array([-1,1]), *popt),'r--',label=f'N$_{{cont}}={popt[0]:0.2f}$')
    ax.set_xlim(-0.4,0.8)
    ax.set_xlabel("Weak lensing convergence, $\kappa$", fontsize=15)
    ax.set_ylabel("$N_{\\rm gal}$ / $\langle N_{\\rm gal}\\rangle$", fontsize=15)
    ax.legend()



    filename = "plots/intrinsic_ell.pdf"
    plt.savefig(filename)

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

    
if __name__ == "__main__":
    #figure1()
    #figure2()
    figure3and4()
    #log_max_like, log_error = figure5()
    #figure6(log_max_like, log_error)
    #figureA1()
    #figureA2()
    #figureA3()
    #figureA4(log_max_like, log_error)
    #figureA5(log_max_like, log_error)



