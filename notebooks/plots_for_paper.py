#!/usr/bin/env python
# coding: utf-8


from get_model_probabilities import *
filter_color = {'f115w':'c', 'f150w':'pink','concat':'k'}
from add_shear_to_data import combine_catalogues

import scienceplots
plt.style.use(["science","grid"])


# Paper Plots
# -----------
# In various notebooks i have carried out some tests, etc, in this notebook, i 
# bring together all the plots that i have done and put the code here for 
# the paper. These include

# # Data plots

# ## Redshift distribution and masses

# ## Intrinsic ellipticity distribution 
# ----
# This requires a2744_f115w_filtered.fits and a2744_f150w_filtered.fits

# In[286]:




# ## Postage stamp examples

# ### Get the ideal data

def figure1():
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
        obs_sampled_data[i][0].dataset.dataset.transform.transforms[2].apply = 0
        if 'source' in i:
            obs_sampled_src = obs_sampled_data[i][0].dataset.dataset[idx]
        if 'target' in i:
            obs_sampled_tgt = obs_sampled_data[i][0].dataset.dataset[idx]


    for i in obs_sampled_data.keys():
        obs_sampled_data[i][0].dataset.dataset.transform.transforms[2].apply = 1.
        if 'source' in i:
            obs_noisy_src = obs_sampled_data[i][0].dataset.dataset[idx]
        if 'target' in i:
            obs_noisy_tgt = obs_sampled_data[i][0].dataset.dataset[idx]




    obs_meta, obs_data = pkl.load(open(f"../data/100/a2744/obs_data_concat.pkl","rb"))
    obs_cat = combine_catalogues(
        "../data/100/a2744/a2744_f115w_filtered.fits",
        "../data/100/a2744/a2744_f150w_filtered.fits"
    )


    # ### Now plot




    fig, ax = plt.subplots(2, 5, figsize=(16,6))
    fig.subplots_adjust(hspace=0.1,wspace=0.)
    cmap='inferno'
    ax[0,2].imshow(ideal_src[0][0], cmap=cmap, origin='lower')
    ax[0,2].text( 5, 90, 'Ideal source domain ($\gamma_1$)', color='white', fontweight='bold')

    ax[1,2].imshow(ideal_tgt[0][1], cmap=cmap, origin='lower')
    ax[1,2].text( 5, 90, 'Ideal target domain ($\gamma_2$)', color='white', fontweight='bold')

    ax[0,3].imshow(obs_sampled_src[0][0], cmap=cmap, origin='lower')
    ax[0,3].text( 5, 90, 'Masked source domain ($\gamma_1$)', color='white', fontweight='bold')

    ax[1,3].imshow(obs_sampled_tgt[0][1], cmap=cmap, origin='lower')
    ax[1,3].text( 5, 90, 'Masked target domain ($\gamma_2$)', color='white', fontweight='bold')

    ax[0,4].imshow(obs_noisy_src[0][0], cmap=cmap, origin='lower')
    ax[0,4].text( 5, 90, 'Noisy source domain ($\gamma_1$)', color='white', fontweight='bold')

    ax[1,4].imshow(obs_noisy_tgt[0][1], cmap=cmap, origin='lower')
    ax[1,4].text( 5, 90, 'Noisy target domain ($\gamma_2$)', color='white', fontweight='bold')


    ax[0,0].imshow(obs_data[0][0], cmap=cmap, origin='lower')
    ax[0,0].text( 5, 90, 'A2744 UNCOVER ($\gamma_1$)', color='white', fontweight='bold')

    ax[1,0].imshow(obs_data[0][1], cmap=cmap, origin='lower')
    ax[1,0].text( 5, 90, 'A2744 UNCOVER ($\gamma_2$)', color='white', fontweight='bold')

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

    im2 = ax[0,1].imshow(kappa_e, cmap=cmap, origin='lower')
    ax[0,1].text( 5, 90, 'A2744 $\\kappa_E$', color='white', fontweight='bold')
    divider = make_axes_locatable(ax[0,4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im2 =ax[1,1].imshow(kappa_b, cmap=cmap, origin='lower')
    ax[1,1].text( 5, 90, 'A2744 $\kappa_B$', color='white', fontweight='bold')

    divider = make_axes_locatable(ax[1,4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')


    for iax in ax.flatten():
        iax.set_xticklabels([])
        iax.set_yticklabels([])
        iax.errorbar( 15, 5,xerr=10, capsize=2, color='white')
        iax.text( 15, 5, '200kpc', ha='center', va='bottom', color='white')
        iax.plot([ 80, 90], [85,85], '-', color='white')
        iax.plot([90, 90],[95, 85], '-', color='white')
        iax.text( 85, 84, 'E', ha='center', va='top', color='white')
        iax.text( 91, 90, 'N', ha='left', va='center', color='white')
        iax.set_xticks([])
        iax.set_yticks([])

    filename = "plots/data_examples.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))


# # Domain Adaptation

def figure2():

    config='baha2dark'
    filter_list  = ['concat']
    cdm = 3e-3

    color = {
            'src':'r',
            'tgt':'b'
        }

    domain = {
        'tgt':'darkskies_obs',
        'src':'bahamas_obs'
    }


    correction = 2.1
    ### MASS WEIGHTS FOR CALCULATING THE MEAN #####
    zs = 1.65
    zl = 0.305
    critical_density = sigma_critical(zl, zs, Planck18).to(units.Msun/units.kpc/units.kpc)

    bah_meta, bah_data = pkl.load(open("../data/100/shear/bahamas_cdm.pkl","rb"))
    dar_meta, dar_data = pkl.load(open("../data/100/shear/darkskies_cdm.pkl","rb"))
    mass_bah = np.log10(np.sum(np.sum(bah_data[:,2],axis=-1)*critical_density* (20*units.kpc)**2, axis=-1).value)
    mass_dark = np.log10(np.sum(np.sum(dar_data[:,2],axis=-1)*critical_density* (20*units.kpc)**2, axis=-1).value)

    yb, x = np.histogram(mass_bah, bins=np.linspace(14,15.5,30))
    yb[ yb == 0] = 1
    yd, x = np.histogram(mass_dark, bins=np.linspace(14,15.5,30))
    yd[yd==0] = 1
    mass_weights = {'src':{'x':x, 'y':1./yb},'tgt':{'x':x, 'y':1./yd}} 
    #######




    fig, axarr = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)

    ifilter = 'concat'


    postive_mass, mass_cut = get_mass_cut( ifilter, nsigma=1., study='harvey' )


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
    fiducial = f"pickles/all_models_{ifilter}_nz_results.pkl"
    tng_flamingo = f"pickles/flamingo_tng_results.pkl"

    colors = {
        'flamingo':'cyan', 'bahamas':'r', 'darkskies':'b','tng':'green'}

    cdm_vals = []
    cdm_val_err = []
    npts = 1000
    xpdf = np.linspace(0.45,0.7, npts)
    marginalised_pdf = np.zeros(npts)

    for ifx, results_file in enumerate( [ fiducial, tng_flamingo ]):
        domain = this_domain[ifx]

        all_results= pkl.load(open(results_file,'rb'))

        for itgt, target in enumerate(['src','tgt']):
            all_thresholds = []

            domain_name = domain[target].split('_')[0]
            if domain_name == 'tng':
                domain_name = 'tng-cluster'
            for imodel in all_results.keys():

                seed = float(imodel.split('_')[1])                


                tgt = get_threshold_for_cross( 
                    all_results[imodel][target], 
                    mass_cut=mass_cut, 
                    integrated_mass=True,
                    function=np.mean,
                    mass_weights=mass_weights[target],
                    dataset=domain[target].split('_')[0],
                    ncomponents=1,
                    quiet=False)


                all_thresholds.append(tgt['thresholds'])

            all_thresholds = np.array(all_thresholds)
            means = np.nanmean(all_thresholds,axis=0)
            errors = np.std(all_thresholds,axis=0) / all_thresholds.shape[0]**0.38*correction

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


                axarr[0].set_xlabel("Self-interaction cross-section [cm$^2$/g]")




            ypdf = norm.pdf( xpdf, *(1-means[0], errors[0]))
            marginalised_pdf += ypdf
            axarr[1].plot(  xpdf, ypdf,
                             color=colour_scheme[f"{domain[target].split('_')[0].lower()}"])

            axarr[1].fill_between( xpdf,
                            ypdf,
                            np.zeros(npts),
                             color=colour_scheme[f"{domain[target].split('_')[0].lower()}"],
                           alpha=0.2, label=domain_name.upper())


            axarr[1].set_xlabel("Model Output", fontsize=12)
            axarr[1].set_ylabel("Probablity Distribution", fontsize=12)

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

    axarr[0].set_xlabel("Self-interaction cross-section [cm$^2$/g]", fontsize=12)
    axarr[0].set_ylabel("Model Output", fontsize=12)

    #
    axarr[0].set_xlim(1e-3,1.5)
    #axarr[0].text(0.7,0.05,f"Noise Floor", transform=axarr[0].transAxes,fontsize=12, ha='left')
    axarr[0].set_xscale('log')


    plot_observations( "pickles/model_on_data.pkl", ifilter, ax=axarr[0], correction=correction, 
                      plot_args={'lw':2, 'color':'k'}, fill_args={'alpha':0.2, 'color':'k'},
                      uncertainty=[68,95])
    #plot_observations( "pickles/model_on_data.pkl", ifilter, ax=axarr[0], correction=correction, 
    #                  plot_args={'lw':2, 'color':'k','ls':'--'}, fill_args={'alpha':0.1, 'color':'b'},
    #                   noise=True )
    #plot_observations( "pickles/model_on_data.pkl", ifilter, ax=ax, noise=True, legend=False, **{'label':'Noise','color':'k'} )
    axarr[0].legend(loc=2,ncols=2)

    plot_observations( "pickles/model_on_data.pkl", ifilter, ax=axarr[1], correction=correction, 
                      plot_args={'lw':2, 'color':'k'}, fill_args={'alpha':0.1, 'color':'k'}, plotpdf=True)

    edges = [np.min(cdm_vals)-0.02, np.max(cdm_vals)+0.02]
    curly_brace(axarr[1], edges[0], edges[1], 80, 10, upward=True)
    axarr[1].text(np.mean(edges), 90, "Collisionless \n Dark Matter", ha='center', fontsize=12)       

    axarr[1].text(0.58, 25, "Observation \n (A2744 UNCOVER)", ha='center', fontsize=12)      

    axarr[1].grid(False)
    axarr[1].set_ylim(0, 100)

    axarr[1].legend()

    fig.align_xlabels()

    filename = "plots/final_model_weighting_with_data.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))
 
    models, prob, noise = pkl.load(open("pickles/model_on_data.pkl","rb"))
    obsprob = 1-np.mean(prob['concat'])
    corr = 30**0.38/2.1
    obsstd = np.std(prob['concat'])/corr
    margingliased_sigma = np.sqrt(np.sum( (xpdf - xpdf[ np.argmax(marginalised_pdf) ])**2*marginalised_pdf)/np.sum(marginalised_pdf))
    total_sigma = np.sqrt( margingliased_sigma**2 + obsstd**2)
    significance = (obsprob - xpdf[ np.argmax(marginalised_pdf) ])/total_sigma
    print(f"Signficance is {significance}")
    cumsum = 1-norm.cdf( xpdf, xpdf[ np.argmax(marginalised_pdf) ], total_sigma)
    ot_prob = cumsum[np.argmin(np.abs(xpdf-0.575))]
    print(f"One tailed prob is {100-100*ot_prob}" )
    sigmas = np.linspace(1,5,1000)
    significance = sigmas[np.argmin(np.abs(norm.sf(sigmas) - ot_prob))]
    print(f"Signficance is {significance}")

# ## A quick estimate of the significance
# 

def figure3():

    # # Get sensitivity plots


    obs_meta, obs_data = pkl.load(open(f"../data/100/a2744/obs_data_concat.pkl","rb"))
    obs_cat = combine_catalogues(
        "../data/100/a2744/a2744_f115w_filtered.fits",
        "../data/100/a2744/a2744_f150w_filtered.fits"
    )


    if np.max(obs_cat['x']) > 1000:
        ra,dec = ra_dec_to_simulation_image_pos( obs_cat)
        obs_cat['x'] = ra
        obs_cat['y'] = dec
    image_size  = 100
    obs_kappa_e, obs_kappa_b = get_kappa(
        obs_cat, smooth=1.5, correct_for_ngal=True, extent=[
                        -image_size//2,image_size//2,-image_size//2,image_size//2
                    ]
    )


    # In[24]:


    segment_size = 3
    sim_fiducial, sim_probabilities = pkl.load(open(f"pickles/sim_senstivity_{segment_size}_select_moving_av.pkl","rb"))      
    cmap_contour = 'YlOrRd'
    cmap_image = 'viridis'
    kappa_bins = np.linspace(0.1,0.8,8)

    fig, axarr=plt.subplots(1, 3,figsize=(15, 5))
    fig.subplots_adjust(wspace=0.1)

    kappas = []
    contrast=5
    vmin = 1

    vmax =  contrast
    ref = None
    dmlabel = ['Collisionless','SIDM0.2','Observation']

    h=0.7
    kappa_idx = 108
    for idx in range(sim_probabilities.shape[0]):
        if idx ==0:
            meta, kappa_data = pkl.load(open("../data/100/shear/darkskies_cdm_3.pkl","rb"))
            meta, data = pkl.load(open("../data/100/obs/concat/darkskies_cdm_3.pkl","rb"))
        else:
            meta, kappa_data = pkl.load(open("../data/100/shear/darkskies_0.2_3.pkl","rb"))
            meta, data = pkl.load(open("../data/100/obs/concat/darkskies_0.2_3.pkl","rb"))

        kappa_e = kappa_data[kappa_idx,2,:,:]/h



        relative_prob = np.mean(sim_fiducial[:,idx,0] - sim_probabilities[idx], axis=-1)

        relative_prob /= relative_prob.std()/2.

        kappas.append(kappa_e)
        ax = axarr[idx]



        snr = ax.imshow(relative_prob, 
                         origin='lower', extent=[-50,50,-50,50],
                         vmin=vmin, vmax=vmax,
                         cmap=cmap_image)
        #

        ax.contour(gaussian_filter(kappa_e,1.5), origin='lower', 
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

    ax = axarr[-1]


    obs_probs = np.mean((1-fiducial) - probabilities ,axis=-1)
    obs_probs /= obs_probs.std()



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


    # In[27]:


    def sidm_sen( y ):
        return np.mean(y>0)
    def sidm_sen_std(y):
        return np.sqrt( np.mean(y>0))

    segment_size = 1
    bins = np.linspace(0.,0.5,20)

    corr = {}
    std = {}
    ncomp = 0
    for dataset in tqdm(["bahamas_cdm","bahamas_0.1","bahamas_0.3","bahamas_1"]):
        meta, data = pkl.load(open(f"../data/100/obs/concat/{dataset}.pkl","rb"))


        fiducial, sim_probabilities = pkl.load(open(f"pickles/sim_senstivity_{segment_size}_{dataset}.pkl","rb"))    

        rel_prob = np.array( [ np.mean(fiducial[:,i,0] - sim_probabilities[i], axis=-1) 
                              for i in range(sim_probabilities.shape[0]) if meta['ncomponents'][i] > ncomp])

        indexes = np.array([ i for i in  range(sim_probabilities.shape[0]) if meta['ncomponents'][i] > ncomp])

        rel_prob /= rel_prob.std()

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


        corr[dataset] = ycdm
        std[dataset] = ycdmstd/np.sqrt(count)

    obsfiducial, obsprobabilities = pkl.load(open(f"pickles/senstivity_1.pkl","rb"))
    obs_probs = np.mean((1-obsfiducial) - obsprobabilities ,axis=-1)
    obs_probs /= obs_probs.std()


    x_bin_me = obs_kappa_e.flatten()
    y_bin_me = obs_probs.flatten()



    y_bin_me = y_bin_me/np.abs(y_bin_me)
    remove = np.isfinite(y_bin_me)

    y_bin_me = y_bin_me[remove]
    x_bin_me = x_bin_me[remove]


    yobs, x, n = binned_statistic(
        x_bin_me, y_bin_me, bins=bins, statistic=sidm_sen)
    yobsstd, x, n = binned_statistic(
        x_bin_me,y_bin_me, bins=bins, statistic=sidm_sen_std)
    yobsn, x, n = binned_statistic(
        x_bin_me,y_bin_me, bins=bins, statistic='count')

    yobsstd /= np.sqrt(yobsn) 


    fig = plt.figure(figsize=(6,4))
    xc = (x[1:] + x[:-1]) / 2.
    ax = plt.gca()

    for dataset in corr.keys():
        ax.errorbar(xc, corr[dataset],
                    std[dataset], fmt='o-', capsize=2,
                lw=2, label=' '.join(dataset.split('_')).upper())

    ax.errorbar( xc, yobs, yobsstd, fmt='o-', color='k', label="Observation", capsize=2)
    leg = ax.legend()
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1.0)
    ax.set_xlabel(r'Weak lensing convergence, $\kappa$')
    ax.set_xlim(0,0.5)
    ax.set_ylim(0.25,0.75)
    top = ax.secondary_xaxis('top')
    top.set_xticks(ax.get_xticks())
    top.set_xticklabels([f"{int((t*lenspack.utils.sigma_critical(0.3,1.65,Planck18)).value)}"
                         for t in ax.get_xticks()])
    top.set_xlabel(r"M$_{\odot}$/pc$^2$")
    top.set_xlabel(r"M$_{\odot}$/pc$^2$")
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
    fname="plots/Sensitivty_relation.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")

# # Particle physics inference

def figure4and5():

    ifilter='concat'
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        *
        Matern(
            length_scale=0.3,
            length_scale_bounds=(1e-2, 10),
            nu=1.5
        )
        +
        WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e-1)
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4
    )

    models, probs = pkl.load(open("pickles/probs_for_cross_concat_nob1.pkl","rb"))
    models, probabilities, probabilities_noise =pkl.load(open("pickles/model_on_data.pkl","rb"))
    nmodels = probabilities[ifilter].shape[0]


    all_thresholds = []
    for imodel in range(nmodels):
        #Get the X and y values for a given model
        all_thresholds.append(1.-np.array([ np.mean(probs[i][imodel]) for i in probs.keys()]))

    all_thresholds = np.array(all_thresholds)
    thresholds = np.mean(all_thresholds,axis=0)
    err = np.std(all_thresholds,axis=0)

    cross = np.array([float(i) for i in probs.keys() ])

    cdm_thresh = thresholds[ cross == 0]

    err = err[ cross != 0 ]
    thresholds = thresholds[ cross != 0 ]
    cross = cross[ cross != 0]

    #cross[cross==0] = 1e-3
    cross = np.log10(cross)


    gp.fit(cross.reshape(-1, 1), thresholds)



    fig, axarr = plt.subplots(1,2,figsize=(8,3))

    fig.subplots_adjust(wspace=0.3)
    #####################
    handles = []
    ###PLOT 1

    #####################
    ax = axarr[0]
    handles.append(ax.errorbar(cross,thresholds,err,fmt='ko',capsize=2,label='Simulations'))
    #ax.plot( cdm_cross[-1], cdm_thresh[-1], 'k*', ms=10)

    xplot=np.linspace(-5.4,1,100)
    gp_prediction, std = gp.predict(xplot.reshape(-1, 1), return_std=True)

    ax.plot( xplot, gp_prediction,'k--')
    handles.append(ax.fill_between( xplot, gp_prediction-std, gp_prediction+std, alpha=0.1, color='k',label='Gaussian Process'))
    ax.plot( xplot, gp_prediction-std, color='k')
    ax.plot( xplot, gp_prediction+std, color='k')

    ax.set_ylim(0.5,0.62)
    ax.set_xlim(-2.5,0.2)
    ax.set_xlabel("log($\sigma_{\\rm DM}/m$)")
    ax.set_ylabel("Mean Model Output")

    #####################

    ###PLOT 2

    #####################
    ax = axarr[1]

    xpdf = np.linspace(0.4,0.8,1000)
    thresh = 1-probabilities['concat']
    correction = 2.08
    corr = correction*nmodels**0.5/nmodels**0.38

    ypdf = np.prod([ norm.pdf(xpdf, i, corr*np.std(probabilities['concat'])) for i in thresh ],axis=0)
    ypdf /= np.sum(ypdf)*(xpdf[1]-xpdf[0])

    ax = axarr[0].twiny()

    xpdf = np.linspace(0.45,0.7,1000)
    thresh = 1-probabilities[ifilter]
    correction = 2.0
    corr = correction*nmodels**0.5/nmodels**0.38

    ypdf = np.prod([ norm.pdf(xpdf, i, corr*np.std(probabilities[ifilter])) for i in thresh ],axis=0)
    ypdf /= np.sum(ypdf)*(xpdf[1]-xpdf[0])

    ax.plot(ypdf[::-1], xpdf[::-1], 'k-')
    handles.append(ax.fill_between(ypdf, xpdf, alpha=0.4, label=f'A2744 UNCOVER', color='b'))
    ax.set_xlim(0, ypdf.max()*1.1)
    ax.set_ylim(0.5, 0.65)
    ax.set_xticklabels([])
    ax.grid(False)
    ax.legend(handles=handles)
    #####################


    ###PLOT 3

    #####################

    ax = axarr[1]

    obs_thresh =  1-np.mean(probabilities['concat'])
    xpdf, obs_loglike  =  gp_invert( gp, obs_thresh )

    likelihood = np.exp(obs_loglike)

    #The prior is that CDM doesnt count so the best constraint
    #I have down to is min(cross)
    likelihood[ xpdf < 10**np.min(cross) ] = 0

    obs_loglike  =  np.array([ gp_invert( gp, i )[1] for i in thresh])
    likelihood = 10**logsumexp( obs_loglike,axis=0)


    ypdf  = likelihood / np.trapz(
        likelihood,
        x=np.log10(xpdf)
    )


    #xc = (x[1:] + x[:-1])/2.





    ax.set_ylim(ax.get_ylim())
    ax.set_xlim(1e-5, 5.)
    ax.set_xscale('log')


    cdm_x, cdm_y = gp_invert( gp, cdm_thresh )


    cdm_cross = cdm_x[np.argmax(cdm_y)]
    #ax.plot([cdm_cross,cdm_cross],[0,ax.get_ylim()[1]*1.1],'k--', label='CDM')
    ax.set_xlabel("$\sigma_{\\rm DM}/m$ [cm$^2$/g]")
    ax.set_ylabel("Posterior Likelihood")


    ### Final constraints
    max_like = xpdf[np.argmax(ypdf)]

    cdf = cumulative_trapezoid(
        ypdf,
        np.log10(xpdf),
        initial=0
    )[0]

    cdf /= cdf[-1]

    low_value = 0.16 #cdf[ np.argmax(ypdf)] - 0.34
    hig_value = 0.84 #cdf[ np.argmax(ypdf)] + 0.34


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

    ax.fill_between(
        xpdf, 
        ypdf[0], 
        alpha=0.4, color='blue')


    #ax.set_
    error = np.array([max_like - low, high - max_like])
    ax.plot(xpdf, ypdf[0],'k-',label=f"$\sigma/m={max_like:0.2f}^{{+{error[1]:0.2f}}}_{{-{error[0]:0.2f}}}$cm$^2$/g")

    #ax.plot(10**exp_method[0], exp_method[1], 'k--', label="Functional fit method")
    ax.legend()

    ax.set_ylim(0,1.)
    #ax.set_
    fig.align_ylabels()
    filename = "plots/output_to_model.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))  
    print(f"${max_like.real:0.2f}_{{-{error[0].real:0.2f}}}^+{{{error[1].real:0.2f}}}")

    
    fig = plt.figure(figsize=(7,4))
    ax = plt.gca()


    x = np.logspace(0,5)

    ax.plot( x, sigma_vd( x, 120,40),  lw=2, color='k', label='Nadler et al 2023')
    ax.plot( x, sigma_vd( x, 3,500),  '--', lw=2, color='k', label='BAHAMAS-VD')


    ax.set_ylabel('$\sigma_{\\rm DM}/m$ [cm$^2$/g]', fontsize=15, wrap=True)
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

    # Create top axis linked via transformation
    #top = ax.secondary_xaxis('top', functions=(v_to_m, m_to_v))
    #top.set_xscale('log')
    #top.set_xlabel('Corresponding halo mass [$\\rm M_\odot$]', fontsize=15)
    #plt.text(0.5, 1.5, 'Corresponding halo mass [$\\rm M_\odot$]', fontsize=15, 
    #         transform=top.transAxes, ha='center', va='bottom')


    bottom_ticks = ax.get_xticks()                   # e.g., [1, 10, 100, ...]
    #top_ticks = v_to_m(bottom_ticks)                 # positions on top axis that correspond
    #top.set_xticks(top_ticks)

    # Optional: set nice tick labels (e.g., log10 of mass) — choose whatever formatting you want:
    #top.set_xticklabels([f"{t:.1e}" for t in top_ticks])
    def sci_notation(x, pos):
        if x == 0:
            return "0"
        exp = np.log10(x)
        #pre = 10**(np.log10(x) - exp)

        return rf"$10^{{{exp:.01f}}}$"  # e.g., 10^10, 10^11

    #top.xaxis.set_major_formatter(FuncFormatter(sci_notation))
    #top.set_xlabel("Halo Mass $[M_\odot]$", fontsize=12)
    ax.tick_params(axis='both', labelsize=15)
    #top.tick_params(axis='both',  labelsize=15)

    plot_constraints( ax=ax, select_these=['Dwarf_correa21','cluster_harvey19','sagunski'],
                    labels={'Dwarf_correa21':'Correa (2021)','cluster_harvey19':'Harvey et al (2019)', 'sagunski':'Sagunski et al (2020)'})


    #plot_constraints( intermediate_constraints,upper_constraints, lower_constraints)

    velocity_disp = [522, 633]
    #velocity_disp = [1750, 1750]
    velocity_merg = [2000,2000]
    ax.plot( np.mean(velocity_disp), max_like, '*',
                markersize=20, color='red', lw=2, label='This Work (Velocity Disp)')

    ax.plot( np.mean(velocity_merg), max_like, '*',
                markersize=20, color='orange', lw=2, label='This Work (Merger Velocity)')

    ax.errorbar( np.mean(velocity_disp), max_like, yerr=error[:,None], 
                markersize=20, color='red', fmt='*', capsize=4, lw=2)

    ax.errorbar( np.mean(velocity_merg), max_like, yerr=error[:,None], 
                markersize=20, color='orange', fmt='*', capsize=4, lw=2)
    print(f"$\sigma_{{\\rm DM}}/m={max_like.real:0.2f}_{{-{error[0].real:0.2f}}}^{{+{error[1].real:0.2f}}}$cm$^2$/g")
    # get handles
    ax.legend()
    fname = "plots/particle_physics.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")

    
def figure6():

    filter_list = ['concat']
    fig, axarr=plt.subplots(len(filter_list),3, figsize=(14,4))
    nbins=50
    if len(filter_list ) == 1:
        axarr = axarr[None,:]

    for ifx, ifilter in enumerate( filter_list): 
        if ifilter=='concat':
            cat_a_name =     f"../data/100/obs/a2744_f115w_filtered.fits"
            cat_b_name =     f"../data/100/obs/a2744_f150w_filtered.fits"
            obs_data = combine_catalogues( cat_a_name, cat_b_name, identifier='NUMBER' )
        else:
            obs_data = fits.open(f"../data/100/obs/a2744_{ifilter}_filtered.fits")[1].data
        ax=axarr[ifx]

        x = np.linspace(0,1,nbins)
        data = np.sqrt(obs_data['e1']**2+obs_data['e2']**2)
        #ax[0].hist(data, density=True,bins=nbins,label='Data',color='g')
        ydata, x = np.histogram( data, bins=x,density=True)
        ax[0].stairs(
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

        ax[0].plot( xc, chi.pdf(xc, *chi.fit(data)), color='r',label='Chi',lw=2)
        ax[0].plot( xc, norm.pdf(xc, *norm.fit(data)), color='b',label='Gaussian',lw=2)



        ax[1].set_ylabel("Probability Distribution",fontsize=15)


        ax[0].set_xlabel("Absolute Ellipiticity",fontsize=15)
        ax[0].set_xlim(0,1.2)
        ax[0].legend(fontsize=15)

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
        ax[1].plot(
            ngal_range, intrinsic_ell, '-o', label='Measured'
        )
        ax[1].plot(
            ngal_range, intrinsic_ell[0]/np.sqrt(ngal_range), '-',label='1/sqrt(N)', lw=2
        )


        ax[1].set_xlabel("Number of galaxies in a bin",fontsize=15)
        ax[1].set_ylabel("Fitted intrinsic ell of chi fit",fontsize=15)
        ratio =  (intrinsic_ell[0]/np.sqrt(ngal_range)) / intrinsic_ell



        def g1_func( snr, a, b, c, d):

            return a + b*np.arctan( ( snr - c)/d)



        popt, pcov = curve_fit(
                        g1_func,
                        ngal_range,
                        ratio)


        ax[1].plot(
            ngal_range, intrinsic_ell[0]/(g1_func( ngal_range, *popt)*np.sqrt(ngal_range)), label='Fit', lw=2
        )
        ax[1].legend()
    ax[0].set_ylabel("Probabilitiy Distribution",fontsize=15)
    ax[2].set_ylabel("Probabilitiy Distribution",fontsize=15)

    ######
    stacked_rotated_modes,obs_kappa_b = pkl.load(open("pickles/bmode_check.pkl","rb"))
    bins = np.linspace(-1,1,50)

    y_noise_floor, x =np.histogram(stacked_rotated_modes,bins=bins, density=True)
    y_b_modes, x =np.histogram(obs_kappa_b,bins=bins, density=True)

    xc = (x[1:] + x[:-1])/2.

    ax = ax[-1]

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
    ax.text(0.31, 0.69, f"Residual B-mode={residual_b:0.2f}", 
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


    filename = "plots/intrinsic_ell.pdf"
    plt.savefig(filename)
    os.system("pdfcrop %s %s" % ( filename, filename))

    markersize = 12

   

    # # Appendix plots

    # ## Latent Space Overlap

    # In[98]:
# ## Galaxy selection checks

def figure7():

    alpha=0.5
    ylim = (0.5,0.62)
    models, probabilities, probabilities_noise, ngalaxies = pkl.load( open("pickles/model_on_data_galaxy_selection.pkl","rb"))

    ifilter = 'concat'
    all_mag_low_cuts = np.linspace(21,24,11)
    all_mag_cuts = np.linspace(28,26,11)
    all_siz_cuts = np.linspace(2,3,11)

    ngalaxies = ngalaxies/ngalaxies[0]

    fig = plt.figure(figsize=(16,3))
    gs = gridspec.GridSpec(1,27)

    ax = [
        plt.subplot(gs[0,:6]),
        plt.subplot(gs[0,6:12]),
        plt.subplot(gs[0,12:18]),
        plt.subplot(gs[0,19:])]


    ifilter='concat'
    fig.subplots_adjust(wspace=0.1)
    start_index = 0
    end_index = all_mag_low_cuts.shape[0]
    means =  np.mean(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    errors =  np.std(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    ngal = ngalaxies[start_index:end_index]

    fiducial = 1-np.mean(probabilities_noise[ifilter][:,0],axis=0) 


    ax[0].errorbar( all_mag_low_cuts, (1-means), errors, capsize=2, fmt='o-', color='k')
    ax[0].set_xlabel("Low Magnitude Cut")  
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel("Model Output")

    for limit in [1.0,0.9,0.8,0.7]:
        if np.sum( ngal < limit) == 0:
            continue

        dx = (all_mag_low_cuts[1] - all_mag_low_cuts[0])/2.
        x = np.linspace( 
            all_mag_low_cuts[ ngal <= limit][0]-dx,
            all_mag_low_cuts[ ngal <= limit][-1]+dx*3,
            100
        )
        y_low = np.zeros( x.shape[0])+ax[0].get_ylim()[0]
        y_hi = np.zeros( x.shape[0])+ax[0].get_ylim()[1]


        ax[0].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[0].text(
            all_mag_low_cuts[ ngal < limit][0]-dx/2., 0.61, f"{limit*100}\%", va='top', ha='left', rotation=90)

        ax[0].set_xlim(
            all_mag_low_cuts[0]-dx,
            all_mag_low_cuts[-1] +dx
        )

    start_index += all_mag_low_cuts.shape[0]
    end_index += all_mag_cuts.shape[0]
    means =  np.mean(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    errors =  np.std(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    ngal = ngalaxies[start_index:end_index]

    ax[1].errorbar( all_mag_cuts, (1-means), errors, capsize=2, fmt='o-', color='k')
    ax[1].set_xlabel("Upper Magnitude Cut")  
    ax[1].set_ylim(ylim)
    ax[1].set_yticklabels([])
    for limit in [1.0,0.9,0.8,0.7]:
        if np.sum( ngal < limit) == 0:
            continue

        dx = (all_mag_cuts[1] - all_mag_cuts[0])/2.
        x = np.linspace( 
            all_mag_cuts[ ngal <= limit][0]-dx,
            all_mag_cuts[ ngal <= limit][-1]+dx*3,
            100
        )
        y_low = np.zeros( x.shape[0])+ax[0].get_ylim()[0]
        y_hi = np.zeros( x.shape[0])+ax[0].get_ylim()[1]


        ax[1].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[1].text(
            all_mag_cuts[ ngal < limit][0]-dx/2., 0.61, f"{limit*100}\%", va='top', ha='left', rotation=90)

        ax[1].set_xlim(
            all_mag_cuts[0]-dx,
            all_mag_cuts[-1] +dx
        )
    start_index += all_mag_cuts.shape[0]
    end_index += all_siz_cuts.shape[0]
    means =  np.mean(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    errors =  np.std(probabilities_noise[ifilter][:,start_index:end_index],axis=0) 
    ngal = ngalaxies[start_index:end_index]

    ax[2].errorbar( all_siz_cuts, (1-means), errors, capsize=2, fmt='o-', color='k')
    ax[2].set_ylim(ylim)

    ax[2].set_xlabel("Size Cut")
    ax[2].set_yticklabels([])
    for limit in [1.0,0.9,0.8,0.7]:
        if np.sum( ngal < limit) == 0:
            continue

        dx = (all_siz_cuts[1] - all_siz_cuts[0])/2.
        x = np.linspace( 
            all_siz_cuts[ ngal <= limit][0]-dx,
            all_siz_cuts[ ngal <= limit][-1]+dx*3,
            100
        )
        y_low = np.zeros( x.shape[0])+ax[0].get_ylim()[0]
        y_hi = np.zeros( x.shape[0])+ax[0].get_ylim()[1]


        ax[2].fill_between( 
            x, y_low, y_hi, color='k', alpha=alpha*(1-limit)
        )
        ax[2].text(
            all_siz_cuts[ ngal < limit][0]-dx, 0.61, f"{limit*100}\%", va='top', ha='left', rotation=90)

        ax[2].set_xlim(
            all_siz_cuts[0]-dx,
            all_siz_cuts[-1] +dx
        )

    srcresults,tgtresults = pkl.load(open("pickles/all_models_concat_ngal_dep_results.pkl","rb"))

    color = {
            'src':colourFromRange([0,len(tgtresults.keys())], cmap='Reds'),
            'tgt':colourFromRange([0,len(tgtresults.keys())], cmap='Blues')
        }

    only_one = False
    ils=0
    ls = ['-','--',':','-.']
    ngalaxy_list = list(tgtresults.keys())[::-1]
    for irx, iresult in enumerate(ngalaxy_list):
        if float(iresult) < 0.6:
            continue

        if f"{float(iresult):0.1f}" == '1.0':
            if only_one:
                continue
            only_one=True
        ax[-1].plot(
            srcresults[iresult][0]+1e-3,
            (srcresults[iresult][1]-srcresults[iresult][1][0])*2.5+ 0.52,
            'o',ls=ls[ils],label=f"{int(np.round(float(iresult),1)*100)}\%", color=color['src'][len(ngalaxy_list) -irx])
        ils+=1
    ax[-1].set_xlabel("Self-Interaction Cross-Section [cm$^2$/g]")
    ax[-1].set_ylabel("Model Output")
    ax[-1].set_xscale('log')
    ax[-1].legend(title='$N/N_{g}$',loc=2)




    # --- gradient-colored 45-degree arrow (dark red -> light red) ---
    # Define start (tail/text side) and end (tip/arrowhead) in axes fraction coords.
    x0, y0 = 0.68, 0.55   # tail
    x1, y1 = 0.8, 0.15   # tip (where the arrow points)

    # Build a custom dark-red -> light-red colormap
    gradient_cmap = mcolors.LinearSegmentedColormap.from_list(
        "darkred_lightred", ["#5c0000", "#ff9999"]
    )

    n_seg = 100
    xs = np.linspace(x0, x1, n_seg)
    ys = np.linspace(y0, y1, n_seg)
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=gradient_cmap,
        norm=plt.Normalize(0, 1),
        linewidth=2,
        transform=ax[-1].transAxes,
        zorder=10,
    )
    lc.set_array(np.linspace(0, 1, n_seg))
    ax[-1].add_collection(lc)

    # Arrowhead at the tip, colored with the "light" end of the gradient
    arrow_head = FancyArrowPatch(
        (xs[-2], ys[-2]), (xs[-1], ys[-1]),
        transform=ax[-1].transAxes,
        arrowstyle='-|>',
        mutation_scale=30,
        color=gradient_cmap(1.0),
        linewidth=0,
        zorder=11,
    )
    ax[-1].add_patch(arrow_head)

    # Text label at the tail end
    ax[-1].text(
        x0, y0, "Reduced Sensitivity",
        transform=ax[-1].transAxes,
        fontsize=13, ha='right', va='bottom', color="#5c0000",
    )



    fig.align_ylabels()
    fname='plots/galaxy_selection.pdf'
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")


def figure8():
    
    latent = pkl.load(open("pickles/latent.pkl","rb"))
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    nmodels = len(latent['latent_space']) - 1
    nshow = 4

    sidm_list = [0.]
    #fig,axarr = plt.subplots(len(sidm_list),nshow, figsize=(nshow*3,3*len(sidm_list)))
    fig = plt.figure(figsize=(7,5))
    fig.subplots_adjust(hspace=0.01,wspace=0.01)

    for isx, sidm in enumerate(sidm_list):
        for imx, imodel in enumerate(range(nshow)):
            ax = plt.subplot(2,2,imx+1)

            A_latent =  latent['latent_space'][imodel+1][1]

            #[
            #        latent['all_cross_sections'][imodel+1][1] == sidm
            #]

            B_latent =  latent['latent_space'][imodel+1][0]

            #[
            #            latent['all_cross_sections'][imodel+1][0] == sidm
            #    ]     
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
                    density_a[0], density_a[1], density_a[2], cmap='Blues', label='DARKSKIES'
                )  
            else:
                fit = reducer.fit( B_latent )
            #ax.scatter(
            #    embedding_a[:, 0]-center[0],
            #    embedding_a[:, 1]-center[1],
            #    s=5,
            #    alpha=0.7
            #)

            if len(B_latent) !=0:


                embedding_b = fit.transform(B_latent)

                density_b =  get_density(
                    embedding_b[:, 0]-center[0], embedding_b[:, 1]-center[1]

                )
                ax.contour(
                    density_b[0], density_b[1], density_b[2], cmap='Reds', label='BAHAMAS'
                )  


            #ax.scatter(
            #    embedding_b[:, 0]-center[0],
            #    embedding_b[:, 1]-center[1],
            #    s=5,
            #    alpha=0.7
            #)

            embedding = fit.transform(latent['latent_space'][0][imodel].detach().numpy()[None,:])
            ax.plot(
                embedding[:, 0]-center[0],
                embedding[:, 1]-center[1],
                'y*',markersize=10
            )

            #ax.scatter(
            #    embedding_a[:, 0]-center[0],
            #    embedding_a[:, 1]-center[1],
            #    s=5,
            #    alpha=0.7
            #)


            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if imx == 0:
                ax.set_ylabel("UMAP-2")

            ax.set_xlabel("UMAP-1")
            ax.text(0.1,0.9,f"Model {1+imx}", transform=ax.transAxes, fontsize=15)

    fname="plots/latent_overlap.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")




    latent_distances = {}
    latent_quantiles = {}
    nmodels = len(latent['all_cross_sections'])-1
    latent_distances_tgt = [{},{}]
    latent_quantiles_tgt = [{},{}]
    for tgt in range(2):


        for imodel in range(nmodels):



                origin = latent['latent_space'][0][imodel].detach().numpy()

                model_cross_sections = latent['all_cross_sections'][imodel+1][tgt]
                latent_spaces = latent['latent_space'][imodel+1][tgt]

                unique_cross = np.unique(model_cross_sections)
                for icross in unique_cross:
                    if icross == -1:
                        raise
                    cross_lab = f"{icross:0.2f}"

                    if not cross_lab in latent_distances.keys():
                        latent_distances[cross_lab] = [[] for i in range( nmodels)]
                        latent_quantiles[cross_lab] = [[] for i in range( nmodels)]

                    if not cross_lab in latent_distances_tgt[tgt].keys():
                        latent_distances_tgt[tgt][cross_lab] = [[] for i in range( nmodels)]
                        latent_quantiles_tgt[tgt][cross_lab] = [[] for i in range( nmodels)]


                    sim_origin = torch.mean(latent_spaces,axis=0).detach().numpy()

                    obs_distance = np.sqrt(np.sum((origin-sim_origin)**2))
                    sim_distance = np.array([ np.sqrt(np.sum( (i.detach().numpy() - sim_origin)**2)) 
                                             for i in latent_spaces[icross == model_cross_sections]])

                    distance = obs_distance / sim_distance

                    latent_distances[cross_lab][imodel] += list(distance)
                    latent_quantiles[cross_lab][imodel] += list(np.quantile( distance, [0.16,0.84]))

                    latent_distances_tgt[tgt][cross_lab][imodel] += list(distance)
                    latent_quantiles_tgt[tgt][cross_lab][imodel] += list(np.quantile( distance, [0.16,0.84]))
    plt.figure(figsize=(5,4))

    ax = plt.gca()


    nmodels=30
    correction = 1
    unique_cross, latent_distances, latent_quantiles = pkl.load(open("pickles/latent_zs_space.pkl","rb"))


    av_distances = np.array([ np.mean([ np.mean(latent_distances[i][j]) for j in range(nmodels)]) for i in latent_distances.keys()] )
    err = np.array([ np.std([ np.mean(latent_distances[i][j]) for j in range(nmodels)])/np.sqrt(nmodels) for i in latent_distances.keys()] )

    unique_cross[0] = 1e-3
    #ax.errorbar( unique_cross, av_distances, err, color='k', fmt='o', capsize=2)
    #ax.fill_between(
    #        np.array(unique_cross).astype(float),
    #        av_distances-err, av_distances+err, color='k', alpha=0.3)

    color=['red','blue']
    domain = ['BAHAMAS','DARKSKIES']
    for tgt in range(2):
        cross =  np.array(list(latent_distances_tgt[tgt].keys())).astype(float)
        cross[ cross ==0] = 1e-3



        av_distances = np.array([ np.mean([ np.mean(latent_distances_tgt[tgt][i][j]) for j in range(nmodels)]) for i in latent_distances_tgt[tgt].keys()] )
        err = np.array([ np.std([ np.mean(latent_distances_tgt[tgt][i][j]) for j in range(nmodels)])/np.sqrt(nmodels) for i in latent_distances_tgt[tgt].keys()] )

        ax.errorbar(cross, av_distances, err, color=color[tgt], fmt='-o', capsize=2)

        ax.fill_between(
            cross,
            av_distances-err, av_distances+err, color=color[tgt], alpha=0.3, label=domain[tgt])
        print( list(latent_distances_tgt[tgt].keys()))


    ax.legend()
    ax.plot([5e-4,2],[1,1],'k--')
    ax.text(0.001,1.0, 'Simulation Mean', ha='left', va='bottom')
    ax.set_xlim(5e-4,2)
    ax.set_xscale('log')
    ax.set_ylabel("Latent Space Distance from Observation")
    ax.set_xlabel("Self-interaction cross-section [cm$^2$/g]")
    fname="plots/goodness_of_fit.pdf"
    plt.savefig(fname)
    os.system(f"pdfcrop {fname} {fname}")


# ## Model correlation



# ## Model systematics

def figure9():
    
    central_value = 0.51 # These are the central values of bahamas.
    central_error = 0.0083
    
    systematics = {
            'cluster_member_contamination': f"pickles/cluster_contamination_concat.pkl",
        'shape_measurement_bias':"pickles/shape_measurement_concat_test.pkl",
         'source_redshift_bias':"pickles/all_models_concat_nzbias_results.pkl",
        'projection_integration':"pickles/flamingo_dz_results.pkl",

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



    for isx, isystematic in enumerate(systematics.keys()):

        fiducial, all_results = pkl.load(open(systematics[isystematic],"rb"))

        print(isystematic)
        colors = colourFromRange([-1,len(all_results.keys())], cmap='Reds')
        ibias_keys = np.sort(list(all_results.keys()))

        color_norm = mpl.colors.Normalize(vmin=-1, vmax=len(all_results.keys()))
        cmap = plt.cm.Reds


        for ibx, ibias in enumerate(ibias_keys):
            all_thresholds = []
            color = cmap(color_norm(ibx))
            for imodel in all_results[ibias].keys():
                if 'seed' not in imodel:
                    continue

                if isinstance(all_results[ibias][imodel],dict):


                    tgt = get_threshold_for_cross( 
                        all_results[ibias][imodel]['src'], 
                        quiet=False)

                    fid = get_threshold_for_cross( 
                        fiducial[imodel]['src'], 
                        quiet=False)
                elif isinstance(all_results[ibias][imodel],list):

                    tgt = get_threshold_for_cross( 
                        all_results[ibias][imodel], 
                        quiet=False)
                    fid = get_threshold_for_cross( 
                        fiducial[imodel], 
                        quiet=False)

                all_thresholds.append((1-tgt['thresholds'])/(1-fid['thresholds']))

            all_thresholds = np.array(all_thresholds)

            means = np.nanmean(all_thresholds,axis=0)* central_value

            errors = np.std(all_thresholds,axis=0) / np.sqrt(all_thresholds.shape[0]) 

            ax.errorbar(  means[ tgt['cross_sections'] == 0], isx, xerr=errors[ tgt['cross_sections'] == 0], 
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

        sm = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1,len(ibias_keys)])
        if isystematic =='shape_measurement_bias':
            ibias_keys = ibias_keys.astype(float)
            ibias_keys[0] = 0.02
            ibias_keys[-1] = 0.00
        cbar.set_ticklabels([ibias_keys[0], ibias_keys[-1]])
        cbar.ax.tick_params(labelsize=12)


        systematic_names.append(isystematic.replace('_',' ').capitalize())

    for isy, isystematic in enumerate(systematics_thresh.keys()):

        fiducial, results = pkl.load(open(systematics_thresh[isystematic],"rb"))

        colors = colourFromRange([-1,len(results.keys())], cmap='Reds')

        ibias_keys = np.sort(list(results.keys()))
        colors = colourFromRange([-1,len(ibias_keys)], cmap='Reds')


        color_norm = mpl.colors.Normalize(vmin=-1, vmax=len(ibias_keys))
        cmap = plt.cm.Reds  
        for ibx, ibias in enumerate(ibias_keys):

            if 'thresholds' in list(results[ibias]['bahamas'].keys()):
                all_thresholds = (1-results[ibias]['bahamas']['thresholds'])/(1-fiducial['bahamas']['thresholds']) 
                cross_sections =  results[ibias]['bahamas']['cross_section']


                means = np.mean(all_thresholds,axis=0)* central_value

                errors = np.std(all_thresholds,axis=0)/np.sqrt(all_thresholds.shape[0])


            else:
                means = (1-results[ibias]['bahamas']['mean'])/(1-fiducial['bahamas']['mean']) * central_value
                errors = results[ibias]['bahamas']['err']/fiducial['bahamas']['mean'] * central_value


            ax.errorbar(  means[cross_sections==0], isy+isx+1, 
                        xerr = errors[cross_sections==0], 
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
        sm = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1,len(ibias_keys)])
        if isystematic =='merging_scenario':
            ibias_keys = ibias_keys.astype(float)
            ibias_keys += 1

        if isystematic == 'mass_dependency':
            ibias_keys = ibias_keys.astype(str)
            ibias_keys[0] = r"$-\sigma$"
            ibias_keys[-1] = r"$+\sigma$"


        cbar.set_ticklabels([ibias_keys[0], ibias_keys[-1]])
        cbar.ax.tick_params(labelsize=12)


        systematic_names.append(isystematic.replace('_',' ').capitalize())
    ax.set_yticks(np.arange(len(systematic_names)))
    ax.set_yticklabels(systematic_names, fontsize=15)   


    model, probs, nois = pkl.load(open( "pickles/model_on_data.pkl","rb"))

    obs_results = 1-np.mean(probs['concat'])
    obs_err = np.std(probs['concat'])/probs['concat'].shape[0]**0.38*corr
    #ax.errorbar( [obs_results, obs_results],  [-1,len(systematic_names)+1], color='k')
    #ax.fill_between( 
    #    [obs_results-obs_err, obs_results+obs_err],
    #    [-1,-1], [len(systematic_names)+1]*2, color='k', alpha=0.2
    #)
    #ax.plot( [obs_results-obs_err, obs_results-obs_err],  [-1,len(systematic_names)+1], ':', color='k')
    #ax.plot( [obs_results+obs_err, obs_results+obs_err],  [-1,len(systematic_names)+1],  ':',color='k')

    #ax.text( obs_results, len(systematic_names)/2., "Observation", ha='right', va='top', rotation=90, fontsize=15)


    ax.plot( [central_value, central_value],  [-1,len(systematic_names)+1], color='k', label='Fiducial')
    ax.plot( [central_value+central_error, central_value+central_error],  [-1,len(systematic_names)+1], ':', color='k')
    ax.plot( [central_value-central_error, central_value-central_error],  [-1,len(systematic_names)+1],  ':',color='k')
    ax.legend(loc=4,fontsize=15)
    ax.fill_between( 
        [central_value-central_error, central_value+central_error],
        [-1,-1], [len(systematic_names)+1]*2, color='k', alpha=0.1
    )
    ax.text(1.15,1.0,r"\underline{Range}",transform=ax.transAxes, fontsize=15, ha='center',va='top')

    ax.text(-0.15,1.0,r"\underline{Systematic}",transform=ax.transAxes, fontsize=15, ha='center',va='top')

    ax.set_ylim(-0.5,len(systematic_names))
    ax.set_xlabel("Model Output",fontsize=15)
    fname="plots/summarised_systematics.png"
    fig.canvas.draw()
    plt.savefig(fname)
    
    
if __name__ == "__main__":
    figure9()

    