#!/usr/bin/env python


from add_shear_to_data import get_obs_data, get_model_names, get_source_redshift, sig_mean
from get_model_probabilities import *
import os

import scienceplots
plt.style.use(["science","grid"])

def main(
    imonte=5,
    zs=1.
    ):
    ifilter = 'concat'
    cuts= {'f115w':{'signal_noise_cut':0, 'stat_type':'snr'}, 
       'f150w':{'signal_noise_cut':5, 'stat_type':'snr'}}
    obs_data = get_obs_data(ifilter, data_dir="../data/100/a2744/", 
                            photoz=True,
                           cuts=cuts)


    fid_n_z = get_variable_redshift_dist(data_dir="../data/100/a2744/")
    

    args.zs = fid_n_z
    args.source_domain = 'darkskies_obs'

    cluster_name = os.getcwd().split('/')[-2]
    for ifilter in ['concat']:
        
        
        all_models, seed_index =  get_model_names(model='tweakedalign_best', include_str='nob1', ifilter=ifilter) 
    
        
        output_name = f"pickles/all_models_{ifilter}_nz_alignbest_wb1_results.pkl"
        
        if os.path.isfile(output_name):
            all_results = pkl.load(open(output_name,'rb'))
        else:
            all_results = {}
             
        for imodel in tqdm(all_models):
    
            seed = imodel.split('_')[seed_index]
    
            args.jwst_filter = ifilter
            args.apply_intrinsic_ell = 1.
                
            domain = {
            'tgt':'darkskies_obs',
            'src':'bahamas_obs'
            }
            
            
            args.ignore_dataset = [''] # Although i ignored during training i want to see duringn testing.
                               
            if f"seed_{seed}" not in all_results.keys():
                
                all_results[f"seed_{seed}"]  = {'src':[],'tgt':[]}
    
            args.unbalance = True
            ndone = len( all_results[f"seed_{seed}"]['src']) 
            for i in range(ndone,imonte):
                
                redshift_select = obs_data['redshift'][ 
                    np.abs(obs_data['redshift']-args.default_zs) > 1e-3
                ]            
                populate_redshift = redshift_select[
                                np.random.randint(
                                    low=0, 
                                    high=redshift_select.shape[0], 
                                    size=sum(~np.isfinite(fid_n_z.flatten())))
                            ]
                n_z = fid_n_z.copy()
                n_z[ ~np.isfinite(fid_n_z)] = populate_redshift
                                             
                args.zs = n_z
                                                                
                for idomain in domain.keys():
    
                    target_domain = domain[idomain]
                    results = get_probabilities( 
                            target_domain,
                            [imodel],
                            args,
                            quiet=True
                    )
                    del results['data_loaders']
    
                    all_results[f"seed_{seed}"][idomain].append( results )
    
            pkl.dump(all_results, open(output_name,"wb"))


if __name__ == "__main__":
    main()