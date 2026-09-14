#!/usr/bin/env python


from add_shear_to_data import get_obs_data, get_model_names, get_source_redshift, sig_mean, get_lens_info
from get_model_probabilities import *
import os

import scienceplots
plt.style.use(["science","grid"])

def main(
    imonte=5,
    zs=1.
    ):
    args.zs = 1 
    args.source_domain = 'darkskies_obs'

    cluster_name = os.getcwd().split('/')[-2]
    cluster_info = get_lens_info( cluster_name )
    filter_list = [ 'concat' if len(cluster_info['filter_list']) > 1 else cluster_info['filter_list'][0] ]
    for ifilter in filter_list:
        
        
        all_models, seed_index =  get_model_names(model='best', include_str='nob1', ifilter=ifilter) 
    
        
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