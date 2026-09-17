#!/usr/bin/env python

from get_model_probabilities import get_mass_cut, get_threshold_for_cross, pkl, tqdm, get_massfunction_weights, np
import os
from add_shear_to_data import get_obs_data, get_model_names, get_source_redshift, sig_mean, get_lens_info

def main(
    ):
    cluster_name = os.getcwd().split('/')[-2]
    cluster_info = get_lens_info( cluster_name )
    filter_list = [ 'concat' if len(cluster_info['filter_list']) > 1 else cluster_info['filter_list'][0] ]




    for ifx, ifilter in enumerate(filter_list):
        results_file = f"pickles/all_models_{ifilter}_nz_alignbest_wb1_results.pkl"
    
        all_results = pkl.load(open(results_file,"rb"))
    
        
        positive_mass, mass_cut = get_mass_cut( ifilter, study='harvey', nsigma=2)
        
    
        domain = {
            'tgt':'darkskies_obs',
            'src':'bahamas_obs'
        }
        model_index = {}
    
        probs_for_cross = {}
    
        for imx, imodel in tqdm(enumerate(all_results.keys())):
            
    
            imodel_probs = {}
            imodel_index = {}
    
            for target in all_results[imodel].keys():
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
                
                for idx, icross in enumerate(tgt['cross_sections']):
                    keyname = f"{icross:0.2f}"
                    if keyname not in imodel_probs.keys():
                        imodel_probs[ keyname ] = []
                        imodel_index[ keyname ] = []
    
                    imodel_probs[keyname].append(
                        tgt['probabilities'][idx][:,0]
                    )
                    imodel_index[keyname].append(tgt['indexes'][idx])
    
            for ikey in imodel_probs.keys():
                if ikey not in probs_for_cross:
                    probs_for_cross[ikey] = []
                    model_index[ikey] = []
    
    
                model_index[ikey].append(np.hstack(imodel_index[ikey]))
                probs_for_cross[ikey].append(np.hstack(imodel_probs[ikey]))
    
    
        
        pkl.dump([model_index,probs_for_cross], open(f"pickles/probs_for_cross_{ifilter}_nob1.pkl","wb"))


if __name__ == "__main__":
    main()