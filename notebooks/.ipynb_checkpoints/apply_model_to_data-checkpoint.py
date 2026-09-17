#!/usr/bin/env python
from get_model_probabilities import *

from add_shear_to_data import get_obs_data, get_model_names, get_source_redshift, sig_mean, get_lens_info

def main():

    cluster_name = os.getcwd().split('/')[-2]
    cluster_info = get_lens_info( cluster_name )
    filter_list = [ 'concat' if len(cluster_info['filter_list']) > 1 else cluster_info['filter_list'][0] ]
    #RUN ALL MODELS ON THE TEST DATA
    #-------------------------------
    
    models = {}
    probabilities = {}
    probabilities_noise = {}
    for ifx, ifilter in enumerate(filter_list):
        
        _, stacked = pkl.load(open(f"../data/100/observations/obs_data_{ifilter}.pkl","rb"))

        all_models, seed_index =  get_model_names(model='best', 
                                                  load_model_args=args,
                                                   include_str='nob1',
                                                  ifilter=ifilter
                                                 )              
        
        print(f"Found {len(all_models)} Models")
        models[ifilter] = all_models
    
        #repeat 
        probabilities_filt = []
        probabilities_noise_filt = []
    
        with torch.no_grad():
    
            for imodel in tqdm(all_models):
                outputs_dict = imodel([torch.tensor(stacked,dtype=torch.float32)])
                probabilities_filt.append(torch.softmax( outputs_dict['classification'], dim=1 )[0,0]) 
                probabilities_noise_filt.append(torch.softmax( outputs_dict['classification'], dim=1 )[1:,0])
                
    
        probabilities_filt = torch.tensor( probabilities_filt).detach().numpy()
        probabilities_noise_filt =  torch.stack( probabilities_noise_filt).detach().numpy()
    
            
        probabilities[ifilter] = probabilities_filt
        probabilities_noise[ifilter] = probabilities_noise_filt
    pkl.dump([models, probabilities, probabilities_noise], open("pickles/model_on_data.pkl","wb"))

if __name__ == "__main__":
    main()