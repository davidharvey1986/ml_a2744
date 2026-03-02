import glob
import os
import pickle
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from collections import defaultdict
from lenspack.utils import sigma_critical
from astropy import cosmology
from scipy.stats import chi, norm

def get_cross_section_from_filename(filename):
    base_name = os.path.basename(filename)
    name = os.path.splitext(base_name)[0].lower()
    if "cdm" in name or "flamingo" in name:
        return 0.0
    matches = re.findall(r"_(\d+\.\d+|\d+)", name)
    if matches:
        return float(matches[0])
    return 0.0

def cross_section_to_binary_label(cross_sections):
    return (cross_sections > 0).long()

class BalancedClusterDataset(Dataset):
    def __init__(self, data_dir, file_pattern, transform=None, normalization_stats=None, args=None, dtype='image'):
        
        self.data = []
        
        self.cross_sections = []
        #Only allow augmetnation for images not catalogue data
        if dtype != 'image':
            self.transform = None 
        else:
            self.transform = transform
            
        self.normalization_stats = [-0.5, 0.5] #normalization_stats
        self.args = args
        self.file_indices = []
        self.files_data = {}
        
        self.by_img_idx = defaultdict(list)
        
        if (dtype=='meta') & (args.meta_names is None):
            raise ValueError("Dtype is meta but no meta names given")
        
        files = glob.glob(os.path.join(data_dir, file_pattern))
        if not files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in directory '{data_dir}'")
        if args.verbose:
            print(f"Loading {len(files)} files matching '{file_pattern}'...")
        
        for file_idx, file_path in enumerate(files):
            file_name = file_path.split('/')[-1]
            if file_name in args.ignore_dataset:
                if args.verbose:
                    print(f"Ignoring {file_path}")
                continue
                
            with open(file_path, "rb") as f:
         
                meta, all_images = pickle.load(f)
                cross_section = get_cross_section_from_filename(file_path)
                
                if 'mass' not in list(meta.keys()):
                    if args.verbose:
                        print(f"NO MASS META FOR {file_path} FOUND SO NO CUT APPLIED")
                    meta['mass'] = np.zeros( all_images.shape[0])+100
                    
                mass_cut = meta['mass'] > args.log_mass_cut

                if args.verbose:
                    print(f"Cutting {sum(~mass_cut)}")
                    
                images = all_images[:, args.mass_index:args.mass_index+args.in_channels]
                
                if args and args.use_log_transform:
                    images = np.log10(images * meta['norms'][:, 0:1, None, None])
                
                self.files_data[file_idx] = {
                    'images': images,
                    'cross_section': cross_section,
                    'filename': os.path.basename(file_path)
                }
                
                all_img_idx = np.arange(len(all_images))
                
                for idx, img_idx in enumerate(all_img_idx[mass_cut]):
                    global_idx = len(self.data)
                  
                    if dtype == 'image':
                        self.data.append(images[img_idx])
                    else:
                        self.data.append(np.atleast_1d([meta[dtype][img_idx] for dtype in args.meta_names]))
                        
                    if np.any(~np.isfinite(images[img_idx])):
                        print(file_path)
                        raise ValueError("Nans found in data")
                        
                    self.cross_sections.append(cross_section)
                    self.file_indices.append((file_idx, img_idx))
                    self.by_img_idx[img_idx].append(global_idx)
  
        self.data = np.array(self.data)
        self.cross_sections = np.array(self.cross_sections)


        binary_labels = cross_section_to_binary_label(torch.tensor(self.cross_sections))
        unique, counts = torch.unique(binary_labels, return_counts=True)
        if args.verbose:
            print(f"Data shape: {self.data.shape}")
            print(f"Unique cross-sections: {np.unique(self.cross_sections)}")
           
            print(f"Classification class distribution:")
            for class_idx, count in zip(unique, counts):
                print(f"  Class {class_idx.item()}: {count.item()} samples ({count.item()/len(binary_labels)*100:.1f}%)")

    def __len__(self):
        return len(self.cross_sections)

    def __getitem__(self, idx):
        image = self.data[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32)
        


        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
   
        if self.normalization_stats is not None and self.args and self.args.use_normalization:
            minvalue, maxvalue = self.normalization_stats
            #image_tensor = (image_tensor - mean) / std
        else:
            minvalue = torch.min(image_tensor)
            maxvalue = torch.max(image_tensor)
       
        image_tensor -= minvalue
        image_tensor /= maxvalue 
        
        #Make sure all the information we have no idea about == 0.5
        if self.args.med_norm >0:
            image_tensor[ image_tensor==torch.median(image_tensor)] = self.args.med_norm
        
        cross_section = torch.tensor(self.cross_sections[idx], dtype=torch.float32)
        file_idx, img_idx = self.file_indices[idx]

        binary_label = cross_section_to_binary_label(cross_section.unsqueeze(0)).squeeze(0)
        if torch.any(~torch.isfinite(image_tensor)):
            print(np.sum(image), file_idx,img_idx)
            raise ValueError("NAN")
        return image_tensor, cross_section, binary_label, file_idx, img_idx

    
    def get_same_position_candidates(self, file_idx, img_idx):
        candidates = []
        for global_idx in self.by_img_idx[img_idx]:
            cand_file_idx, _ = self.file_indices[global_idx]
            if cand_file_idx != file_idx:
                candidates.append(global_idx)
        return candidates
        
    @staticmethod
    def compute_normalization_stats(dataset):
        all_data = []
        if isinstance(dataset, Subset):
            original_dataset = dataset.dataset
            indices = dataset.indices
            for idx in indices:
                all_data.append(original_dataset.data[idx])
        else:
            for idx in range(len(dataset)):
                all_data.append(dataset.data[idx])

        all_data = np.stack(all_data)
        mean = np.mean(all_data)
        std = np.std(all_data)
        
        if std == 0:  
            std = 1.0
                
        return mean, std

def create_balanced_test_set(dataset, test_size=0.2, random_state=42, verbose=True, unbalance=False):
    binary_labels = cross_section_to_binary_label(torch.tensor(dataset.cross_sections))

    class_0_indices = torch.where(binary_labels == 0)[0].numpy()
    class_1_indices = torch.where(binary_labels == 1)[0].numpy()
    if verbose:
        print(f"Class distribution:")
        print(f"  Class 0: {len(class_0_indices)} samples")
        print(f"  Class 1: {len(class_1_indices)} samples")
    
    total_test_samples = int(test_size * len(dataset))
    samples_per_class = total_test_samples // 2
    samples_per_class = min(samples_per_class, len(class_0_indices), len(class_1_indices))
    if verbose:
        print(f"Creating balanced test set with {samples_per_class} samples per class ({samples_per_class * 2} total)")
    
    np.random.seed(random_state)
    if not unbalance:
        test_0_indices = np.random.choice(class_0_indices, size=samples_per_class, replace=False)
        test_1_indices = np.random.choice(class_1_indices, size=samples_per_class, replace=False)
    else:
        n0_idx = len(class_0_indices)
        n1_idx = len(class_1_indices)
        test_0_indices = np.random.choice(class_0_indices, size=int(n0_idx * test_size), replace=False)
        test_1_indices = np.random.choice(class_1_indices, size=int(n1_idx * test_size), replace=False)
    
    test_indices = np.concatenate([test_0_indices, test_1_indices])
    np.random.shuffle(test_indices)
    
    test_indices = test_indices.tolist()
    
    test_labels = binary_labels[test_indices]
    test_unique, test_counts = torch.unique(test_labels, return_counts=True)
    if verbose:
        print(f"Balanced test set distribution:")
        for class_idx, count in zip(test_unique, test_counts):
            print(f"  Class {class_idx.item()}: {count.item()} samples ({count.item()/len(test_labels)*100:.1f}%)")

    return test_indices

def prepare_dataloaders(args):
    data_loaders = {}
    #Note that all data is in idealised form - there are transforms to forward model the data.
    #Generally the folders contain
    #data/convergence - projected maps only where the first index is total mass
    #data/shear - shear (not normalised as this is done in the __getitem__ fct as first two indexes with an assumed zl and zs - can be converted in transforms
    #data/obs - shear sampled at the positions of a2744 with an assumed zl and zs - can be converted in transforms
    #data/a2744 - all the data from JWST a2744 unconvers data
    
    domain_patterns = {
        "bahamas":"shear/bahamas*", #idealised simulations that include shear - main source training data
        "darkskies":"shear/darkskies*", #idealised simulations that include shear - main target data
        "test":"obs/test_set/*", #independent test sets including tng, flamingo and darkskies 0.07
        "darkskies_obs":f"obs/{args.jwst_filter}/darkskies*", #idealised simulations projected on to the source galaxy positions of A2744 
        "bahamas_obs":f"obs/{args.jwst_filter}/bahamas*", #idealised simulations projected on to the source galaxy positions of A2744 
        "a2744":f"a2744/obs_data_{args.jwst_filter}*", #actual data as measured from a2744 JWST with 1000 rotated noise configurations
        "highmass":"obs/highmass/darkskies*", # halos with a mass > 8e14
        "tng_obs":f"obs/{args.jwst_filter}/tng*",
        "flamingo_obs":f"obs/{args.jwst_filter}/flamingo*",
    }

    if args.source_domain not in domain_patterns or args.target_domain not in domain_patterns:
        raise ValueError(f"Source and target domains must be one of: {list(domain_patterns.keys())}")

    source_pattern = domain_patterns[args.source_domain]
    target_pattern = domain_patterns[args.target_domain]

    if args.aug_rotation_prob > 0:
        train_transform = transforms.Compose([
            rescale_lens_source_configuration(args),
            apply_intrinsic_ell( args.apply_intrinsic_ell, args.jwst_filter),
            transforms.RandomHorizontalFlip(args.aug_h_flip_prob),
            transforms.RandomVerticalFlip(args.aug_v_flip_prob),
            transforms.Pad( int( args.image_size//2*(np.sqrt(2)-1)), padding_mode='reflect'),
            transforms.RandomApply([transforms.RandomRotation(degrees=args.aug_rotation_degrees, fill=0)], p=args.aug_rotation_prob),
            transforms.CenterCrop( size=(args.image_size, args.image_size)),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=(args.image_size, args.image_size), 
                                           scale=(args.aug_crop_scale_min, args.aug_crop_scale_max), 
                                           ratio=(1.0, 1.0))
            ], p=args.aug_crop_prob),
        ])
    else:
        train_transform = transforms.Compose([
            rescale_lens_source_configuration(args),
            apply_intrinsic_ell(  args.apply_intrinsic_ell, args.jwst_filter),
            transforms.RandomHorizontalFlip(args.aug_h_flip_prob),
            transforms.RandomVerticalFlip(args.aug_v_flip_prob),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=(args.image_size, args.image_size), 
                                           scale=(args.aug_crop_scale_min, args.aug_crop_scale_max), 
                                           ratio=(1.0, 1.0))
            ], p=args.aug_crop_prob),
        ]) 
        
    test_transform = transforms.Compose([
            rescale_lens_source_configuration(args),
            apply_shape_measurement_bias(args.shape_measurement_bias),
            apply_intrinsic_ell( args.apply_intrinsic_ell,  args.jwst_filter),
    ])
    
    temp_source_dataset = [
        BalancedClusterDataset(args.data_dir, source_pattern, args=args, dtype=i, transform=train_transform) for i in args.dtypes
    ]
    
    
    source_test_indices = create_balanced_test_set(
        temp_source_dataset[0], 
        test_size=(1 - args.train_split),
        random_state=args.seed,
        verbose=args.verbose,
        unbalance=args.unbalance
    )
    all_source_indices = set(range(len(temp_source_dataset[0])))
    source_test_indices_set = set(source_test_indices)
    source_train_indices = list(all_source_indices - source_test_indices_set)
    if args.verbose:
        print(f"Source domain setup:") 
        print(f"  Training samples: {len(source_train_indices)}")
        print(f"  Test samples: {len(source_test_indices)}")
        
    temp_source_train_dataset = [
        Subset(i, source_train_indices) for i in temp_source_dataset
    ]
        
    source_stats = BalancedClusterDataset.compute_normalization_stats(temp_source_train_dataset[0])
    if args.verbose:
        print(f"Source dataset normalization - Mean: {source_stats[0]:.4f}, Std: {source_stats[1]:.4f}")
    
    source_dataset_train = [
        BalancedClusterDataset(args.data_dir, source_pattern, transform=train_transform, 
                                                 normalization_stats=source_stats, args=args, dtype=i)
        for i in args.dtypes
    ]
        
    source_dataset = [
        BalancedClusterDataset(args.data_dir, source_pattern, normalization_stats=source_stats, 
                                           args=args, dtype=i, transform=test_transform)
        for i in args.dtypes
    ]
    
    target_dataset_train = [
        BalancedClusterDataset(args.data_dir, target_pattern, transform=train_transform, 
                                                 normalization_stats=source_stats, args=args, dtype=i)
        for i in args.dtypes
    ]
    target_dataset = [
        BalancedClusterDataset(args.data_dir, target_pattern, normalization_stats=source_stats, 
                                           args=args, dtype=i, transform=test_transform)
        for i in args.dtypes
    ]
        

    source_train_dataset = [
        Subset(i, source_train_indices) for i in source_dataset_train
        
    ]
    source_test_dataset = [ 
        Subset(i, source_test_indices) for i in source_dataset
    ]

    target_test_indices = create_balanced_test_set(
        target_dataset[0], 
        test_size=(1 - args.train_split),
        random_state=args.seed,
        verbose=args.verbose,
        unbalance=args.unbalance
    )

    all_target_indices = set(range(len(target_dataset[0])))
 
    target_test_indices_set = set(target_test_indices)
    target_train_indices = list(all_target_indices - target_test_indices_set)
    if args.verbose:
        print(f"Target domain setup:")
        print(f"  Training samples: {len(target_train_indices)}")
        print(f"  Test samples: {len(target_test_indices)}")
    
    target_train_dataset = [
        Subset(i, target_train_indices) for i in target_dataset_train
    ]
    target_test_dataset = [
        Subset(i, target_test_indices) for i in target_dataset
    ]

    data_loaders["source_train"] = [
        DataLoader(
        i, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=True
    ) for i in source_train_dataset
    ]
        

    data_loaders["source_val"] = [ DataLoader( 
        i, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    ) for i in source_test_dataset]

    data_loaders["target_train"] = [ DataLoader(
        i, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=True
    ) for i in target_train_dataset]

    data_loaders["target_test"] = [ DataLoader(
        i, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    ) for i in target_test_dataset]
    if args.verbose:
        print(f"DataLoader summary:")
        print(f"  Source train: {len(source_train_dataset[0])} samples")
        print(f"  Source test: {len(source_test_dataset[0])} samples") 
        print(f"  Target train: {len(target_train_dataset[0])} samples")
        print(f"  Target test: {len(target_test_dataset[0])} samples")

    return data_loaders




    
class apply_intrinsic_ell(object):
    
    
    def __init__(self, apply, jwst_filter):
        '''
        This is NOT A GAUSSIAN - it is s fitted chi in ellpiticituy
        then randomly select rotations
        
        It fits directly to the data therefore intrinsic ell is simply
        a switch that turns it on or off
        
        See section 4.) in add_shear.ipynb for fitting
        See section 2.) in prepare_data for the dump of ngal_{jwst_filter}.pkl 
                        which is the number density of gals and the chi fit
                        with scipy
        '''
        
        home="/Users/davidharvey/Work/Dark_Matter_DA/code/notebooks/pickles"
        #Intrinsic ell here is the intrinsic ellipticity for a single component.
        data_dict = pickle.load( open(f"{home}/ngal_{jwst_filter}.pkl","rb") )
        self.ngal = data_dict['ngal']
        self.chi_fit = data_dict['chi_fit_function']
        self.apply = apply

        
    def __call__( self, sample):
        #assume a-b/a+b
        if self.apply != 0.:
            
        
            # Techincally it doesnt go as 1/sqrt(N) but in add_shear_to_data.ipynb 
            # i boost the galaxy density by the factor**2 so here it is simple 1/sqrt(N)
            scale = self.chi_fit[2]/np.sqrt(self.ngal)*self.apply 
        
            source_ell = torch.tensor( chi.rvs( 
                df=self.chi_fit[0], 
                loc=self.chi_fit[1], 
                scale=scale
            ), dtype=torch.float32)
        
            source_chi = 2.*source_ell / (1+source_ell**2) # do everything in chi not epsilon
            
            source_the = torch.rand(
                sample[0].shape
            )*np.pi
        
            source_chi[ self.ngal ==0 ] = 0
    
            source_shape = torch.cat(
                [source_chi[None,:,:]*np.cos(2.*source_the[None,:,:]),
                 source_chi[None,:,:]*np.sin(2.*source_the[None,:,:])]
            )
        else:
            source_shape = sample * 0.
            
        #sample_conj = sample.clone()
        #sample_conj[1] *= -1
        
        source_conj = source_shape.clone()
        source_conj[1] *= -1 

        g = torch.sqrt(torch.sum(sample**2,axis=0))
    
        g_shapeconj = sample*source_conj
        #return_sample = (sample + source_shape) / \
        #    ( 1 + source_shape*sample_conj)
        
        return_sample = (source_shape + 2.*sample + sample**2*source_conj) / \
            (1 + g**2 + 2.*g_shapeconj[0])
                

        return_sample[ (sample == 0) ] = 0

        return return_sample
        
        
class apply_shape_measurement_bias( object ):
    
    def __init__(self, bias ):
        self.bias = bias
        
    def __call__( self, sample ):
        
        sample[0] += self.bias['e1']['c'] + self.bias['e1']['m']*sample[0]        
        sample[1] += self.bias['e2']['c'] + self.bias['e2']['m']*sample[1]
        
        return sample
  

        
        
class rescale_lens_source_configuration( object ):
    '''
    When creating the data it assumes that zs=1.36, zl=0.15
    '''
    def __init__(self, args ):
        self.zl = args.zl
        self.zs = args.zs
        
        if args.verbose:
            print(f"SOURCE REDSHIFT: {args.zs}")
        default_zl = 0.305
        default_zs = 1.65
        
        if (args.apply_intrinsic_ell == 0) or ( (args.zl==default_zl) & (args.zs==default_zs)):
            self.apply = False
        else:
            self.apply = True
            
            sigma_crit_fid = 2354. # hard coded to speed up
            
            sigma_crit = sigma_critical(self.zl, self.zs, cosmology.Planck18).value
            
            self.rescale = sigma_crit_fid/sigma_crit
            
            
    def __call__( self, sample):
        
        if self.apply:
            
            sample *= self.rescale
            
            return sample
        else:
            return sample