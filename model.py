import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights, resnet18, ResNet34_Weights, resnet34,
    MobileNet_V3_Small_Weights, mobilenet_v3_small,
    SqueezeNet1_1_Weights, squeezenet1_1,
)
from adaptation import DomainDiscriminator, GradientReverseLayer, MultiLinearMap
from netloader.network import Network


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(SmallCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, 
                              groups=base_channels*4, padding=1, bias=False)
        self.conv4_pointwise = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels*4)
        
        self.conv5 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(base_channels*4)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_dim = base_channels * 4
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))        
        x = F.relu(self.bn2(self.conv2(x)))        
        x = F.relu(self.bn3(self.conv3(x)))        
        
        x = self.conv4(x)                          
        x = self.conv4_pointwise(x)                
        x = F.relu(self.bn4(x))                   
        
        x = F.relu(self.bn5(self.conv5(x)))        
        
        features = self.adaptive_pool(x)          
        features = features.view(features.size(0), -1)  
        
        return features

class Model(nn.Module):
    def __init__(self, 
                 model_name="resnet18", 
                 pretrained=True, 
                 in_channels=1, 
                 args=None):
        super(Model, self).__init__()
        self.device = args.device
        self.num_meta_info = len(args.meta_names)
        base_channels = args.cnn_base_channels if args else 32        
        
        
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif model_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            feature_dim = 576
            if in_channels != 3:
                backbone.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
            
        elif model_name == "squeezenet1_1":
            weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
            backbone = squeezenet1_1(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
                
            self.backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))
            
        elif model_name == "small_cnn":
            self.backbone = SmallCNN(in_channels=in_channels, base_channels=base_channels)
            feature_dim = base_channels * 4
            
        elif model_name == "ethan":
            self.backbone = Network(
                "network_v10_encoder.json",
                "/Users/davidharvey/Work/Dark_Matter_DA/code/ethan_models",
                (in_channels, args.image_size, args.image_size),  # excluding batch size, so (C,H,W)
                (base_channels * 4,),  # again, excluding batch size, so (Classes)
            )
            feature_dim = base_channels * 4
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Store feature_dim as instance attribute
        self.feature_dim = feature_dim
        
        hidden_dim = feature_dim // 2
        
        if self.num_meta_info > 0:
            self.additional_input = nn.Sequential(
                nn.Linear(self.num_meta_info, base_channels * 4 ),
                nn.Linear(base_channels * 4, feature_dim)
            )
            self.concat_features = nn.Sequential(
                nn.Linear(2*self.feature_dim, self.feature_dim)
            )
                
        self.num_avgpool_head =  args.num_avgpool_head
                            
        if self.num_avgpool_head > 0:
            #self.dynamic_pool = LearnableAvgPool(pool_sizes=list(range(0,self.num_avgpool_head+1)))
            self.dynamic_pool = AdaptivePoolOrGaussian(
                args.in_channels,
                pool_sizes=list(range(0,self.num_avgpool_head+1))
            )
            
            
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Linear(hidden_dim, 2)
        )
        
        self.domain_discriminator = None
        self.grl = None
        self.multilinear_map = None

        if args and args.adaptation in ["dann", "cdan"]:
            self._init_domain_adaptation_components(args)
            
    def _init_domain_adaptation_components(self, args):
        if args.adaptation == "dann":
            self.domain_discriminator = DomainDiscriminator(
                in_feature=self.feature_dim, 
                hidden_size=getattr(args, 'domain_discriminator_hidden', self.feature_dim)
            )
            self.grl = GradientReverseLayer(alpha=1.0)
            
        elif args.adaptation == "cdan":
            self.multilinear_map = MultiLinearMap()
            discriminator_input_dim = self.feature_dim * 2
                
            self.domain_discriminator = DomainDiscriminator(
                in_feature=discriminator_input_dim,
                hidden_size=getattr(args, 'domain_discriminator_hidden', self.feature_dim//2)
            )
            self.grl = GradientReverseLayer(alpha=1.0)
    
    def update_grl_alpha(self, alpha):
        if self.grl is not None:
            self.grl.set_alpha(alpha)
    
    def forward(self, x):

        x_image = torch.squeeze(x[0])
        
        
        if self.num_avgpool_head > 0:
            
            x_image = self.dynamic_pool(x_image)
            
        features = self.backbone(x_image)        
            
        if self.num_meta_info > 0:
            meta_features = self.additional_input(x[1])
            features = torch.cat((features, meta_features), dim=1)
            
            features = self.concat_features(features)
   
        features = features.view(features.size(0), -1)

        outputs = {
            "features": features,
            "classification": self.classification_head(features),
        }
        return outputs

class LearnableAvgPool(nn.Module):
    def __init__(self, pool_sizes=[0, 1, 2]):
        super(LearnableAvgPool, self).__init__()
        
        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2*k+1, stride=1, padding=k) for k in pool_sizes
        ])
        self.alpha = nn.Parameter(torch.randn(len(pool_sizes)))  # Learnable weights

    def forward(self, x):
        pooled_outputs = [pool(x) for pool in self.pool_layers]  # Apply all pools
        weights = F.softmax(self.alpha, dim=0)  # Normalize weights
        out = sum(w * o for w, o in zip(weights, pooled_outputs))  # Weighted sum
        return out

class LearnableGaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=5, init_sigma=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = 2*kernel_size+1
        self.sigma = nn.Parameter(torch.tensor(float(init_sigma)))  # learnable σ

    def forward(self, x):
        # Ensure sigma > 0
        sigma = torch.abs(self.sigma) + 1e-6

        # Create 1D Gaussian
        ax = torch.arange(self.kernel_size, device=x.device) - (self.kernel_size - 1) / 2.
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss = gauss / gauss.sum()
        # Make 2D Gaussian kernel
        kernel2d = torch.outer(gauss, gauss)
        kernel2d = kernel2d / kernel2d.sum()

        # Depthwise filter (same kernel per channel)
        kernel = kernel2d.expand(self.channels, 1, self.kernel_size, self.kernel_size)

        return F.conv2d(x, kernel, padding=(self.kernel_size-1) // 2, groups=self.channels)
    
class AdaptivePoolOrGaussian(nn.Module):
    def __init__(self, in_channels, 
                 pool_sizes=[0, 1, 2, 3, 5], 
                 gaussians=[(5, 1.), (5, 2.0), (5, 3.0)]):
        super().__init__()

        # AvgPool options
        self.pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2*k+1, stride=1, padding=k) for k in pool_sizes
        ])

        # Gaussian options (each has its own learnable σ)
        self.gaussians = nn.ModuleList([
            LearnableGaussianSmoothing(in_channels, k, sigma) for (k, sigma) in gaussians
        ])

        # One weight per branch (pools + gaussians)
        self.alpha = nn.Parameter(torch.zeros(len(pool_sizes) + len(gaussians)))

    def forward(self, x):
        outputs = []

        # Pooling outputs
        for pool in self.pools:
            outputs.append(pool(x))

        # Gaussian outputs
        for g in self.gaussians:
            outputs.append(g(x))

        # Softmax over branch weights
        weights = torch.softmax(self.alpha, dim=0)

        out = sum(w * o for w, o in zip(weights, outputs))
        return out
    
    
def create_model(args):
    model = Model(
        model_name=args.model,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        args=args
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        if "model_state_dict" in checkpoint:
            new_state_dict = {}
            for k, v in checkpoint["model_state_dict"].items():
                # Remove '_orig_mod.' prefix if it exists
                new_key = k.replace('_orig_mod.', '')
                new_state_dict[new_key] = v
    
            model.load_state_dict(new_state_dict, strict=True)
            model.args = checkpoint['args']
        else:
            model.load_state_dict(checkpoint, strict=True)
            model.args = checkpoint['args']
        if args.verbose:
            print(f"Loaded model from {args.checkpoint}")

    return model