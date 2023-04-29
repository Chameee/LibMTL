import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

INFLUENCER_BRAND_ENCODER = 24
INFLUENCER_CAMPAIGN_ENCODER = 12

from create_dataset import InfluencerBrandDataset

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

def parse_args(parser):
    parser.add_argument('--dataset', default='ins_data', type=str, help='ins_data')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    if params.dataset == 'ins_data':
        task_name = ['MTL']
        class_num = 3
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
    
    # define tasks
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]} for task in task_name}
    
    # prepare dataloaders
    data_loader, _ = InfluencerBrandDataset(dataset=params.dataset, batchsize=params.bs, root_path=params.dataset_path)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}
    
    # define encoder and decoders
    class InfluencerBrandEncoder(nn.Module):
        def __init__(self):
            super(InfluencerBrandEncoder, self).__init__()
            hidden_dim = 512
            self.hidden_layer_list = [nn.Linear(INFLUENCER_BRAND_ENCODER, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = inputs
            out = torch.flatten(out)
            out = self.hidden_layer(out)
            return out

    class InfluencerCampaignEncoder(nn.Module):
        def __init__(self):
            super(InfluencerCampaignEncoder, self).__init__()
            hidden_dim = 512
            self.hidden_layer_list = [nn.Linear(INFLUENCER_CAMPAIGN_ENCODER, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)

        def forward(self, inputs):
            out = inputs
            out = torch.flatten(out)
            out = self.hidden_layer(out)
            return out

    class MTLEncoder(nn.Module):
        def __init__(self):
            super(MTLEncoder, self).__init__()
            self.influencer_brand_encoder = InfluencerBrandEncoder()
            self.influencer_campaign_encoder = InfluencerCampaignEncoder()



    influencer_brand_decoders = nn.ModuleDict({task: nn.Linear(512, 1) for task in list(task_dict.keys())})
    influencer_campaign_decoders = nn.ModuleDict({task: nn.Linear(512, 3) for task in list(task_dict.keys())})

    Model = Trainer(task_dict=task_dict,
                          weighting=weighting_method.__dict__[params.weighting], 
                          architecture=architecture_method.__dict__[params.arch], 
                          encoder_class=InfluencerBrandEncoder,
                          decoders=influencer_brand_decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)
    Model.train(train_dataloaders=train_dataloaders,
                      val_dataloaders=val_dataloaders,
                      test_dataloaders=test_dataloaders, 
                      epochs=100)
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
