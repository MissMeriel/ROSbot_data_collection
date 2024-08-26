import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import onnx
# import onnx_tensorrt.backend as backend

# ADAPTED FROM https://github.com/peasant98/Dronet-Pytorch/

class DronetTorch(nn.Module):
    def __init__(self, img_dims, img_channels, output_dim):
        """
        Define model architecture.
        
        ## Arguments

        `img_dim`: image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.

        """
        super(DronetTorch, self).__init__()

        # get the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.img_dims = img_dims
        self.channels = img_channels
        self.output_dim = output_dim
        self.conv_modules = nn.ModuleList()
        self.beta = torch.Tensor([0]).float().to(self.device)

        # Initialize number of samples for hard-mining

        self.conv_modules.append(nn.Conv2d(self.channels, 32, (5,5), stride=(2,2), padding=(2,2)))
        filter_amt = np.array([32,64,128])
        for f in filter_amt:
            x1 = int(f/2) if f!=32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3,3), stride=(2,2), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3,3), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1,1), stride=(2,2)))
        # create convolutional modules
        self.maxpool1 = nn.MaxPool2d((3,3), (2,2))

        bn_amt = np.array([32,32,32,64,64,128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())
        self.dropout1 = nn.Dropout()

        self.linear1 = nn.Linear(6272, output_dim)
        self.linear2 = nn.Linear(6272, output_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()
        self.decay = 0.1

        

    def init_weights(self):
        '''
        intializes weights according to He initialization.

        ## parameters

        None
        '''
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        '''
        forward pass of dronet
        
        ## parameters

        `x`: `Tensor`: The provided input tensor`
        '''
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)
        
        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx+1](x2)
            x2 = self.relu_modules[relu_idx+1](x2)
            x2 = self.conv_modules[conv_idx+1](x2)
            x1 = self.conv_modules[conv_idx+2](x1)
            x3 = torch.add(x1,x2)
            x1 = x3
            bn_idx+=2
            relu_idx+=2
            conv_idx+=3

        x4 = torch.flatten(x3).reshape(-1, 6272)
        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        steer = self.linear1(x5)

        collision = self.linear2(x5)
        collision = self.sigmoid1(collision)

        return steer, collision

    def loss(self, k, steer_true, steer_pred, coll_true, coll_pred):
        '''
        loss function for dronet. Is a weighted sum of hard mined mean square
        error and hard mined binary cross entropy.

        ## parameters

        `k`: `int`: the value for hard mining; the `k` highest losses will be learned first,
        and the others ignored.

        `steer_true`: `Tensor`: the torch tensor for the true steering angles. Is of shape
        `(N,1)`, where `N` is the amount of samples in the batch.

        `steer_pred`: `Tensor`: the torch tensor for the predicted steering angles. Also is of shape
        `(N,1)`.

        `coll_true`: `Tensor`: the torch tensor for the true probabilities of collision. Is of 
        shape `(N,1)`

        `coll_pred`: `Tensor`: the torch tensor for the predicted probabilities of collision.
        Is of shape `(N,1)`
        '''
        # for steering angle
        mse_loss = self.hard_mining_mse(k, steer_true, steer_pred)
        # for collision probability
        bce_loss = self.beta * (self.hard_mining_entropy(k, coll_true, coll_pred))
        return mse_loss + bce_loss

    def hard_mining_mse(self, k, y_true, y_pred):
        '''
        Compute Mean Square Error for steering 
        evaluation and hard-mining for the current batch.

        ### parameters
        
        `k`: `int`: number of samples for hard-mining

        `y_true`: `Tensor`: torch Tensor of the expected steering angles.
        
        `y_pred`: `Tensor`: torch Tensor of the predicted steering angles.
        '''
        loss_steer = (y_true - y_pred)**2

        # hard mining        
        # get value of k that is minimum of batch size or the selected value of k
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_steer, k=k_min, dim=0)
        max_loss_steer = torch.gather(loss_steer, dim=0, index=indices)
        # mean square error
        hard_loss_steer = torch.div(torch.sum(max_loss_steer), k_min)
        return hard_loss_steer
        

    def hard_mining_entropy(self, k, y_true, y_pred):
        '''
        computes binary cross entropy for probability collisions and hard-mining.

        ## parameters

        `k`: `int`: number of samples for hard-mining

        `y_true`: `Tensor`: torch Tensor of the expected probabilities of collision.

        `y_pred`: `Tensor`: torch Tensor of the predicted probabilities of collision.
        '''

        loss_coll = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_coll, k=k_min, dim=0)
        max_loss_coll = torch.gather(loss_coll, dim=0, index=indices)
        hard_loss_coll = torch.div(torch.sum(max_loss_coll), k_min)
        return hard_loss_coll
    

# class DronetOnnx():
#     def __init__(self, img_dims, img_channels, output_dim, verbose=False):
#         super(DronetOnnx).__init__()
#         self.dronet = DronetTorch(img_dims, img_channels, output_dim)
#         self.dronet.to(self.dronet.device)
#         inference_shape = (1, img_channels, img_dims[0], img_dims[1])
#         inputs = torch.randn(inference_shape).to(self.dronet.device)
#         torch.onnx.export(self.dronet, inputs, 'dronet.onnx', verbose=verbose,
#                             output_names=['steer', 'coll'])
    
#     def load_model(self, path):
#         self.model = onnx.load(path)
#         if not torch.cuda.is_available():
#             raise NotImplementedError('TensorRT backend does not work for non-CUDA devices.')
#         # get first gpu
#         self.engine = backend.prepare(self.model, device='CUDA:0')

#     def infer(self, data, verbose=False):
#         input_data = data.numpy().astype(np.float32)
#         output_data = self.engine.run(input_data)
#         steer = output_data['steer']
#         coll = output_data['coll']
#         if verbose:
#             print(f'Steering Angle: {steer[0]} radians')
#             print(f'Collision Prob: {coll[0]}')
#         return steer, coll

# # one dim for steering angle, another for prob. of collision
# # dronet = DronetTorch(img_dims=(224,224), img_channels=3, output_dim=1)
# # dronet.cuda()
# # steer_true = torch.Tensor([[0.1], [0.1], [0.1], [0.1]]).cuda()
# # coll_true = torch.Tensor([[0.1], [0.1], [0.1], [0.1]]).cuda()
# # m = torch.ones((4,3, 224, 224)).cuda()
# # steer_pred, coll_pred = dronet(m)