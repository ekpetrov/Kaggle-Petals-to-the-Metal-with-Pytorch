import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from time import time
from torcheval.metrics.functional import multiclass_f1_score

class VGG11(nn.Module):
    def __init__(self, conv_arch=None, resolution=(192, 192)):
        super().__init__()

        self.resolution = resolution
        self.pool_kernel_size = 2
        if conv_arch is None:
            pass
        self.layers = self.make_net(conv_arch)

        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        torch.device(self.device)
        self.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.score_fn = multiclass_f1_score

        learning_rate = 0.01
        weight_decay = 5e-4
        self.optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=learning_rate)

    def vgg_block(self, num_convs, num_in, num_out):
        layers = []
        for ic in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels=num_in, out_channels=num_out,
                    kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            num_in = num_out + 0
        layers.append(nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=2))
        return layers
    
    def make_net(self, conv_arch):
        layers = []
        resolution_after_conv = \
            np.array(self.resolution) // (self.pool_kernel_size**len(conv_arch))
        
        num_in = 3
        for ic, (num_convs, num_out) in enumerate(conv_arch):
            block = self.vgg_block(num_convs, num_in, num_out)
            layers += block
            num_in = num_out + 0

        fc_dim_in = resolution_after_conv.prod()*conv_arch[-1][-1]
        layers = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(fc_dim_in, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 104),
            nn.Softmax(dim=1))
        return nn.Sequential(*layers)
    
    def forward(self, img):
        out = img + 0.0
        for layer in self.layers:
            out = layer(out)
        return(out)

    def fit(self, dataset_train, dataset_val):
        print_period = 200
        # print_period = len(dataset_train) // 100
        num_epochs = 10
        batch_size = 16
        loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        loader_val = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

        for epoch in range(num_epochs):
            s = f'Epoch {epoch+1} started.'
            print(s)
            t_epoch_start = time()
            losses_ep = []
            targets_true = []
            targets_pred = []

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                    torch.set_grad_enabled(True)
                    loader = loader_train
                elif phase == 'val':
                    self.eval()
                    torch.set_grad_enabled(False)
                    loader = loader_val
                
                for i, (xb, yb) in enumerate(loader):
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    yb_pred = self(xb)
                    loss = self.loss_fn(yb_pred, yb)
                    losses_ep.append(loss.item())
                    targets_true.append(yb)
                    targets_pred.append( yb_pred.argmax(axis=1) )

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if (i + 1) % print_period == 0 and phase == 'train':
                        s = f'Epoch: {epoch+1}, batch {i+1}. Train mean loss: {np.mean(losses_ep): 2.5f}'
                        print(s)
            t_epoch_end = time()
            dtime = round(t_epoch_end - t_epoch_start, 1)
            y_train = [y for (_, y) in dataset_train]
            y_val = [y for (_, y) in dataset_val]
            y_train_pred = [ self(x) for (x, _) in dataset_train ]
            y_val_pred = [ self(x) for (x, _) in dataset_val ]

            score_train = multiclass_f1_score(y_train_pred, y_train)
            score_val = multiclass_f1_score(y_val_pred, y_val)
            s = f'Epoch {epoch+1} train score: {score_train}, val score: {score_val}, time: {dtime}'
            print(s)



                    
                
