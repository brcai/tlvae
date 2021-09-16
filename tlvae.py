import numpy as np
import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms, utils
from funcs import *
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

parser = argparse.ArgumentParser(description='[model] Time Series Forecasting')
parser.add_argument('--data', type=str, required=False, default='dd', help='data')
parser.add_argument('--window', type=int, required=False, default=10, help='window')
parser.add_argument('--slid', type=int, required=False, default=5, help='slid')
parser.add_argument('--zdim', type=int, required=False, default=10, help='zdim')
parser.add_argument('--znum', type=int, required=False, default=2, help='znum')
parser.add_argument('--ydim', type=int, required=False, default=1, help='ydim')
parser.add_argument('--hdim', type=int, required=False, default=10, help='hdim')
parser.add_argument('--grudim', type=int, required=False, default=10, help='grudim')
parser.add_argument('--epoch', type=int, required=False, default=100, help='epoch')
parser.add_argument('--learn_rate', type=float, required=False, default=0.001, help='learn_rate')
parser.add_argument('--weight_decay', type=int, required=False, default=-1, help='weight_decay')
parser.add_argument('--warm_up', type=int, required=False, default=15, help='warm_up')
parser.add_argument('--batch', type=int, required=False, default=32, help='batch')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = [args.batch, args.batch, args.batch]
WINDOW = args.window
SLID = args.slid
#HIDDEN_LAYERS = [512, 256, 128, 64, 32]
Z_DIM = args.zdim
Z_NUM = args.znum
Y_DIM = args.ydim
H_DIM = args.zdim
GRU_H_DIM = args.zdim
FILE = args.data

N_EPOCHS = args.epoch
LEARNING_RATE = args.learn_rate #1e-3 #(PAPER ORIGINAL)
WEIGHT_DECAY = args.weight_decay
N_WARM_UP = args.warm_up

#N_SAMPLE = 64

#N_LAYERS = len(HIDDEN_LAYERS)

SAVE_MODEL_EPOCH = N_EPOCHS - 10

PATH = 'saved_models/'

para = str(SLID)+"_"+str(Z_DIM)+"_"+str(Z_NUM)+"_"+str(H_DIM)+"_"+str(GRU_H_DIM)+".txt"
#fp = open("svae_res/"+FILE+"/"+para,'w')


beta = DeterministicWarmup(n_steps=N_WARM_UP, t_max=1) # Linear warm-up from 0 to 1 over 50 epochs


train_loader, valid_loader, test_loader = ReadData(FILE, BATCH_SIZE, WINDOW)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi ) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param var: variance of distribution
    :return: log N(x|mu,var)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - torch.log(var + 1e-8) / 2 - (x - mu)**2 / (2 * var + 1e-8)
    # print('Size log_pdf:', log_pdf.shape)
    return torch.sum(log_pdf, dim=-1)


def merge_gaussian(mu1, var1, mu2, var2):
    # we have to compute the precision 1/variance
    precision1 = 1 / (var1 + 1e-8)
    precision2 = 1 / (var2 + 1e-8)

    # now we have to compute the new mu = (mu1*prec1 + mu2*prec2)/(prec1+prec2)
    new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

    # and the new variance var = 1/(prec1 + prec2)
    new_var = 1 / (precision1 + precision2)

    # we have to transform the new var into log_var
    # new_log_var = torch.log(new_var + 1e-8)
    return new_mu, new_var


def reparametrization_trick(mu, var):
    '''
    Function that given the mean (mu) and the logarithmic variance (log_var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample
    :param mu: mean of the z_variables
    :param var: variance of the latent variables (as in the paper)
    :return: z = mu + sigma * noise
    '''
    # compute the standard deviation from the variance
    std = torch.sqrt(var)

    # we have to sample the noise (we do not have to keep the gradient wrt the noise)
    eps = Variable(torch.randn_like(std), requires_grad=False)
    z = mu.addcmul(std, eps)

    return z


class EncoderMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        This is a single block that takes x or d as input and return
        the last hidden layer and mu and _var
        (Substantially this is a small MLP as the encoder we in the original VAE)
        :param input_dim:
        :param hidden_dims:
        :param latent_dim:
        '''

        super(EncoderMLPBlock, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

        self.mu = nn.Linear(hidden_dim, z_dim)
        self._var = nn.Linear(hidden_dim, z_dim)


    def forward(self, d):
        d = self.hidden_layer(d)
        d = F.leaky_relu(self.batchnorm(d))

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))


        return d, _mu, _var


class DecoderMLPBlock(nn.Module):
    def __init__(self, z1_dim, hidden_dim, z2_dim):
        '''
        This is also substantially a MLP, it takes the z obtained from the
        reparametrization trick and it computes the mu and var of the _z of the layer
        below, which, during the inference, has to be merged with the mu and _var obtained
        by at the EncoderMLPBlock at the same level.
        :param z1_dim:
        :param hidden_dims:
        :param z2_dim:
        '''
        super(DecoderMLPBlock, self).__init__()

        self.hidden_layer =nn.Linear(z1_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        ## we have two output: mu and # sigma^2
        self.mu = nn.Linear(hidden_dim, z2_dim)
        self._var = nn.Linear(hidden_dim, z2_dim)

    def forward(self, d):


        d = self.hidden_layer(d)
        d = F.leaky_relu(self.batchnorm(d), 0.1)

        _mu = self.mu(d)
        _var = F.softplus(self._var(d))

        return _mu, _var




class FinalDecoder(nn.Module):
    def __init__(self, z_final, hidden_dim, input_dim):
        '''
        This is the final decoder, the one that is used only in the generation process. It takes the z_L
        and then it learn to reconstruct the original x.
        :param z_final:
        :param hidden_dims:
        :param input_dim:
        '''
        super(FinalDecoder, self).__init__()
        ## now we have to create the architecture
        # neurons = [z_final, *hidden_dims]
        ## common part of the architecture
        self.hidden_layer = nn.Linear(z_final, hidden_dim)
        # test_set_reconstruction layer
        self.reconstruction = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):

        z = F.relu(self.hidden_layer(z))
        # print(self.test_set_reconstruction(x).shape)
        return self.output_activation(self.reconstruction(z))


class TLVAE(torch.nn.Module):
    def __init__(self, d_x, x_window, d_z, z_num, d_y=1, d_h=5, d_gru_h=10, device=None):
        super(TLVAE, self).__init__()

        self.t = d_x - x_window
        self.z_num = z_num
        self.d_z = d_z
        self.x_window = x_window
        self.d_x = d_x
        self.d_y = d_y
        self.d_gru_h = d_gru_h
        self.z = torch.zeros(d_x - x_window, d_z, z_num)
        self.eps = 1e-10
        self.device = device

        self.encoder_xz = EncoderMLPBlock(x_window+d_gru_h+d_z, d_h, d_z)
        self.encoder_hz = EncoderMLPBlock(d_h, d_h, d_z)
        self.encoder_zz = EncoderMLPBlock(d_z, d_h, d_z)

        self.prior_zz = EncoderMLPBlock(d_z, d_h, d_z)

        self.decoder_zz = DecoderMLPBlock(d_z+x_window, d_h, d_z)
        self.decoder_zxz = DecoderMLPBlock(d_z+d_gru_h, d_h, d_z)
        self.decoder_zx = FinalDecoder(d_z+d_gru_h, d_h, x_window)

        self.gru = nn.GRUCell(x_window, d_gru_h)

        self.predict = torch.nn.Sequential(
                torch.nn.Linear(d_z+d_gru_h, d_y),
            )
        self.output_activation = nn.Sigmoid()

        self.decoder_loss = torch.nn.MSELoss()
        
        self.predict_loss = torch.nn.MSELoss()

        self.z_last = 0

    def reparam(self, mean, delta):
        eps = torch.randn_like(mean)
        std = torch.exp(delta)
        rep = eps.mul(std) + mean
        return rep

    def rec_loss_cal(self, y, y_predict):
        rec_loss = self.decoder_loss(y, y_predict)
        return rec_loss

    def _approximate_kl(self, z, q_params, p_params = None):
        ## we have to compute the pdf of z wrt q_phi(z|x)
        (mu, var) = q_params
        qz = log_gaussian(z, mu, var)
        # print('size qz:', qz.shape)
        ## we should do the same with p
        if p_params is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_params
            pz = log_gaussian(z, mu, log_var)
            # print('size pz:', pz.shape)

        kl = qz - pz

        return kl

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return torch.sum(kld_element,dim=1)
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        kl_loss = 0
        rec_loss = 0
        predict_loss = 0
        #self.z = torch.zeros(x.size()[0], self.t, self.z_num, self.d_z)
        z = self.reparam(torch.zeros(x.size()[0], self.d_z), torch.ones(x.size()[0], self.d_z)).to(self.device)
        gru_h = torch.zeros(x.size()[0], self.d_gru_h)
        gru_h = gru_h.to(self.device)

        #for i in range(1):
        for i in range(self.t):
            x_tmp = self.x[:,i:i+self.x_window]
            mean_encs = []
            delta_encs = []
            d = x_tmp

            #encoding
            d, mean_enc, delta_enc = self.encoder_xz(torch.cat([x_tmp,z,gru_h],1))
            mean_encs.append(mean_enc)
            delta_encs.append(delta_enc)

            mean_prior, delta_prior = self.decoder_zxz(torch.cat([gru_h, z], 1))

            z = self.reparam(mean_enc, delta_enc)

            kl_tmp = self._kld_gauss(mean_encs[0], delta_encs[0], mean_prior, delta_prior)

            kl_loss += kl_tmp

            
            for j in range(1, self.z_num):
                d, mean_enc, delta_enc = self.encoder_zz(z)

                mean_prior, delta_prior = self.decoder_zz(torch.cat([z, x_tmp], 1))

                #merged_mu, merged_var = merge_gaussian(mean_encs[j], delta_encs[j], mean_prior, delta_prior)

                z = reparametrization_trick(mean_enc, delta_enc)

                kl_tmp = self._kld_gauss(mean_prior, delta_prior, mean_enc, delta_enc)
                kl_loss += kl_tmp
            

            x_reconstructed = self.decoder_zx(torch.cat([z, gru_h], 1))
            rec_loss += self.decoder_loss(x_tmp, x_reconstructed)
        
            gru_h = self.gru(x_tmp, gru_h)

        x_pred = self.predict(torch.cat([z,gru_h], 1))
        x_pred = self.output_activation(x_pred)
        pred_loss = self.predict_loss(x_pred, y)

        return kl_loss, rec_loss, pred_loss, x_pred


model = TLVAE(WINDOW, SLID, Z_DIM, Z_NUM, Y_DIM, H_DIM, GRU_H_DIM, device)
model.to(device)
print('Model overview and recap\n')
#print(model)
print('\n')
print(FILE+"   "+"mode")

## optimization
if WEIGHT_DECAY > 0:
    # we add small L2 reg as in the original paper
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

## training loop
training_loss = []
approx_kl = []

maxmse = 1000000000
maxmae = 1000000000
maxmape = 1000000000

mse_patience = 0.0
patience = 0

## we have also to retrieve and store the mean of the kl for each layer
print('.....Starting trianing')
_beta = 0
for epoch in range(N_EPOCHS):
    n_batch = 0
    total_loss = torch.zeros(1).type(torch.FloatTensor)
    total_loss = total_loss.to(device)
    for i, data in enumerate(train_loader, 0):
        n_batch += 1
        ts, labels = data
        ts = ts.to(device)
        labels = labels.to(device)

        kl_loss, rec_loss, predict_loss, _ = model(ts, labels)

        loss = rec_loss + _beta * torch.sum(kl_loss) + predict_loss

        L = loss / len(ts)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss+=loss
    print("epoach: ", epoch, "     loss: ",total_loss[0])
    ## update the beta
    _beta = next(beta)

    with torch.no_grad():

        y = torch.zeros(0,1).to(device)
        preds = torch.zeros(0,1).to(device)
        for i, data in enumerate(valid_loader, 0):
            ts, labels = data
            ts = ts.to(device)
            labels = labels.to(device)

            _, _, _, pred = model(ts, labels)
            y = torch.cat((y,labels),0)
            preds = torch.cat((preds,pred),0)

        preds = preds.cpu()
        preds = preds.detach().numpy()
        y = y.cpu()
        y = y.detach().numpy()

        mse = mean_squared_error(preds, y)
        mae = mean_absolute_error(preds, y)
        mape = mean_absolute_percentage_error(preds, y)

        if mse < mse_patience:
            patience += 1

        if patience > 3 or epoch == N_EPOCHS - 1:
            y = torch.zeros(0,1).to(device)
            preds = torch.zeros(0,1).to(device)
            for i, data in enumerate(test_loader, 0):
                ts, labels = data
                ts = ts.to(device)
                labels = labels.to(device)

                _, _, _, pred = model(ts, labels)
                y = torch.cat((y,labels),0)
                preds = torch.cat((preds,pred),0)

            preds = preds.cpu()
            preds = preds.detach().numpy()
            y = y.cpu()
            y = y.detach().numpy()

            mse = mean_squared_error(preds, y)
            mae = mean_absolute_error(preds, y)
            mape = mean_absolute_percentage_error(preds, y)
            if mse<maxmse:
                maxmse=mse
                maxmae=mae
                maxmape=mape
            break

print(FILE)
print('{0:.5f}'.format(maxmse)+"; "+'{0:.5f}'.format(maxmae)+"; "+'{0:.5f}'.format(maxmape)+"\n")
print("end")
