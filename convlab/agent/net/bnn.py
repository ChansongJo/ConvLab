"""Bayesian neural network models."""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np

import inspect

from functools import partial
from torch.nn import Parameter
from collections import Iterable, OrderedDict

from convlab.agent.net import net_util
from convlab.agent.net.base import Net
from convlab.lib import math_util, util, logger


logger = logger.get_logger(__name__)


class BDropout(torch.nn.Dropout):

    """Binary dropout with regularization and resampling.

    See: Gal Y., Ghahramani Z., "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", 2016.
    """

    def __init__(self, rate=0.25, reg=1.0, **kwargs):
        """Constructs a BDropout.

        Args:
            rate (float): Dropout probability.
            reg (float): Regularization scale.
        """
        super(BDropout, self).__init__(**kwargs)
        self.register_buffer("rate", torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer("reg", torch.tensor(reg))
        self.register_buffer("noise", torch.bernoulli(self.p))

    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p = 1 - self.rate
        weight_reg = self.p * (weight**2).sum()
        bias_reg = (bias**2).sum() if bias is not None else 0
        return self.reg * (weight_reg + bias_reg)

    def resample(self):
        """Resamples the dropout noise."""
        self._update_noise(self.noise)

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=False, mask_dims=0, **kwargs):
        """Computes the binary dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        if sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
        elif resample:
            return x * torch.bernoulli(self.p.expand(x.shape))

        return x * self.noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "rate={}".format(self.rate)


class CDropout(BDropout):

    """Concrete dropout with regularization and resampling.

    See: Gal Y., Hron, J., Kendall, A. "Concrete Dropout", 2017.
    """

    def __init__(self, temperature=0.1, rate=0.5, reg=1.0, **kwargs):
        """Constructs a CDropout.

        Args:
            temperature (float): Temperature.
            rate (float): Initial dropout rate.
            reg (float): Regularization scale.
        """
        super(CDropout, self).__init__(rate, reg, **kwargs)
        self.temperature = Parameter(
            torch.tensor(temperature), requires_grad=False)

        # We need to constrain p to [0, 1], so we train logit(p).
        self.logit_p = Parameter(-torch.log(self.p.reciprocal() - 1.0))

    
    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p.data = self.logit_p.sigmoid()
        reg = super(CDropout, self).regularization(weight, bias)
        reg -= -(1 - self.p) * (1 - self.p).log() - self.p * self.p.log()
        return reg

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.noise.data = torch.rand_like(x)

    def forward(self, x, resample=False, mask_dims=0, **kwargs):
        """Computes the concrete dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        if sample_shape != noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
            noise = self.noise
        elif resample:
            noise = torch.rand_like(x)

        self.p.data = self.logit_p.sigmoid()
        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        concrete_noise = (concrete_p / self.temperature).sigmoid()

        return x * concrete_noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "temperature={}".format(self.temperature)


class BSequential(torch.nn.Sequential):

    """Extension of Sequential module with regularization and resampling."""

    def resample(self):
        """Resample all child modules."""
        for child in self.children():
            if isinstance(child, BDropout):
                child.resample()

    
    def regularization(self, device):
        """Computes the total regularization cost of all child modules.

        Returns:
            Total regularization cost (Tensor).
        """
        reg = torch.tensor(0.0).to(device)
        children = list(self._modules.values())
        for i, child in enumerate(children):
            if isinstance(child, BSequential):
                reg += child.regularization()
            elif isinstance(child, BDropout):
                for next_child in children[i:]:
                    if hasattr(next_child, "weight") and hasattr(
                            next_child, "bias"):
                        reg += child.regularization(next_child.weight,
                                                    next_child.bias)
                        break
        return reg

    def forward(self, x, resample=False, **kwargs):
        """Computes the model.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.

        Returns:
            Output (Tensor).
        """
        for module in self._modules.values():
            if isinstance(module, (BDropout, BSequential)):
                x = module(x, resample=resample, **kwargs)
            else:
                x = module(x)
        return x


def bayesian_model(in_features,
                   out_features,
                   hidden_features,
                   nonlin=torch.nn.ReLU,
                   output_nonlin=None,
                   weight_initializer=partial(
                       torch.nn.init.xavier_normal_,
                       gain=torch.nn.init.calculate_gain("relu")),
                   bias_initializer=partial(
                       torch.nn.init.uniform_, a=-1.0, b=1.0),
                   dropout_layers=CDropout,
                   input_dropout=None):
    """Constructs and initializes a Bayesian neural network with dropout.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        hidden_features (list<int>): Ordered list of hidden dimensions.
        nonlin (Module): Activation function for all hidden layers.
        output_nonlin (Module): Activation function for output layer.
        weight_initializer (callable): Function to initialize all module
            weights to pass to module.apply().
        bias_initializer (callable): Function to initialize all module
            biases to pass to module.apply().
        dropout_layers (Dropout or list<Dropout>): Dropout type to apply to
            hidden layers.
        input_dropout (Dropout): Dropout to apply to input layer.

    Returns:
        Bayesian neural network (BSequential).
    """
    dims = [in_features] + hidden_features
    if not isinstance(dropout_layers, Iterable):
        dropout_layers = [dropout_layers] * len(hidden_features)

    modules = OrderedDict()

    # Input layer.
    if inspect.isclass(input_dropout):
        input_dropout = input_dropout()
    if input_dropout is not None:
        modules["drop_in"] = input_dropout

    # Hidden layers.
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        drop_i = dropout_layers[i]
        if inspect.isclass(drop_i):
            drop_i = drop_i()

        modules["fc_{}".format(i)] = torch.nn.Linear(din, dout)
        if drop_i is not None:
            if len(dims[:-1]) > 1:
                modules["drop_{}".format(i)] = drop_i
            else:
                modules["drop"] = drop_i

    modules[f"nonlin"] = nonlin

    # Output layer.
    modules["fc_out"] = torch.nn.Linear(dims[-1], out_features)
    if output_nonlin is not None:
        modules["nonlin_out"] = output_nonlin

    def init(module):
        if callable(weight_initializer) and hasattr(module, "weight"):
            weight_initializer(module.weight)
        if callable(bias_initializer) and hasattr(module, "bias"):
            if module.bias is not None:
                bias_initializer(module.bias)

    # Initialize weights and biases.
    net = BSequential(modules)
    net.apply(init)

    return net


def gaussian_log_likelihood(targets, pred_means, pred_stds=None, dbg=None):
    deltas = pred_means - targets
    if pred_stds is not None:

        lml = -((deltas / pred_stds)**2).sum(-1)*0.5 - pred_stds.log().sum(-1)*0.5 - 0.5*torch.log(2 * torch.tensor(math.pi))
        if torch.isinf(lml):
            logger.info(f'deltas/pred_stds squared sum : {((deltas / pred_stds)**2).mean(-1) * 0.5}')
            logger.info(f'deltas/preds : {deltas / pred_stds}')
            logger.info(f'deltas : {deltas}')
            logger.info(f'pred_stds : {pred_stds}')
            logger.info(f'pred_stds raw : {dbg}')
            exit()
    else:
        lml = -(deltas**2).sum(-1)
    
    return lml

class BNNbyDropout(Net, nn.Module):
    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=True,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation', # relu
            'out_layer_activation', # log_softmax
            'init_fn',
            'dropout_layer',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
            'tau'
        ])

        self.dropout_layer = BDropout if self.dropout_layer == 'BDropout' else CDropout
        nonlin = net_util.get_activation_fn(self.hid_layers_activation)
        output_nonlin = net_util.get_activation_fn(self.out_layer_activation) if self.out_layer_activation is not None else None
        weight_initializer = torch.nn.init.kaiming_normal_
        #weight_initializer = torch.nn.init.zeros_
        bias_initializer = partial(torch.nn.init.uniform_, a=-1.0, b=1.0)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.model = bayesian_model(self.in_dim, self.out_dim *2, self.hid_layers,
                                    nonlin, output_nonlin, weight_initializer, bias_initializer, self.dropout_layer)


        self.to(self.device)
    
    def forward(self, x, resample=False):
        output = self.model(x, resample)
        self.mean, self.log_std = output.split([self.out_dim, self.out_dim], dim=-1)
        self.log_std = torch.tanh(self.log_std)
        #out = self.sample_gaussian(self.mean, self.log_std)
        return self.mean
    
    def regularization(self):
        return self.model.regularization(self.device)
    
    def reset_noise(self):
        '''for updating and reloading nets'''
        children = list(self.model._modules.values())
        for child in children:
            if isinstance(child, self.dropout_layer):
                child._update_noise(torch.zeros((1, 1)))

    def sample_gaussian(self, mu, logvar, std=1.5):
        assert mu.size() == logvar.size()
        logvar = torch.tanh(logvar)
        _size = logvar.size()
        epsilon = Variable(torch.normal(mean=torch.zeros(*_size), std=std))
        std = torch.exp(0.5 * logvar)
        epsilon = epsilon.to(self.device)
        return mu + std*epsilon     



    

        
    


