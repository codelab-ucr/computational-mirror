#!/usr/bin/env python

import importlib

import utils
import siren
import losses
import wire
import gaussian

utils = importlib.reload(utils)
siren = importlib.reload(siren)
losses = importlib.reload(losses)
wire = importlib.reload(wire)
gaussian = importlib.reload(gaussian)

def get_model(config, nchunks):
    if  config.nonlin == 'sine':
        model = siren.AdaptiveMultiSiren(
                in_features=config.in_features,
                out_features=config.out_features, 
                n_channels=nchunks,
                hidden_features=config.nfeat, 
                hidden_layers=config.nlayers,
                outermost_linear=True,
                share_weights=config.share_weights,
                first_omega_0=config.omega_0,
                hidden_omega_0=config.omega_0
            ).cuda()
    elif config.nonlin == 'wire':
        hidden_omega_0 = 1.0
        
        model = wire.AdaptiveMultiWIRE(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=config.scale,
            const=1.0,
            mode='2d'
        ).cuda()

    elif config.nonlin == 'gauss':
        model = gaussian.AdaptiveMultiGauss(
            in_features=config.in_features,
            out_features=config.out_features, 
            n_channels=nchunks,
            hidden_features=config.nfeat, 
            hidden_layers=config.nlayers,
            outermost_linear=True,
            share_weights=config.share_weights,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0,
            scale=config.scale,
            const=1.0
        ).cuda()

    return model