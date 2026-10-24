#!/usr/bin/env python

'''
    Losses for fitting the PSF models
'''


class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return abs(x).mean()

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return (x.pow(2)).mean()