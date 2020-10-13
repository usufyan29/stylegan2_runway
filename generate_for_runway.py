import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------

def generate_images(network_pkl, seed, truncation_psi):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])

    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    return PIL.Image.fromarray(images[0], 'RGB')

#----------------------------------------------------------------------------