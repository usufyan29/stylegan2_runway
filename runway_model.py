import runway
import random
from generate_for_runway import *
inputs= {
    "z": runway.number(random.randint(1,1000))
}

@runway.command('generate',
               outputs={ 'image': runway.image })
def generate(inputs):
  network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl'
  seed=inputs['z']
  trunc=0.5
  return generate_images(network_pkl, seed, trunc)



if __name__ == '__main__':
    runway.run(port=5232)