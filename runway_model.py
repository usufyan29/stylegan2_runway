import runway
import random
from generate_for_runway import *
inputs= {
    "z": runway.number(random.randint(1,1000))
}

@runway.command('generate',
                inputs=inputs,
               outputs={ 'image': runway.image })
def generate(model , inputs):
  network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl'
  seed=inputs['z']
  trunc=0.5
  output_image =  generate_images(network_pkl, seed, trunc)
  return {
            'image': output_image
        }



if __name__ == '__main__':
    runway.run(port=8080)