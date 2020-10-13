import runway
from runway.data_types import number, image
import random
from generate_for_runway import *
input= {
    "z": number(random.randint(1,1000))
}

setup_options = {
    'truncation': number(min=5, max=100, step=1, default=0.5)
}
@runway.setup(options=setup_options)
def setup(opts):
    return opts


@runway.command(name='generate_image',inputs=input,outputs={ 'image': image })
def generate_image(model , args):
    network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl'
    seed=args['z']
    trunc=0.5
    output_image =  generate_images(network_pkl, seed, trunc)
    return {'image': output_image}



if __name__ == '__main__':
    runway.run(port=8080)