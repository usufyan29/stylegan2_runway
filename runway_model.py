import runway
from runway.data_types import number, image
import random
from generate_for_runway import *
input= {
    "z": number(random.randint(1,1000))
}

setup_options = {
    'truncation': number(min=5, max=100, step=1, default=5)
}
@runway.setup(options=setup_options)
def setup(opts):
    return opts


@runway.command(name='generate',inputs=input,outputs={ 'image': image })
def generate_(model , args):
    # _run_cmd('export CUDA_HOME=/usr/local/cuda')
    # _run_cmd('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64')
    # _run_cmd('export PATH=$PATH:$CUDA_HOME/bin')
    _run_cmd('nvidia-smi')
    network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl'
    seed=args['z']
    trunc=0.5
    output_image =  generate_images(network_pkl, seed, trunc)
    return {'image': output_image}


def _run_cmd(cmd):
    with os.popen(cmd) as pipe:
        output = pipe.read()
        status = pipe.close()
    if status is not None:
        raise RuntimeError('See below for full command line and output log:\n\n%s\n\n%s' % (cmd, output))


if __name__ == '__main__':
    runway.run()