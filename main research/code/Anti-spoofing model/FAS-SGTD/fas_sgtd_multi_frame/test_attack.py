from easydict import EasyDict as edict
import numpy as np

#import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
from generate_network import generate_network as model_fn

test_config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1}, allow_soft_placement=True, log_device_placement=True)
test_config.gpu_options.allow_growth=True 
print('a')