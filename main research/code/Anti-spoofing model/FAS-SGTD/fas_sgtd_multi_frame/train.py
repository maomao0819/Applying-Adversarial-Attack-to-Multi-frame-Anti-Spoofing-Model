import numpy as np
import tensorflow as tf
import FLAGS_tf1
# from generate_data_train import input_fn_maker as input_fn_maker_train
from generate_data_train_without_blank_depth import input_fn_maker as input_fn_maker_train 
from generate_data_test import input_fn_maker as input_fn_maker_test
from generate_network_tf1 import generate_network as model_fn

flags=FLAGS_tf1.flags # setting paras
# log info setting
# tf.logging.set_verbosity(tf.logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
# data fn

train_data_list=[flags.path.train_file]

# train_input_fn = lambda: input_fn_maker_train(train_data_list, shuffle = True, batch_size = flags.paras.batch_size_train, epoch = flags.paras.epoch)
# train_input_fn = input_fn_maker_train(train_data_list, shuffle = True, batch_size = flags.paras.batch_size_train, epoch = flags.paras.epoch)
def train_input_fn():
    return input_fn_maker_train(train_data_list, shuffle = True, batch_size = flags.paras.batch_size_train, epoch = flags.paras.epoch)

test_data_list = [flags.path.test_file]
if flags.path.dev_file != 'NoFile':
    test_data_list.append(flags.path.dev_file)

test_input_fn = input_fn_maker_test(test_data_list, shuffle = False, batch_size = flags.paras.batch_size_test, epoch = 1)

# model fn
model_fn_this = model_fn
# GPU config 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
# # create estimator
this_config = tf.estimator.RunConfig(
    save_summary_steps = flags.display.summary_iter,
    save_checkpoints_steps = flags.display.display_iter,
    keep_checkpoint_max = 102400,
    log_step_count_steps = flags.display.log_iter,
    # session_config = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True))
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement = True, gpu_options = tf.compat.v1.GPUOptions(allow_growth = True))
)

mnist_classifier = tf.estimator.Estimator(model_fn = model_fn_this, config = this_config, model_dir = flags.path.model)

# only run train set
mnist_classifier.train(input_fn = train_input_fn)
