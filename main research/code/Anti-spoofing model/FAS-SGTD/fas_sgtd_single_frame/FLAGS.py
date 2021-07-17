from easydict import EasyDict as edict
import os

flags=edict()

flags.path=edict()

# path_gen_save = './model_save_single_frame/'
path_gen_save = './model_save_single_frame_inverse/'

# flags.path.dataset = 'OULU-NPU'
# flags.path.dataset = 'CASIA-FASD'
# flags.path.dataset = 'replayattack'
flags.path.dataset = 'MSU-MFSD'
path_gen_save = os.path.join(path_gen_save, flags.path.dataset)

dataset_path = os.path.join('/', 'mnt', 'sda1', 'mike', flags.path.dataset)
if flags.path.dataset == 'OULU-NPU':
    flags.path.train_file = [os.path.join(dataset_path, 'mtcnn', 'Train_features'), os.path.join(dataset_path, 'Train_Depth_all')]
    flags.path.dev_file = [os.path.join(dataset_path, 'mtcnn', 'Dev_features'), os.path.join(dataset_path, 'Dev_Depth_all')]
    flags.path.test_file = [os.path.join(dataset_path, 'mtcnn', 'Test_features'), os.path.join(dataset_path, 'Test_Depth_all')]
elif flags.path.dataset == 'CASIA-FASD':
    flags.path.train_file = [os.path.join(dataset_path, 'Train_frame_mtcnn'), os.path.join(dataset_path, 'Train_Depth_all')]
    flags.path.dev_file = 'NoFile'
    flags.path.test_file = [os.path.join(dataset_path, 'Test_frame_mtcnn'), os.path.join(dataset_path, 'Test_Depth_all')]
elif flags.path.dataset == 'replayattack':
    flags.path.train_file = [os.path.join(dataset_path, 'mtcnn', 'Train_features'), os.path.join(dataset_path, 'Train_Depth_all')]
    flags.path.dev_file = [os.path.join(dataset_path, 'mtcnn', 'Devel_frames_choose'), os.path.join(dataset_path, 'Devel_Depth_frames_choose')]
    flags.path.test_file = [os.path.join(dataset_path, 'mtcnn', 'Test_features'), os.path.join(dataset_path, 'Test_Depth_all')]
elif flags.path.dataset == 'MSU-MFSD':
    flags.path.train_file = [os.path.join(dataset_path, 'mtcnn'), os.path.join(dataset_path, 'Depth_all')]
    flags.path.dev_file = 'NoFile'
    flags.path.test_file = [os.path.join(dataset_path, 'mtcnn'), os.path.join(dataset_path, 'Depth_all')]
flags.name=edict()
flags.name.padding_zero = True

# if flags.path.dataset == 'CASIA-FASD' or flags.path.dataset == 'replayattack':
#     flags.name.padding_zero = True

flags.dataset=edict()
flags.dataset.protocal = 'ijcb_protocal_3' #'ijcb_protocal_1'
if flags.path.dataset == 'OULU-NPU':
    path_gen_save = path_gen_save + '_' + flags.dataset.protocal.split('_')[-1] 

flags.path.model= path_gen_save #v10.1.1 for normal conv3d; v10.1.2 for 1.4 conv3d

flags.paras=edict()
flags.paras.isFocalLoss= False
flags.paras.isWeightedLoss= False
flags.paras.isRealAttackPair= False #(real, print1/print2/replay1/replay2) or (real, print1, print2, replay1, replay2)
flags.paras.isAugment= False
flags.paras.num_classes = 2
flags.paras.interval_seq = 3  # interval stride between concesive frames
flags.paras.len_seq = 1   # length of video sequence
flags.paras.stride_seq = 10 # sample stride of each sample
flags.paras.stride_seq_dev=64
flags.paras.fix_len = 16
flags.paras.resize_size=[256,256]
flags.paras.resize_size_face=[128,128]
flags.paras.reshape_size=[256,256,3]
flags.paras.reshape_size_face=[128,128,3]

flags.paras.batch_size_train = 5
flags.paras.batch_size_test = 6
flags.paras.hidden_size=16
flags.paras.learning_rate= 0.0001# 0.003#0.0001
flags.paras.padding_info = {'images':[256, 256, 3 * flags.paras.len_seq],
                            'maps': [32, 32, 1 * flags.paras.len_seq],
                            'masks': [32, 32, 1 * flags.paras.len_seq],
                            'labels':[1]
                        }

flags.paras.epoch = 100
flags.paras.epoch_eval = 2
flags.paras.shuffle_buffer=500
flags.paras.prefetch = flags.paras.batch_size_train * 2
flags.paras.depth_blank = True
flags.paras.inverse = True
flags.paras.depth_length = 32

if flags.path.dataset == 'CASIA-FASD' or flags.path.dataset == 'replayattack':
    flags.name.depth_blank = True

if flags.paras.inverse == False and os.path.isdir(flags.path.model):
    for f in os.listdir(flags.path.model):
        if '19501' in f:
            flags.paras.inverse = True
            break    

flags.display=edict()
flags.display.max_iter=20000
flags.display.display_iter=500
flags.display.summary_iter=500
flags.display.max_to_keeper=1024