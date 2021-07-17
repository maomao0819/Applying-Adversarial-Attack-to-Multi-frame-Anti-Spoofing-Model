from easydict import EasyDict as edict
import os

flags=edict()

flags.path=edict()

# path_gen_save = './model_finetune/'
path_gen_save = './model_github_backup/'
# path_gen_save = './model_finetune_OULU_2'
# path_gen_save = './model_finetune_with_blank_19501_inverse'
# path_gen_save = './model_finetune_CASIA_19501_inverse'
# path_gen_save = './model_finetune_replayattack'
# path_gen_save = './model_finetune_replayattack'
# path_gen_save = './model_multipler_2'

flags.path.dataset = 'OULU-NPU'
# flags.path.dataset = 'CASIA-FASD'
# flags.path.dataset = 'replayattack'

# path_gen_save = os.path.join(path_gen_save, flags.path.dataset)

# flags.path.train_file=['/mnt/sda1/mike/OULU-NPU/Train_features_cont_frame_nested', '/mnt/sda1/mike/OULU-NPU/Train_Depth_features_cont_frame_nested']
# flags.path.train_file=['/mnt/sda1/mike/OULU-NPU/Train_frames', '/mnt/sda1/mike/OULU-NPU/Train_Depth_all']
flags.path.train_file=['/mnt/sda1/mike/OULU-NPU/mtcnn/Train_features', '/mnt/sda1/mike/OULU-NPU/Train_Depth_all']

#flags.path.dev_file=['/mnt/sda1/mike/OULU-NPU/Dev_features_cont_frame_nested', '/mnt/sda1/mike/OULU-NPU/Dev_Depth_features_cont_frame_nested']
# flags.path.dev_file=['/mnt/sda1/mike/OULU-NPU/Dev_frames', '/mnt/sda1/mike/OULU-NPU/Dev_Depth_all']
flags.path.dev_file=['/mnt/sda1/mike/OULU-NPU/mtcnn/Dev_features', '/mnt/sda1/mike/OULU-NPU/Dev_Depth_all']
#flags.path.dev_file=['/mnt/sda1/mike/OULU-NPU/little_test_frame', '/mnt/sda1/mike/OULU-NPU/little_test_depth']

# flags.path.test_file=['/mnt/sda1/mike/OULU-NPU/Test_features_cont_frame_nested', '/mnt/sda1/mike/OULU-NPU/Test_Depth_features_cont_frame_nested']
#flags.path.test_file=['/mnt/sda1/mike/OULU-NPU/little_test_frame', '/mnt/sda1/mike/OULU-NPU/little_test_depth']
flags.path.test_file=['/mnt/sda1/mike/OULU-NPU/Test_frames', '/mnt/sda1/mike/OULU-NPU/Test_Depth_all']
# flags.path.test_file=['/mnt/sda1/mike/OULU-NPU/mtcnn/Test_features_rm', '/mnt/sda1/mike/OULU-NPU/Test_Depth_all']

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

flags.path.model = path_gen_save #v10.1.1 for normal conv3d; v10.1.2 for 1.4 conv3d
flags.path.single_frame_model = os.path.join('model_single_frame_multipler_2', flags.path.dataset)

flags.name=edict()
flags.name.padding_zero = True

# if flags.path.dataset == 'CASIA-FASD' or flags.path.dataset == 'replayattack':
#     flags.name.padding_zero = True

flags.dataset=edict()
flags.dataset.protocal = 'ijcb_protocal_3' #'ijcb_protocal_1'
if flags.path.dataset == 'OULU-NPU':
    path_gen_save = path_gen_save + '_' + flags.dataset.protocal.split('_')[-1]   
    # flags.path.model = path_gen_save #v10.1.1 for normal conv3d; v10.1.2 for 1.4 conv3d
    flags.path.single_frame_model = flags.path.single_frame_model + '_' + flags.dataset.protocal.split('_')[-1]
# flags.path.model = './model_multipler_2/OULU-NPU_1'
flags.paras=edict()
flags.paras.isFocalLoss= False
flags.paras.isWeightedLoss= False
flags.paras.isRealAttackPair= False #(real, print1/print2/replay1/replay2) or (real, print1, print2, replay1, replay2)
flags.paras.isAugment= False
flags.paras.num_classes = 2
flags.paras.interval_seq = 3  # interval stride between concesive frames
# flags.paras.interval_seq = 10  # interval stride between concesive frames
flags.paras.len_seq = 5   # length of video sequence
# flags.paras.len_seq = 10   # length of video sequence
flags.paras.stride_seq = 10 # sample stride of each sample
flags.paras.stride_seq_dev=64
flags.paras.fix_len = 16
flags.paras.resize_size=[256,256]
flags.paras.resize_size_face=[128,128]
flags.paras.reshape_size=[256,256,3]
flags.paras.reshape_size_face=[128,128,3]

flags.paras.batch_size_train = 2
flags.paras.batch_size_test = 6
flags.paras.hidden_size=16
flags.paras.learning_rate= 0.01# 0.003#0.0001
flags.paras.padding_info = {'images':[256, 256, 3 * flags.paras.len_seq],
                            'maps': [32, 32, 1 * flags.paras.len_seq],
                            'masks': [32, 32, 1 * flags.paras.len_seq],
                            'labels':[1]
                        }

flags.paras.single_ratio = 0.4# 0.9# 0.9 #0.5 #0.01
flags.paras.cla_ratio = 0.8#0.1#0.8 #0.01

flags.paras.epoch = 1000
flags.paras.epoch_eval = 2
flags.paras.shuffle_buffer=500
flags.paras.prefetch = flags.paras.batch_size_train * 2
flags.paras.depth_blank = True
flags.paras.inverse = True
# flags.paras.inverse = True -> real = 0, spoof = 1
# flags.paras.inverse = False -> real = 1, spoof = 0
flags.paras.depth_length = 32
flags.paras.attack_type = 1
# flags.paras.attack_type:    1: >=
#                             2: >
#                             3: >= 0.5
#                             4: > 0.5

if flags.path.dataset == 'CASIA-FASD' or flags.path.dataset == 'replayattack':
    flags.name.depth_blank = True

if flags.paras.inverse == False and os.path.isdir(flags.path.model):
    for f in os.listdir(flags.path.model):
        if '19501' in f:
            flags.paras.inverse = True
            break    

flags.display=edict()
flags.display.max_iter=300000
flags.display.display_iter=500
flags.display.log_iter=100
flags.display.summary_iter=100
flags.display.max_to_keeper=102400