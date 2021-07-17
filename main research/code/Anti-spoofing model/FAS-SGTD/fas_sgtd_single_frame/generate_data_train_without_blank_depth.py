import numpy as np
import tensorflow as tf
from PIL import Image
from FLAGS import flags
from util.util_dataset import *

import glob
import os
import cv2
import datetime
import random
import scipy.io as sio

# print(flags.paras.padding_info)

Dataset = tf.data.Dataset

suffix1 = 'scene.jpg'
suffix2 = 'depth1D.jpg'
suffix2_2 = 'depth.jpg'
suffix3 = 'scene.dat'
suffix4 = '.png'
face_scale = 1.3

interval_seq = flags.paras.interval_seq
num_classes = flags.paras.num_classes
padding_info = flags.paras.padding_info
fix_len = flags.paras.fix_len
name_padding_zero = flags.name.padding_zero
dataset = flags.path.dataset
depth_blank = flags.paras.depth_blank
label_inverse = flags.paras.inverse
depth_length = flags.paras.depth_length

def random_float(f_min, f_max):
    return f_min + (f_max-f_min) * random.random()


def Contrast_and_Brightness(img, alpha=None, gamma=None):
    gamma = random.randint(-40, 40)
    alpha = random_float(0.5, 1.5)
    dst = cv2.addWeighted(img, alpha, img, 0, gamma)
    return dst


def get_cut_out(img, y, x, length_new, length=50):
    h, w = img.shape[0], img.shape[1]    # Tensor [1][2],  nparray [0][1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)
        
    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    mask[y1: y2, x1: x2] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    img *= np.array(mask, np.uint8)
    return img


def get_face_info(image, face_name_full):
    f = open(face_name_full, 'r')
    lines = f.readlines()
    y1, x1, w, h = [float(ele) for ele in lines[:4]]
    f.close()
    y2 = y1+w
    x2 = x1+h

    y_mid = (y1+y2)/2.0
    x_mid = (x1+x2)/2.0
    w_img, h_img = image.size
    w_scale = 1.0*w
    h_scale = 1.0*h

    y1 = y_mid-w_scale/2.0 - random_float(0.1, 0.4)/2.0*w
    x1 = x_mid-h_scale/2.0 - random_float(0.1, 0.4)/2.0*h
    y2 = y_mid+w_scale/2.0 + random_float(0.1, 0.4)/2.0*w
    x2 = x_mid+h_scale/2.0 + random_float(0.1, 0.4)/2.0*h

    y1 = max(y1, 0.0)
    x1 = max(x1, 0.0)
    y2 = min(y2, float(w_img))
    x2 = min(x2, float(h_img))

    if random_float(0.0, 1.0) < 0.5:
        is_random_flip = True
    else:
        is_random_flip = False

    if random_float(0.0, 1.0) < 0.5:
        is_change_color = True
    else:
        is_change_color = False
    gamma = random.randint(-40, 40)
    alpha = random_float(0.5, 1.5)
    color_info = [is_change_color, alpha, gamma]

    if random_float(0.0, 1.0) < 0.5:
        is_cut_out = True
    else:
        is_cut_out = False
    length = 50
    y_cut_out = np.random.randint(h_img)
    x_cut_out = np.random.randint(w_img)
    length_cut_out = np.random.randint(1, length)
    cut_info = [is_cut_out, y_cut_out, x_cut_out, length_cut_out]

    face_info = [[y1, x1, y2, x2], is_random_flip, color_info, cut_info]
    return face_info


def crop_face_from_scene(image, face_info, is_depth):
    face_rect, is_random_flip, color_info, cut_info = face_info
    # print(face_info)

    region = image.crop(face_rect)

    if is_depth:
        if is_random_flip:
            region.transpose(Image.FLIP_LEFT_RIGHT)
        return region

    face = np.array(region)
    face = face[:, :, ::-1]

    if is_random_flip:
        face = cv2.flip(face, 1)

    if color_info[0]:  # is_change_color:
        face = Contrast_and_Brightness(face, color_info[1], color_info[2])

    if cut_info[0]:  # is_cut_out:
        face = get_cut_out(face, cut_info[1], cut_info[2], cut_info[3])

    face = face[:, :, ::-1]
    # region=image[x1:x2,y1:y2]
    region = Image.fromarray(face)
    return region
    lucky = 'lucky'


def exists_face_depth(name_padding_zero, path_image, name_pure, suffix, rangevar):
    if name_padding_zero:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full = os.path.join(path_image, name_pure, '%04d' % i + '_' + suffix)
            # print('face_name_full', face_name_full)
            if(not os.path.exists(face_name_full)):
                return False
    else:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full = os.path.join(path_image, name_pure, str(i) + '_' + suffix)
            if(not os.path.exists(face_name_full)):
                return False
    return True


def exists_face(name_padding_zero, path_image, name_pure, suffix, rangevar):
    if name_padding_zero:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full = os.path.join(path_image, name_pure, '%04d' % i + suffix)
            if(not os.path.exists(face_name_full)):
                return False
    else:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full = os.path.join(path_image, name_pure, str(i) + suffix)
            if(not os.path.exists(face_name_full)):
                return False
    return True


def get_res_list(res_list, label):
    if label == 0:
        fix_len_this = fix_len
    else:
        fix_len_this = int(fix_len/4)

    len_list = len(res_list)
    each_len = max(int(len_list/fix_len_this), 1)
    res_list_new = []
    for i in range(0, len_list, each_len):
        res_list_new.append(res_list[i])
    return res_list_new


def generate_existFaceLists_perfile(name_pure, path_scene, IMAGES):
    '''
    name_pure: pure name of each video
    IMAGES: image(frame) list of each video
    return: lists of [path_image, start_ind, end_ind, label, face_name_full]
    '''
    res_list = []
    len_seq = flags.paras.len_seq
    stride_seq = flags.paras.stride_seq
    num_image= len(IMAGES) + 100
    # num_image = len(IMAGES)
    # path_image=IMAGES[0][:-len(os.path.split(IMAGES[0])[-1])]
    path_image = IMAGES[0][:-len(os.path.split(IMAGES[0])[-1])]
    path_image = path_image[:-len(os.path.split(path_image)[-1])]
    path_image = os.path.join(IMAGES[0].split(name_pure[:2])[0])
    if dataset == 'OULU-NPU':
        path_image = os.path.join(IMAGES[0].split(name_pure[:2])[0])
    elif dataset == 'CASIA-FASD':
        path_image = os.path.join('/', *(IMAGES[0].split('/')[:-2]))
    elif dataset == 'replayattack':
        path_image = IMAGES[0].split(name_pure)[0]
    # name_pure = name_pure.split('-')[0]
    label_name = path_scene.split('/')[-1] if dataset == 'replayattack' else 'null'
    label_name2 = path_scene.split('/')[-2] if dataset == 'replayattack' else 'null'
    # print('label_name', label_name)
    if label_name == 'hack':  # casia and replayAttack
        label = 2
        if (name_pure.split('_')[0] == 'CASIA'):
            stride_seq *= 5  # down sampling for negative samples
    # replayattack
    elif label_name == 'real':
        label = 1
    elif label_name2 == 'attack':
        label = 0
    elif dataset == 'OULU-NPU':  # ijcb train and dev
        label = int(name_pure.split('_')[-1])
        if(label >= 2 and label <= 3):  # down sampling for negative samples
            stride_seq *= 8
        if(label >= 4 and label <= 5):  # down sampling for negative samples
            stride_seq *= 8
        if num_classes == 2:
            label = 1 if label == 1 else 2
        label = 1 if label == 1 else 0
        # label = label - 1
    elif dataset == 'CASIA-FASD':
        stride_seq *= 5 
        label = 0
        if name_pure == '1' or name_pure == '2' or name_pure == 'HR_1':
            label = 1
    if label_inverse:
        # 0, 1 -> 1, 0
        label = (label - 1) * -1
    # down sampling for negative samples
    start_ind = 1
    # end_ind=start_ind+ (len_seq-1)*interval_seq
    end_ind = start_ind + (len_seq - 1) * interval_seq
    while (end_ind < num_image):
    # if (end_ind > num_image):
    # if True:
        # print('%d-%d'%(start_ind,end_ind))
        feature_dict = {}
        # if(not exists_face_image(path_image,name_pure,suffix2, range(start_ind, end_ind+1, interval_seq))):
        if(not exists_face_depth(name_padding_zero, path_image, name_pure, suffix2_2, range(start_ind, end_ind + 1, interval_seq))):
            if exists_face(name_padding_zero, path_scene, name_pure, suffix4, range(start_ind, end_ind + 1, interval_seq)):
                face_name_full = 'no_depth'
                # face_name_full = os.path.join(path_image, name_pure, str(start_ind) + '_' + suffix2_2)
                res_list.append([name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full])
            start_ind += stride_seq
            end_ind += stride_seq
        # print('Lack of face image(s)')
            continue

        # feature_dict['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        # print('lal')
        # face_name_full=os.path.join(path_image,name_pure+'_%03d'%start_ind +'_'+suffix2)      
        face_name_full = os.path.join(path_image, name_pure, str(start_ind) + '_' + suffix2_2)
        res_list.append([name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full])

        start_ind += stride_seq
        end_ind += stride_seq
    
    res_list = get_res_list(res_list, label)
    # print('res_list', np.shape(res_list))
    return res_list

# Must be changed if inputs changed


def read_data_decode(name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full):
    name_pure = name_pure.decode()
    path_image = path_image.decode()
    path_scene = path_scene.decode()
    face_name_full = face_name_full.decode()

    image_face_list = []
    vertices_map_list = []

    # scene_name_full = os.path.join(path_scene, name_pure+'_%03d'%start_ind +'_'+suffix1)
    scene_name_full = os.path.join(path_scene, name_pure, str(start_ind) + suffix4) if name_padding_zero == False else os.path.join(path_scene, name_pure, '%04d' % start_ind + suffix4)
    face_dat_name = os.path.join(path_scene, name_pure+'_%03d' % start_ind + '_'+suffix3)
    image = Image.open(scene_name_full)
    # face_info = get_face_info(image, face_dat_name)

    for i in range(start_ind, end_ind+1, interval_seq):
        # scene_name_full = os.path.join(path_scene, name_pure+'_%03d'%i +'_'+suffix1)
        if name_padding_zero == False:
            scene_name_full = os.path.join(path_scene, name_pure, str(i) + suffix4)
            is_depth_file_exist = os.path.exists(os.path.join(path_image, name_pure, str(i) + '_' + suffix2_2))
        else:
            scene_name_full = os.path.join(path_scene, name_pure, '%04d' % i + suffix4)
            is_depth_file_exist = os.path.exists(os.path.join(path_image, name_pure, '%04d' % i + '_' + suffix2_2))
        # mesh_name_full = os.path.join(path_image, name_pure+'_%03d'%i +'_'+suffix2)

        if name_padding_zero == False:
            if is_depth_file_exist:
                mesh_name_full = os.path.join(path_image, name_pure, str(i) + '_' + suffix2_2)
        else:
            mesh_name_full = os.path.join(path_image, name_pure, '%04d' % i + '_' + suffix2_2)

        image = Image.open(scene_name_full)
        image_face = image
        # image_face = crop_face_from_scene(image, face_info, is_depth = False)
        #image_face = crop_face_from_scene(image, face_dat_name, face_scale)
        image_face = image_face.resize([padding_info['images'][0], padding_info['images'][1]])
        image_face = np.array(image_face, np.float32) - 127.5

        depth_cond = is_depth_file_exist
        if depth_blank:
            if dataset == 'OULU-NPU':
                real_cond = name_pure.split('_')[-1] == '1'
            elif dataset == 'CASIA-FASD' or dataset == 'replayattack':
                real_cond = True
            depth_cond = depth_cond and real_cond
        
        if depth_cond:
            depth1d = Image.open(mesh_name_full)
        else:
            depth1d = Image.fromarray(np.zeros((depth_length, depth_length)))

        #depth1d_face = crop_face_from_scene(depth1d, face_dat_name, face_scale)
        # depth1d_face = crop_face_from_scene(depth1d, face_info, is_depth = True)
        depth1d_face = depth1d
        depth1d_face = depth1d_face.resize([padding_info['maps'][0], padding_info['maps'][1]])
        vertices_map = np.array(depth1d_face, np.float32)
        vertices_map = np.expand_dims(vertices_map, axis=0)
        vertices_map = np.expand_dims(vertices_map, axis=-1)
        image_face_list.append(image_face)
        vertices_map_list.append(vertices_map)

    image_face_cat = np.concatenate(image_face_list, axis=-1)
    image_face_cat = image_face_cat.astype(np.float32)
    vertices_map_cat = np.concatenate(vertices_map_list, axis=-1)
    vertices_map_cat = vertices_map_cat.astype(np.float32)
    mask_cat = np.array(vertices_map_cat > 0.0, np.float32)
    mask_cat = mask_cat.astype(np.float32)
    if not label == 0:
        vertices_map_cat = np.zeros(vertices_map_cat.shape, dtype=np.float32)

    #print(label, image_face_cat.shape, vertices_map_cat.shape, mask_cat.shape)
    ALLDATA = [image_face_cat, vertices_map_cat, mask_cat]
    #print(np.concatenate(vertices_map_list, axis=-1).shape)

    return ALLDATA


class InputFnGenerator:
    def __init__(self, train_list):
        def find_path_scene(path_depthmap):
            path_gen_scene = []
            path_gen_depthmap, name_pure = os.path.split(path_depthmap)
            #print('path_gen_depthmap', path_gen_depthmap)
            #print('name_pure', name_pure)
            for path_list in train_list:
                if dataset == 'OULU-NPU':
                    if path_list[1] == path_gen_depthmap:
                        path_gen_scene = path_list[0]
                elif dataset == 'CASIA-FASD':
                    path_gen_depthmap, name_pure2 = os.path.split(path_gen_depthmap)    
                    if path_list[1] == path_gen_depthmap:
                        path_gen_scene = path_list[0]
                    path_gen_scene = os.path.join(path_gen_scene, name_pure2)
                elif dataset == 'replayattack':
                    path_gen_depthmap, name_pure2 = os.path.split(path_gen_depthmap)    
                    if name_pure2 == 'real':
                        if path_list[1] == path_gen_depthmap:
                            path_gen_scene = path_list[0]
                    else:
                        path_gen_depthmap, name_pure3 = os.path.split(path_gen_depthmap)
                        if path_list[1] == path_gen_depthmap:
                            path_gen_scene = path_list[0]
                        path_gen_scene = os.path.join(path_gen_scene, name_pure3)
                    path_gen_scene = os.path.join(path_gen_scene, name_pure2)
            # print('rrrrrrrrrrrrrrrrrrr', path_gen_scene)
            if path_gen_scene == []:
                print('Can\'t find correct path scene')
                exit(1)
            # name_pure = name_pure.split('_d')[0] + '.png'
            # path_scene = os.path.join(path_gen_scene, name_pure)
            path_scene = path_gen_scene
            return path_scene

        if(not type(train_list) == list):
            raise NameError
        FILES_LIST = []
        for fInd in range(len(train_list)):
            path_train_file = train_list[fInd]
            # FILES=glob.glob(os.path.join(path_train_file[1],'*'))
            FILES = glob.glob(os.path.join(path_train_file[0], '*'))
            if dataset == 'CASIA-FASD':
                for i in range(len(FILES)):
                    FILES[i] = glob.glob(os.path.join(FILES[i], '*'))
                # FILES = list(np.reshape(FILES, (-1)))
                FILES = [item for sublist in FILES for item in sublist]
            elif dataset == 'replayattack':
                for i in range(len(FILES)):
                    FILES[i] = glob.glob(os.path.join(FILES[i], '*'))
                    if not 'real' in FILES[i][0]:
                        for j in range(len(FILES[i])):
                            FILES[i][j] = glob.glob(os.path.join(FILES[i][j], '*')) 
                        FILES[i] = list(np.reshape(FILES[i], (-1)))  
                FILES = [item for sublist in FILES for item in sublist]
            for i in range(len(FILES)):
                FILES[i] = FILES[i].replace(path_train_file[0], path_train_file[1])
            FILES_LIST = FILES_LIST+FILES
        
        # select protocol of IJCB
        if dataset == 'OULU-NPU':
            data_object = IJCB(flags.dataset.protocal, 'train')
        elif dataset == 'CASIA-FASD':
            data_object = Casia('train') 
        elif dataset == 'replayattack':
            data_object = ReplayAttack('train')    
        FILES_LIST = data_object.dataset_process(FILES_LIST)

        self.existFaceLists_all = []
        for i in range(len(FILES_LIST)):
            path_image = FILES_LIST[i]
            path_scene = find_path_scene(path_image)
            name_pure = os.path.split(path_image)[-1]
            # IMAGES=glob.glob(os.path.join(path_image,'*'+suffix2))
            IMAGES = glob.glob(os.path.join(path_image, '*' + suffix2_2))
            if IMAGES == []:
                IMAGES = [os.path.join(path_image, '0_0_0_0.' + suffix2_2)]
            existFaceLists = generate_existFaceLists_perfile(name_pure, path_scene, IMAGES)
            self.existFaceLists_all += existFaceLists
        print("existFaceLists", len(self.existFaceLists_all))

    def input_fn_generator(self, shuffle):
        if shuffle:
            random.shuffle(self.existFaceLists_all)
        for existList in self.existFaceLists_all:
            [name_pure, path_image, path_scene, start_ind,
                end_ind, label, face_name_full] = existList
            ALLDATA = [name_pure.encode(), path_image.encode(), path_scene.encode(),
                       start_ind, end_ind, label, face_name_full.encode()]
            yield tuple(ALLDATA)

# Must be changed if inputs changed


def parser_fun(name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full):
    # name_pure=name_pure.decode()
    # path_image=path_image.decode()
    # face_name_full=face_name_full.decode()

    ALLDATA = tf.py_func(read_data_decode,
                         [name_pure, path_image, path_scene, start_ind,
                             end_ind, label, face_name_full],
                         [tf.float32, tf.float32, tf.float32]
                         )
    features = {}
    
    print(np.array(padding_info['images']).shape)
    print(ALLDATA[0])
    #print(features['images'].shape)
    
    features['images'] = tf.reshape(ALLDATA[0], padding_info['images']) / 255.0
    features['maps'] = tf.reshape(ALLDATA[1], padding_info['maps'])
    features['masks'] = tf.reshape(ALLDATA[2], padding_info['masks'])
    features['labels'] = tf.reshape(label, padding_info['labels'])
    features['names'] = tf.reshape(tf.cast(name_pure, tf.string), [1])

    return features


def input_fn_test():
    for i in range(100):
        yield np.array([i])


def input_fn_maker(train_list, shuffle=True, batch_size=None, epoch=1, padding_info=None):

    GEN_OBJ = InputFnGenerator(train_list)

    def input_fn():
        def input_fn_handle():
            return GEN_OBJ.input_fn_generator(shuffle)
        ds = Dataset.from_generator(input_fn_handle,
                                    (tf.string, tf.string, tf.string,
                                     tf.int32, tf.int32, tf.int32, tf.string)
                                    )
        if (flags.paras.prefetch > 1):
            ds = ds.prefetch(flags.paras.prefetch)
        ds = ds.map(parser_fun, num_parallel_calls=20)
        if (padding_info):
            ds.padded_batch(batch_size, padded_shapes=padding_info)
        else:
            ds = ds.batch(batch_size)
        ds = ds.repeat(epoch)
        ds = ds.filter(lambda x: tf.equal(
            tf.shape(x['images'])[0], flags.paras.batch_size_train))

        value = ds.make_one_shot_iterator().get_next()

        return value
    return input_fn()

    lucky = 1


if __name__ == '__main__':
    train_list = [flags.path.train_file]

    def input_fn_handle():
        return input_fn_generator(train_list)

    ds = input_fn_maker(train_list, shuffle=False, batch_size=3, epoch=2)
    value = ds()
    #value = ds().make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        start = datetime.datetime.now()
        # for x in range(100):
        x = 0
        while True:
            val_, maps_, names_ = sess.run(
                [value['images'], value['maps'], value['names']])
            # print(x, val_.shape, maps_.shape, names_[0][0].decode() )

            val_ = val_[0, :, :, 0:3]
            image = val_ + 127.5
            image = np.squeeze(image)
            image = np.array(image, dtype=np.uint8)
            image_pil = Image.fromarray(image)

            maps_ = maps_[0, :, :, 0]
            depth = np.squeeze(maps_)
            depth = np.array(depth, dtype=np.uint8)
            depth_pil = Image.fromarray(depth)
            #depth_pil = depth_pil.resize((32, 32))

            name_pure = names_[0][0].decode()
            if True:  # name_pure.split('_')[-1] == '1':
                image_pil.save('./tmp/%d_image.bmp' % (x))
                depth_pil.save('./tmp/%d_maps.bmp' % (x))

            lucky = 1
            x += 1
            #print(x, val_[0].shape, val_[1].shape, val_[2].shape)
        end = datetime.datetime.now()
        # print('Time consuming:', (end-start).seconds )

    lucky = 1
