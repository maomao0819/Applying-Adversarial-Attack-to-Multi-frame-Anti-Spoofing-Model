import numpy as np 
import tensorflow as tf
from PIL import Image
from FLAGS_tf1 import flags
# tf1
from util_tf1.util_dataset import *
# tf2
# from util.util_dataset import *

import glob 
import os
import datetime
import random
import scipy.io as sio

#print(flags.paras.padding_info)

Dataset=tf.data.Dataset

suffix1='scene.jpg'
suffix2='depth1D.jpg'
suffix2_2 ='depth.jpg'
suffix3='scene.dat'
suffix4 = '.png'
#scale_scene=2.0
#scale_face=1.2
face_scale = 1.3

interval_seq=flags.paras.interval_seq
num_classes = flags.paras.num_classes
padding_info = flags.paras.padding_info

name_padding_zero = flags.name.padding_zero
dataset = flags.path.dataset
depth_blank = flags.paras.depth_blank
label_inverse = flags.paras.inverse
depth_length = flags.paras.depth_length

def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(y1,0.0)
    x1=max(x1,0.0)
    y2=min(y2,float(w_img))
    x2=min(x2,float(h_img))

    region=image.crop([y1,x1,y2,x2])
    return region
    lucky='lucky'

# depth face
def exists_face_image(path_image,name_pure,suffix,rangevar):
    if dataset == 'MSU-MFSD':
        for i in rangevar:
            face_name_full = os.path.join(path_image, name_pure, name_pure + '_' + '%03d' % i + '_' + suffix)
            # print('face_name_full', face_name_full)
            if(not os.path.exists(face_name_full)):
                # print('face_name_full', face_name_full)
                return False
    else:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full=os.path.join(path_image, name_pure, str(i) + '_' + suffix)
            if(not os.path.exists(face_name_full)):
                return False
    return True

def exists_face(name_padding_zero, path_image,name_pure,suffix,rangevar):
    if name_padding_zero:
        if dataset == 'MSU-MFSD':
            for i in rangevar:
                face_name_full = os.path.join(path_image, name_pure, name_pure + '_' + '%03d' % i + suffix)
                # print('face_name_full', face_name_full)
                if(not os.path.exists(face_name_full)):
                    # print('face_name_full', face_name_full)
                    return False
        else:
            for i in rangevar:
                # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
                face_name_full=os.path.join(path_image, name_pure, '%04d'%i + suffix)
                if(not os.path.exists(face_name_full)):
                    return False
    else:
        for i in rangevar:
            # face_name_full=os.path.join(path_image,name_pure+'_%03d'%i +'_'+suffix2)
            face_name_full=os.path.join(path_image, name_pure, str(i) + suffix)
            if(not os.path.exists(face_name_full)):
                return False
    return True

def get_res_list(res_list):
    len_list = len(res_list)
    each_len = int(len_list/6) if int(len_list/6)>0 else 1
    res_list_new = []
    for i in range(0, len_list, each_len):
        res_list_new.append(res_list[i])
    return res_list_new
    
def generate_existFaceLists_perfile(name_pure, IMAGES, path_scene):
    '''
    name_pure: pure name of each video
    IMAGES: image(frame) list of each video
    return: lists of [path_image, start_ind, end_ind, label, face_name_full]
    '''
    res_list=[]

    len_seq=flags.paras.len_seq
    stride_seq=1#flags.paras.stride_seq * 16
    num_image= len(IMAGES) + 100
    # num_image = len(IMAGES)

    # path_image=IMAGES[0][:-len(os.path.split(IMAGES[0])[-1])]
    path_image=IMAGES[0][:-len(os.path.split(IMAGES[0])[-1])]
    path_image=path_image[:-len(os.path.split(path_image)[-1])]
    path_image = os.path.join(IMAGES[0].split(name_pure[:2])[0])
    if dataset == 'OULU-NPU':
        path_image = os.path.join(IMAGES[0].split(name_pure[:2])[0])
    elif dataset == 'CASIA-FASD':
        path_image = os.path.join('/', *(IMAGES[0].split('/')[:-2]))
    elif dataset == 'replayattack':
        path_image = IMAGES[0].split(name_pure)[0]    
    elif dataset == 'MSU-MFSD':
        path_image = IMAGES[0].split(name_pure)[0]

    label_name = 'null'
    label_name2 = 'null'
    if dataset == 'replayattack':
        label_name = path_scene.split('/')[-1]
        label_name2 = path_scene.split('/')[-2]
    elif dataset == 'MSU-MFSD':
        label_name = path_scene.split('/')[-1]
        label_name2 = path_scene.split('/')[-1]
    if label_name=='hack': # casia and replayAttack
        label=2
        if (name_pure.split('_')[0]=='CASIA'):
            stride_seq*=5 # down sampling for negative samples 
    elif label_name=='real' :
        label = 1
    elif label_name2 == 'attack':
        label = 0
    elif dataset == 'OULU-NPU': # ijcb train and dev
        label=int(name_pure.split('_')[-1])
        if(label>=2 and label<=3): # down sampling for negative samples 
            stride_seq *= 1
        if(label>=4 and label<=5): # down sampling for negative samples 
            stride_seq *= 1
        if num_classes == 2:
            label=1 if label==1 else 2
        label=1 if label==1 else 0
        #label = label - 1
    elif dataset == 'CASIA-FASD':
        # stride_seq *= 5 
        label = 0
        if name_pure == '1' or name_pure == '2' or name_pure == 'HR_1':
            label = 1
    if label_inverse:
        # 0, 1 -> 1, 0
        label = (label - 1) * -1
    # down sampling for negative samples  
    start_ind=1
    end_ind=start_ind+ (len_seq-1)*interval_seq
    # while (end_ind<num_image):
    # if (end_ind > num_image):
    if True:
        #print('%d-%d'%(start_ind,end_ind))
        feature_dict={}
        if(not exists_face_image(path_image,name_pure,suffix2_2, range(start_ind, end_ind, interval_seq))):
            if exists_face(name_padding_zero, path_scene,name_pure,suffix4, range(start_ind, end_ind, interval_seq)):
                face_name_full='no_depth'
                res_list.append([path_image, start_ind, end_ind, label, face_name_full])   
            else:
                start_ind+=stride_seq
                end_ind+=stride_seq
            #print('Lack of face image(s)')
            # continue
        #feature_dict['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        else:
            face_name_full=os.path.join(path_image, name_pure, str(start_ind) + '_' + suffix2_2)
            res_list.append([path_image, start_ind, end_ind, label, face_name_full])
            start_ind+=stride_seq
            end_ind+=stride_seq
    #return res_list
    return get_res_list(res_list)
    '''
    if len(res_list) > 0:
        return [ res_list[0] ]
    else:
        return []
    '''

# Must be changed if inputs changed
def read_data_decode(name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full):
    name_pure=name_pure.decode()
    path_image=path_image.decode()
    path_scene=path_scene.decode()
    face_name_full=face_name_full.decode()

    image_face_list = []
    vertices_map_list = []

    face_dat_name = os.path.join(path_scene, name_pure+'_%03d'%start_ind +'_'+suffix3)

    for i in range(start_ind, end_ind+1, interval_seq):
        # scene_name_full = os.path.join(path_scene, name_pure+'_%03d'%i +'_'+suffix1)
        if name_padding_zero == False:
            scene_name_full = os.path.join(path_scene, name_pure, str(i) + suffix4)
            while os.path.exists(scene_name_full) == False:
                if os.path.exists(scene_name_full):
                    break
                i += 1
                scene_name_full = os.path.join(path_scene, name_pure, str(i) + suffix4)
            is_depth_file_exist = os.path.exists(os.path.join(path_image, name_pure, str(i) + '_' + suffix2_2))
        else:
            if dataset == 'MSU-MFSD':
                scene_name_full = os.path.join(path_scene, name_pure, name_pure + '_' + '%03d' % i + suffix4)
            else:
                scene_name_full = os.path.join(path_scene, name_pure, '%04d'%i + suffix4)
            while os.path.exists(scene_name_full) == False:
                if os.path.exists(scene_name_full):
                    break
                i += 1
                if dataset == 'MSU-MFSD':
                    scene_name_full = os.path.join(path_scene, name_pure, name_pure + '_' + '%03d' % i + suffix4)
                else:
                    scene_name_full = os.path.join(path_scene, name_pure, '%04d'%i + suffix4)
            if dataset == 'MSU-MFSD':
                is_depth_file_exist = os.path.exists(os.path.join(path_image, name_pure, name_pure + '_' + '%03d' % i + '_' + suffix2_2))    
            else:
                is_depth_file_exist = os.path.exists(os.path.join(path_image, name_pure, '%04d'%i + '_' + suffix2_2))
        # mesh_name_full = os.path.join(path_image, name_pure+'_%03d'%i +'_'+suffix2)
        if is_depth_file_exist:
            if name_padding_zero == False:
                mesh_name_full = os.path.join(path_image, name_pure, str(i) + '_' + suffix2_2)  
            else:
                if dataset == 'MSU-MFSD':
                    mesh_name_full = os.path.join(path_image, name_pure, name_pure + '_' + '%03d' % i + '_' + suffix2_2)
                else:
                    mesh_name_full = os.path.join(path_image, name_pure, '%04d'%i + '_' + suffix2_2)
        
        # if is_depth_file_exist == False:     
        #     # print(is_depth_file_exist, os.path.join(path_image, name_pure, '%04d'%i + '_' + suffix2_2))
        #     continue
        
        image = Image.open(scene_name_full)
        image_face = image
        # image_face = crop_face_from_scene(image, face_dat_name, face_scale)
        image_face = image_face.resize([padding_info['images'][0], padding_info['images'][1]])
        image_face = np.array(image_face, np.float32) - 127.5
        
        depth_cond = is_depth_file_exist
        if depth_blank:
            if dataset == 'OULU-NPU':
                #real_cond = name_pure.split('_')[-1] == '1'
                real_cond = True
            elif dataset == 'CASIA-FASD' or dataset == 'replayattack' or dataset == 'MSU-MFSD':
                # real_cond = (name_pure == '1' or name_pure == '2' or name_pure == 'HR1')
                real_cond = True
            depth_cond = depth_cond and real_cond
        
        if depth_cond:
            depth1d = Image.open(mesh_name_full)
            depth1d = depth1d.resize((depth_length, depth_length))
        else:
            depth1d = Image.fromarray(np.zeros((depth_length, depth_length)))
 
        # depth1d_face = crop_face_from_scene(depth1d, face_dat_name, face_scale)
        depth1d_face = depth1d
        depth1d_face = depth1d_face.resize([padding_info['maps'][0], padding_info['maps'][1]])
        vertices_map = np.array(depth1d_face, np.float32)
        vertices_map = np.expand_dims(vertices_map, axis = 0)
        vertices_map = np.expand_dims(vertices_map, axis = -1)
        image_face_list.append(image_face)
        vertices_map_list.append(vertices_map)

    image_face_cat = np.concatenate(image_face_list, axis=-1)
    image_face_cat = image_face_cat.astype(np.float32)
    vertices_map_cat = np.concatenate(vertices_map_list, axis=-1)
    vertices_map_cat = vertices_map_cat.astype(np.float32)
    mask_cat = np.array(vertices_map_cat > 0.0, np.float32)
    mask_cat = mask_cat.astype(np.float32)
    if label_inverse:
        if not label == 0:
            vertices_map_cat = np.zeros(vertices_map_cat.shape, dtype=np.float32)
    else:
        if not label == 1:
            vertices_map_cat = np.zeros(vertices_map_cat.shape, dtype=np.float32)      

    #print(label, image_face_cat.shape, vertices_map_cat.shape, mask_cat.shape)
    ALLDATA=[image_face_cat, vertices_map_cat, mask_cat]
    #print(np.concatenate(vertices_map_list, axis=-1).shape)

    return ALLDATA

def input_fn_generator(train_list, shuffle):
    def find_path_scene(path_depthmap):
        path_gen_scene = []
        path_gen_depthmap, name_pure=os.path.split(path_depthmap)
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
            elif dataset == 'MSU-MFSD':
                path_gen_depthmap, name_pure2 = os.path.split(path_gen_depthmap)    
                if path_list[1] == path_gen_depthmap:
                    path_gen_scene = path_list[0]
                path_gen_scene = os.path.join(path_gen_scene, name_pure2)
        
        if path_gen_scene == []:
            print('Can\'t find correct path scene')
            exit(1)
        # path_scene = os.path.join(path_gen_scene, name_pure)  
        path_scene = path_gen_scene      
        return path_scene

    if(not type(train_list)==list):
        raise NameError
    FILES_LIST=[]
    for fInd in range(len(train_list)):
        path_train_file=train_list[fInd]
        # FILES=glob.glob(os.path.join(path_train_file[1],'*'))
        FILES=glob.glob(os.path.join(path_train_file[0],'*'))
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
        elif dataset == 'MSU-MFSD':
            remove_idx = []
            for i in range(len(FILES)):
                if not 'choose' in FILES[i]:
                    FILES[i] = glob.glob(os.path.join(FILES[i], '*'))
                else:
                    remove_idx.append(i)
            for i in range(len(remove_idx)):
                FILES.pop(remove_idx[i] - i)
            FILES = [item for sublist in FILES for item in sublist]
        for i in range(len(FILES)):
            FILES[i] = FILES[i].replace(path_train_file[0], path_train_file[1])
        FILES_LIST = FILES_LIST+FILES

    ## select protocol of IJCB
    if dataset == 'OULU-NPU':
        FILES_LIST = IJCB(flags.dataset.protocal, 'test').dataset_process(FILES_LIST)#\
                # + IJCB(flags.dataset.protocal, 'dev').dataset_process(FILES_LIST)
    elif dataset == 'CASIA-FASD':
        FILES_LIST = Casia('test').dataset_process(FILES_LIST) 
    elif dataset == 'replayattack':
        FILES_LIST = ReplayAttack('test').dataset_process(FILES_LIST)
    elif dataset == 'MSU-MFSD':
        FILES_LIST = MSU_MFSD('test').dataset_process(FILES_LIST)
    if shuffle:
        random.shuffle(FILES_LIST) # random shuffle

    for i in range(len(FILES_LIST)):
        path_image=FILES_LIST[i]
        path_scene = find_path_scene(path_image)
        name_pure=os.path.split(path_image)[-1]
        #print(i,name_pure)
        # IMAGES=glob.glob(os.path.join(path_image,'*'+suffix2))
        IMAGES=glob.glob(os.path.join(path_image, '*' + suffix2_2))
        if IMAGES == []:
            IMAGES = [os.path.join(path_image, '0_0_0_0.' + suffix2_2)]
        if shuffle:
            random.shuffle(IMAGES) # random shuffle
        # if dataset == 'OULU-NPU':
        #     folder_lack_image = ['6_3_40_1', '6_2_44_1']
        #     if name_pure in folder_lack_image:
        #         continue
        existFaceLists=generate_existFaceLists_perfile(name_pure, IMAGES, path_scene)
        for existList in existFaceLists:
            [path_image, start_ind, end_ind, label, face_name_full]=existList
            ALLDATA=[name_pure.encode(), path_image.encode(), path_scene.encode(), \
                    start_ind, end_ind, label, face_name_full.encode()]
            #ALLDATA=read_data_decode(name_pure, path_image, start_ind, end_ind, face_name_full)
            #ALLDATA.append(np.array([label],np.int32))                
            yield tuple(ALLDATA)

# Must be changed if inputs changed
def parser_fun(name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full):
    #name_pure=name_pure.decode()
    #path_image=path_image.decode()
    #face_name_full=face_name_full.decode()
    ALLDATA=tf.py_func(read_data_decode,
                    [name_pure, path_image, path_scene, start_ind, end_ind, label, face_name_full],
                    [tf.float32, tf.float32, tf.float32]
                    )
    
    features={}    
    
    features['images']=tf.reshape(ALLDATA[0], padding_info['images']) / 255.0
    features['maps']=tf.reshape(ALLDATA[1], padding_info['maps'])
    features['masks']=tf.reshape(ALLDATA[2], padding_info['masks'])
    features['labels']=tf.reshape(label, padding_info['labels'])    
    features['names']=tf.reshape(tf.cast(name_pure, tf.string), [1])
    return features

def input_fn_test():
    for i in range(100):
        yield np.array([i])

def input_fn_maker(train_list, shuffle=True, batch_size=None, epoch=1, padding_info=None):
    def input_fn():
        def input_fn_handle():
            return input_fn_generator(train_list, shuffle)

        ds=Dataset.from_generator(input_fn_handle, \
                     (tf.string, tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.string)
                     )

        if (flags.paras.prefetch>1):
            ds=ds.prefetch(flags.paras.prefetch)
        ds=ds.map(parser_fun, num_parallel_calls=20)
        if (shuffle):
            ds=ds.shuffle(buffer_size=flags.paras.shuffle_buffer)
        if (padding_info):
            ds.padded_batch(batch_size, padded_shapes=padding_info)
        else:
            ds=ds.batch(batch_size)
        ds=ds.repeat(epoch)

        value = ds.make_one_shot_iterator().get_next()
        return value
    return input_fn()

    lucky=1

if __name__=='__main__':
    train_list=[flags.path.train_file]

    def input_fn_handle():
        return input_fn_generator(train_list)
    
    ds=input_fn_maker(train_list, shuffle=False, batch_size=3, epoch=2)
    value=ds()
    #value = ds().make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        start=datetime.datetime.now()  
        #for x in range(100):
        x = 0
        while True:
            val_, maps_, names_=sess.run( [value['images'], value['maps'], value['names'] ]) 
            print(x, val_.shape, maps_.shape, names_[0][0].decode() )

            val_ = val_[0,:,:,0:3]            
            image = val_ + 127.5
            image = np.squeeze(image)
            image = np.array(image, dtype = np.uint8)
            image_pil = Image.fromarray(image)
            
            maps_ = maps_[0, :, :, 0]
            depth = np.squeeze(maps_)
            depth = np.array(depth, dtype = np.uint8)
            depth_pil = Image.fromarray(depth)
            #depth_pil = depth_pil.resize((32, 32))

            name_pure = names_[0][0].decode()
            if True:#name_pure.split('_')[-1] == '1':
                image_pil.save('./tmp/%d_image.bmp'%(x))
                depth_pil.save('./tmp/%d_maps.bmp'%(x))
            

            lucky=1   
            x += 1
            #print(x, val_[0].shape, val_[1].shape, val_[2].shape)
        end=datetime.datetime.now() 
        print('Time consuming:', (end-start).seconds )
    
    lucky=1
