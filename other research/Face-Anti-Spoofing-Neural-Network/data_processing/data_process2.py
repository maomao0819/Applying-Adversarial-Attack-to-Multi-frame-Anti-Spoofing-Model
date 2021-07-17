import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf, reconstruct_paf_anchor
import argparse
import torch.backends.cudnn as cudnn
import os
from PIL import Image

STD_SIZE = 120

def resize_depth(imgdepth):
    img=np.array(imgdepth)
    res=np.zeros((32,32,1),dtype=float)
    for x in range(32):
        realx = 8*x
        for y in range(32):
            realy=8*y
            res[x,y,0]=np.mean(img[realx:realx+8,realy:realy+8])
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    args = parser.parse_args()

    folder = {}
    indice = 0

    for phone in range(1,6):
        for session in range(1,4):
            for user in range(1,21):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    currentFrame=0
                    while(currentFrame<5):
                        # name = '/mnt/sda1/mike/OULU-NPU/Train_files_OULU/Train_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        name = '/mnt/sda1/mike/OULU-NPU/Train_features/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'-'+ str(currentFrame)+'.png'
                        currentFrame += 1
                        if os.path.exists(name) == False:

                            continue
                        folder[str(indice)]=name

                        
                        indice += 1

    for phone in range(1,6):
        for session in range(1,4):
            for user in range(21,36):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    currentFrame=0
                    while(currentFrame<5):
                        # name = '/mnt/sda1/mike/OULU-NPU/Dev_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        name = '/mnt/sda1/mike/OULU-NPU/Dev_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'-'+str(currentFrame)+'.png'
                        currentFrame += 1
                        if os.path.exists(name) == False:
                            continue
                        folder[str(indice)]=name

                        
                        indice += 1

    for phone in range(1,6):
        for session in range(1,4):
            for user in range(36,56):
                for file in range(1,6):
                    nom= str(user)
                    if user<10:
                        nom = '0'+nom
                    currentFrame=0
                    while(currentFrame<5):
                        # name = '/mnt/sda1/mike/OULU-NPU//Test_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'_'+str(currentFrame)+'.png'
                        name = '/mnt/sda1/mike/OULU-NPU//Test_files/'+str(phone)+'_'+str(session)+'_'+nom+'_'+str(file)+'-'+str(currentFrame)+'.png'
                        currentFrame += 1
                        if os.path.exists(name) == False:
                            continue
                        folder[str(indice)]=name

                        # currentFrame += 1
                        indice += 1
    
    label = np.load('/mnt/sda1/maomao/Face-Anti-Spoofing-Neural-Network/data_processing/label.npz')

    Anchors = {}
    Labels_D = {}
    
    # 1. Enregistrement des modèles pré-entraînés
    checkpoint_fp = '/mnt/sda1/maomao/Face-Anti-Spoofing-Neural-Network/data_processing/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    #model = getattr('/mnt/sda1/maomao/Face-Anti-Spoofing-Neural-Network/data_processing/mobilenet_v1.py', arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    #model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model = mobilenet_v1.mobilenet_1(num_classes=62, input_channel=3)
    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. Fonction de détection des visages
    face_detector = dlib.get_frontal_face_detector()

    # 3. Détection
    tri = sio.loadmat('/mnt/sda1/maomao/Face-Anti-Spoofing-Neural-Network/data_processing/visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    for item in folder:
        print('folder[item]', folder[item])
        if os.path.exists(folder[item]) == False:
            continue
        img_ori = cv2.imread(folder[item])
        rects = face_detector(img_ori, 1)
        if len(rects) != 0:
        
            for rect in rects:
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)
                img = crop_img(img_ori, roi_box)
                img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                
            vertices_lst = [] 
            # print('param', param)
            # print('param shape', np.shape(param))
            vertices = predict_dense(param, roi_box)
            vertices_lst.append(vertices)
            #print('param shape', np.shape(param))
            #print(param)
            #print(args.paf_size)
            # Anchor = gen_anchor(param=param,kernel_size=args.paf_size)
            # Anchor = gen_anchor(param, args.paf_size)
            # Anchor = reconstruct_paf_anchor(param)
            #print(Anchor)
            Anchor = gen_img_paf(img_crop=img, param=param)
            Anchors[item]=Anchor
            
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1) 
            if(int(item)%100==0):
                print(item)
            if label[item]==1: #real face
                Labels_D[item]=resize_depth(depths_img)
            else: #spoof face
                Labels_D[item]=np.zeros((32,32,1),dtype=float)
        else:
            #Case our cropping didn't work
            Anchors[item] = np.zeros((2,4096),dtype=float)
            print('fausse image:'+item)
            Labels_D[item] = np.zeros((32,32,1),dtype=float)
        #print(Anchors[item])
                    
    np.savez("anchors.npz",**Anchors)
    np.savez("labels_D.npz",**Labels_D)
    np.savez("folder.npz",**folder)
