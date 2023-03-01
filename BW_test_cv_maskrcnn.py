#In[]
# from ast import MatchClass
import glob
import os
from stringprep import map_table_b2
import cv2
import copy
import numpy as np
from sklearn.model_selection import cross_val_predict
import tqdm 
import random
import pickle
from skimage.io import imread, imshow
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow as tf

from PA_test_cv import proMask_produce

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
category = 7

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.per_process_gpu_memory_fraction = 0.6 # 使用50%的GPU暫存  
session = tf.compat.v1.Session(config=cfg )

# weights_aware_categorical_crossentropy loss
def weights_categorical_crossentropy_loss(y_true, y_pred):
    weight = 5
    wbce = K.categorical_crossentropy(y_true, y_pred)
    if(y_true.shape[0]):
        weight_map = np.ones((y_true.shape[:-1]), dtype=np.float32)
        weight_map[weight_map[...,-1]==1] = weight
        weight_map = tf.convert_to_tensor(weight_map)
        wbce = wbce * weight_map

    return wbce

def focal_loss(y_true, y_pred, gamma=float(2.), alpha=float(0.25)):

    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})
    Returns:
        [tensor] -- loss.
    """
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


def retrieve_mask(maskname):
    f = open(maskname, 'rb')
    masklist = pickle.load(f)
    teeth_mask_list = []
    for idx, mask in enumerate(masklist['masks']):
        teeth_mask_list.append(mask.astype(int))
    
    f.close()
    
    return teeth_mask_list


def retrieve_bbox(img):
    y_max = max(np.argwhere(img==1)[:, 0])
    y_min = min(np.argwhere(img==1)[:, 0])
    x_max = max(np.argwhere(img==1)[:, 1])
    x_min = min(np.argwhere(img==1)[:, 1])

    return x_min, y_min, x_max, y_max

'''
蛀牙評估
'''
def IOU(pred, gt):
    pred_temp = copy.deepcopy(pred)
    gt_temp = copy.deepcopy(gt)
    # pred_temp[pred_temp != 0] = 1
    # gt_temp[gt_temp != 0] = 1
    temp = pred_temp + gt_temp
    universe = np.bincount(temp.flatten())
    print(universe)
    if universe.shape[0] < 3:
        return 0

    iou = universe[2] / (universe[1] + universe[2])

    return iou 


def retrieve_carries(ori_img, fullteeth=None, carries=0, carries_threshold=0.003):
    image = copy.deepcopy(ori_img.astype('uint8'))
    if carries == 0:
        image[image != 6] = 0
    
    image[fullteeth == 0] = 0
    num, label, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    # print(stats)
    area = image.shape[0] * image.shape[1]
    # print('\n before: ', num)
    # print(np.bincount(label.flatten()))
    for idx in range(1, num):
        carriesarea = stats[idx][-1]
        ratio = carriesarea / area
        if ratio < carries_threshold:
            label[label == idx] = 0
            num -= 1
            continue
    
    le = LabelEncoder()
    le.fit(label.flatten())
    temp = le.transform(label.flatten())
    # print('\nSort: ', np.bincount(temp))
    temp = np.reshape(temp, (ori_img.shape[0], ori_img.shape[1]))
    
    return num, temp

def hungarian(iou):
    """Hungarian algorithm.
    The difference between Hungarian and greedy is that Hungarian finds a set with min total cost.
     """
    match = []
    unmatch = {
        'tracklets': set(range(iou.shape[0])),
        'detections': set(range(iou.shape[1]))
    }
    unmatch_tracklets = set(range(iou.shape[0]))
    unmatch_dets = set(range(iou.shape[1]))
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)

    for r, c in zip(row_ind, col_ind):
        # match.append((r, c, dist[r, c]))
        match.append((r, c))
        unmatch['tracklets'].remove(r)
        unmatch['detections'].remove(c)

    return match, unmatch

def labeling(num_GT, GT, num_Pred, Pred):
    iou_list = []
    Pred_list = []
    # print(np.bincount(GT.flatten()))
    label_list = []
    check_pred = np.zeros(num_Pred, dtype=int)
    for idx_GT in range(1, num_GT):
        GT_label = np.zeros((GT.shape[0], GT.shape[1]), dtype=int)
        GT_label[GT == idx_GT] = 1
        label_dict = {}
        label_dict['Pred'] = []

        for idx_Pred in range(1, num_Pred):
            Pred_label = np.zeros((GT.shape[0], GT.shape[1]), dtype=int)
            Pred_label[Pred == idx_Pred] = 1
            
            p_map = GT_label + Pred_label
            universe = np.bincount(p_map.flatten())
            # print("NP_WHERE:", np.argwhere(p_map == 2).shape)
            # print('Universe:', universe)
            if universe.shape[0] < 3:
                intersect = 0
                iou = 0
            else:
                intersect = universe[2]
                union = universe[1] + universe[2]
                iou = float(intersect / union)
 
            if iou > 0 and check_pred[idx_Pred] == 0:    
                label_dict['Pred'].append(idx_Pred)
                check_pred[idx_Pred] = 1

        label_list.append(label_dict)

    return label_list


'''
    清除雜訊
'''



def eliminate_noise(preds_image_t):
    Morphologic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    # dilation = cv2.dilate(preds_image_t.astype('uint8'), Morphologic_kernel)
    # erosion = cv2.erode(dilation.astype('uint8'), Morphologic_kernel)
    
    
    erosion = cv2.erode(preds_image_t.astype('uint8'), Morphologic_kernel)
    dilation = cv2.dilate(erosion.astype('uint8'), Morphologic_kernel)
   

    # erosion = erosion.astype('int64')
    
    preds_image_e = dilation
    
    return preds_image_e




    '''
    
    for idx_Pred in range(1, num_Pred):
        Pred_label = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=int)
        Pred_label[temp == idx_Pred] = 1
        _iou = []
        for idx_GT in range(1, num_GT):
            GT_label = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=int)
            GT_label[GT == idx_GT] = 1

            universe = GT_label + Pred_label
            # print('\n', np.bincount(universe.flatten()))
            intersect = np.argwhere(universe == 2).shape[0]
            union = np.argwhere(universe >= 1).shape[0] 

            iou = float(intersect / union)
            # print(iou)
            _iou.append(iou)
            # if iou > _max:
            #     _max = iou
        iou_list.append(_iou)
        Pred_list.append(Pred_label)
        # if _max > 0.5:
        #     iou_list.append(_iou)
        #     Pred_list.append(Pred_label)
    
    iou_temp = np.array(iou_list)
    iou_temp = np.resize(iou_temp, (num_Pred-1, num_GT-1))
    # print('\nIOU:', iou_temp)
    match, unmatch = hungarian(iou_temp)
    # print(num_GT-1, len(iou_temp), len(match), num_Pred-1)
    # print(match if match else 0)
    '''

    # return match, unmatch



'''
    找出每個pixel分數最高的類別
'''
def argmax_in(preds_image, carries_threshold = 0):
    # threshold = 0.5
    carries_map = copy.deepcopy(preds_image[:, :, 6])
    # carries_map = np.sqrt(carries_map)
    carries_map[carries_map > carries_threshold] =  1
    carries_map[carries_map != 1] = 0
    # carries_map[preds_image[:, :, 6] > 0.5] 
    preds_image[preds_image < 0] = 0
    preds_image_t = np.argmax(preds_image, axis=2)
    #print(preds_image_t.shape)
    # for i in range(IMG_HEIGHT):
    #     for j in range(IMG_WIDTH):
    #         if preds_image[i,j,preds_image_t[i,j]] < threshold:
    #             preds_image_t[i,j] = -1
    # preds_image_t += 1
    return preds_image_t, carries_map.astype("uint8")





'''
    從類別圖轉成RGB圖
'''
def preds2rgb(preds_image_e):
    preds_image_rgb = np.zeros((preds_image_e.shape[0], preds_image_e.shape[1], 3), dtype=np.uint8)
    # preds_image_rgb[preds_image_e==0,:] = [0, 0, 0]
    # preds_image_rgb[preds_image_e==1,:] = [34, 139, 34]
    # preds_image_rgb[preds_image_e==2,:] = [0, 255, 255]
    # preds_image_rgb[preds_image_e==3,:] = [0, 0, 255]
    # preds_image_rgb[preds_image_e==4,:] = [181, 228, 255]
    # preds_image_rgb[preds_image_e==5,:] = [128, 128, 128]

    preds_image_rgb[preds_image_e==0,:] = [0, 0, 0]
    preds_image_rgb[preds_image_e==1,:] = [34, 139, 34] # 法郎值
    preds_image_rgb[preds_image_e==2,:] = [0, 255, 255] # 牙本值
    preds_image_rgb[preds_image_e==3,:] = [0, 0, 255] # 牙隨
    preds_image_rgb[preds_image_e==4,:] = [181, 228, 255] # 人工
    preds_image_rgb[preds_image_e==5,:] = [128, 128, 128] # 齒曹骨
    preds_image_rgb[preds_image_e==6,:] = [255, 0, 0] # 蛀牙

    preds_image_rgb[preds_image_e==8,:] = [128, 0, 75] # 預測正確
    preds_image_rgb[preds_image_e==9,:] = [203, 192, 255]  # 預測失敗

    # preds_image_rgb[preds_image_e==0,:] = [0, 0, 0]
    # preds_image_rgb[preds_image_e==1,:] = [0, 255, 255] # 法郎值
    # preds_image_rgb[preds_image_e==2,:] = [0, 255, 255] # 牙本值
    # preds_image_rgb[preds_image_e==3,:] = [0, 255, 255] # 牙隨
    # preds_image_rgb[preds_image_e==4,:] = [181, 228, 255] # 人工
    # preds_image_rgb[preds_image_e==5,:] = [128, 128, 128] # 齒曹骨
    # preds_image_rgb[preds_image_e==6,:] = [0, 255, 255] # 蛀牙

    
    return preds_image_rgb


'''
    Test
'''
def pa_test(model_weight_path, jpgpattern, jpg_cross_data, lblfilepath=None, predict_path=None):
    # model = load_model(model_weight_path)
    model = load_model(model_weight_path, compile=False)

    # test_ids = [os.path.join(jpgpattern, jpg_name) for jpg_name in os.listdir(jpgpattern)]
    test_ids = [os.path.join(jpgpattern, os.listdir(jpgpattern)[i]) for i in jpg_cross_data]

    # test_ids = [os.path.join(jpgpattern, jpg_name) for jpg_name in jpg_cross_data]
    # print(test_ids)
    # testdata = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    # radiodata = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # groundtruth = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH))
    # full_teeth = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH))
    # ori_size = []
    # ori_imgs = []

    # Load test image, label
    # start = 0
    total_pred = 0 
    total_gt = 0
    FP = 0
    FN = 0
    TP = 0
    count = 0

    for n in tqdm.tqdm(range(len(test_ids))):
        img = cv2.imread(test_ids[n])
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fname, _ = os.path.splitext(os.path.basename(test_ids[n]))
        # print(test_ids[n])
        # h_, w_ = img.shape[:2]
        # ori_img = copy.deepcopy(img)
        # ori_imgs.append(ori_img)
        # ori_size.extend([[img.shape[0], img.shape[1]]])
        # if (IMG_WIDTH / h_) < (IMG_HEIGHT / w_) :
            
        #     new_width = int(w_ * float((IMG_WIDTH / h_)))
        #     # print(new_width)
        #     img = cv2.resize(img, (new_width, IMG_HEIGHT), cv2.INTER_NEAREST)
            
        # else:
        #     new_height = int(h_ * float((IMG_WIDTH / w_)))
        #     # print(new_height)
        #     img = cv2.resize(img, (IMG_WIDTH, new_height), cv2.INTER_NEAREST)
        # h_, w_ = img.shape[:2]
        # # print(h_, w_)
        # top = int((IMG_HEIGHT - h_) / 2)
        # down = int((IMG_HEIGHT - h_ + 1) / 2)
        # left = int((IMG_WIDTH - w_) / 2)
        # right = int((IMG_WIDTH - w_ + 1) / 2)
        # # print(top, down, left, right)
        # img = cv2.copyMakeBorder(img, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])
        # print(img.shape[:2])
        # img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # fname, _ = os.path.splitext(os.path.basename(test_ids[n]))
        # radiodata[n] = img
        
        keypoint = []

        if(lblfilepath):
            print(fname)
            if fname == 'P0124318_17_45_down' or fname == 'P0124318_17_45_up':
                continue
            # if fname != "23399866_20191118_17_down":    continue

            # npypath = os.path.join(lblfilepath, fname) + '.npy'
            pklpath = os.path.join(lblfilepath, fname) + '.pkl'     # Ground Truth

            # dictpath = os.path.join('./Data/Iso_maskrcnn/cut_label', fname) + '.pkl'  # Maskrcnn Path
            # dictpath = os.path.join('./Data/Iso_maskrcnn/label', fname) + '_*.pkl'  # Maskrcnn Path
            dictpath = os.path.join('./Data/Iso_maskrcnn/maskrcnn_label', fname) + '.pkl'  # Maskrcnn Path
            
            
            isopath = glob.glob(f'./Data/Iso_img/{fname}_*.JPG')
            # print(glob.glob(dictpath))
            

            #讀GT
            filedata = open(pklpath, 'rb')
            pkldata = pickle.load(filedata)
            filedata.close()
            label = proMask_produce(pkldata)
            pred_label = np.zeros((label.shape[0], label.shape[1]), dtype=int)



            # h, w = pkldata['imgsize']
            # label = pkldata['label'].astype(np.uint8)
            #讀全齒與全齒bbox
            try:
                # print(dictpath)
                
                fullteeths= retrieve_mask(dictpath)


                # dictfile = open(dictpath, 'rb')
                # dictdata = pickle.load(dictfile)
                # print(dictdata)
                # dictfile.close()
                a = 1 / len(fullteeths)
                testdata = np.zeros((len(fullteeths), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                full_teeth = np.zeros((len(fullteeths), IMG_HEIGHT, IMG_WIDTH))
                iso_img_array = np.zeros((len(fullteeths), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                bboxes = np.zeros((len(fullteeths), 4), dtype=int)
                # continue

            except:         # 假設maskrcnn沒讀到牙齒，則只計算GT的數量
                print('Zero Prediction!!!!')
                # start += 1
                print(f'{fname}\n')
                num_GT, label_GT = retrieve_carries(label, carries_threshold=0)
                num_GT -= 1
                total_gt += num_GT
                gt_img = copy.deepcopy(img)
                gt_img[label == 6, :] = [255, 0, 0]
                pred_img = preds2rgb(pred_label)
                compare = np.concatenate([gt_img, pred_img, pred_img], axis=1)
                cv2.imwrite(predict_path+ fname + '_p.jpg', compare)
                continue

            # bboxes = dictdata['bbox']
            # fullteethes = dictdata['fullteeth']
            # # print('\n', len(bboxes), len(fullteethes))


            

            for idx, fullteeth in enumerate(fullteeths):
    


                x_min, y_min, x_max, y_max = retrieve_bbox(fullteeth)
                bboxes[idx] = [x_min, y_min, x_max, y_max]
                fullteeth = fullteeth[y_min:y_max, x_min:x_max].astype('uint8')
      
                radiograph = gray_img[y_min:y_max, x_min:x_max]
                iso_img = img[y_min:y_max, x_min:x_max]
                # label = np.load(npypath).astype(np.uint8)
                h_, w_ = radiograph.shape[:2]  
                if (IMG_WIDTH / h_) < (IMG_HEIGHT / w_) :
                    
                    new_width = int(w_ * float((IMG_WIDTH / h_)))
                    # print(new_width)
                    radiograph = cv2.resize(radiograph, (new_width, IMG_HEIGHT), cv2.INTER_NEAREST)
                    fullteeth = cv2.resize(fullteeth, (new_width, IMG_HEIGHT), cv2.INTER_NEAREST)
                    iso_img = cv2.resize(iso_img, (new_width, IMG_HEIGHT), cv2.INTER_NEAREST)
                else:
                    new_height = int(h_ * float((IMG_WIDTH / w_)))
                    # print(new_height)
                    radiograph = cv2.resize(radiograph, (IMG_WIDTH, new_height), cv2.INTER_NEAREST)
                    fullteeth = cv2.resize(fullteeth, (IMG_WIDTH, new_height), cv2.INTER_NEAREST)
                    iso_img = cv2.resize(iso_img, (IMG_WIDTH, new_height), cv2.INTER_NEAREST)
                

                h_, w_ = radiograph.shape[:2] 
                top = int((IMG_HEIGHT - h_) / 2)
                down = int((IMG_HEIGHT - h_ + 1) / 2)
                left = int((IMG_WIDTH - w_) / 2)
                right = int((IMG_WIDTH - w_ + 1) / 2)
                # print(top, down, left, right)
                temp = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                temp_iso = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                temp_fullteeth = np.zeros((IMG_HEIGHT, IMG_WIDTH))
                temp[top:top+h_, left:left+w_] = radiograph
                temp_iso[top:top+h_, left:left+w_] = iso_img

                temp_fullteeth[top:top+h_, left:left+w_] = fullteeth
                keypoint.append([top, left, h_, w_])
            # label = cv2.copyMakeBorder(label, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0])
            # label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                testdata[idx] = temp
                full_teeth[idx] = temp_fullteeth
                iso_img_array[idx] = temp_iso

        """
            預測
        """
        preds_test_flag = False
        end_flag = False
        for i in range(len(testdata)):
            if((i+1)*10 < len(testdata)):
                preds = model.predict(testdata[i*10:(i+1)*10], verbose=1)
            else:
                preds = model.predict(testdata[i*10:len(testdata)], verbose=1)
                end_flag = True

            if(not preds_test_flag):
                preds_test = preds
                preds_test_flag = True
            else:
                preds_test = np.concatenate((preds_test, preds),axis=0)
                # print(preds_test.shape)
            if(end_flag):
                break
        
        ## 儲存全部Test裡原大小的 Pixel classfication 預測圖

        for ix in tqdm.tqdm(range(len(preds_test))):
            count += 1
            preds_image = preds_test[ix]
            # print(preds_image.shape)
            radio = testdata[ix]
            fullteeth = full_teeth[ix]
            preds_image_t, carries_map = argmax_in(preds_image, 0.5)

            carries_map = eliminate_noise(carries_map)
            '''
            
            ### ----  假設一個sliding window裡有兩塊預測,則選取較大面積的預測-------


            radio[fullteeth == 0] = 0
            # erosion = cv2.erode(radio, np.ones((5, 5), dtype=np.uint8), iterations=1)
            # edge = radio - erosion
            edge = cv2.Canny(radio, 70, 210)

            n = np.argwhere(edge == 255).shape[0]
            _range = 24
            top, left, h_, w_ = keypoint[ix]
            x_min, y_min, x_max, y_max = bboxes[ix]
            for i in range(9):
                for j in range(9):
                    start_x = 12 + i*25
                    start_y = 12 + j*25
                    carries_temp = carries_map[start_x-_range//2:start_x+_range//2, start_y-_range//2:start_y+_range//2 ]
                    _num, _label, _stats, _centroids = cv2.connectedComponentsWithStats(carries_temp, connectivity=4)
                    if _num > 2:
                        before = copy.deepcopy(carries_map)
                        print(_stats)
                        print(_stats[:, -1])
                        maxarea = np.argmax(_stats[1:, -1]) + 1 
                        print("\nargmax:", maxarea)
                        carries_temp[_label != maxarea] = 0
                        carries_map[start_x-_range//2:start_x+_range//2, start_y-_range//2:start_y+_range//2 ] = carries_temp
                    
                        # cv2.imshow('',np.hstack((preds2rgb(before), preds2rgb(carries_map))))
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
            '''
            
            # for i in range(n):
            #     start_x, start_y = np.argwhere(edge == 255)[i, :]
            # print(np.argwhere(edge == 255))
            # start_x, start_y = np.argwhere(edge == 255)[0, :]
            



                # carries_temp = carries_map[top:top+h_, left:left+w_]
                # radio_temp = radio[top:top+h_, left:left+w_]
                # edge_temp = edge[top:top+h_, left:left+w_]
                
                
                # carries_range = carries_map[start_x-_range:start_x+_range, start_y-_range:start_y+_range]
                # # print(np.bincount(carries_range.flatten()))
                
                # edge_range = edge[start_x-_range:start_x+_range, start_y-_range:start_y+_range]
                # _num, _label, _stats, _centroids = cv2.connectedComponentsWithStats(carries_range, connectivity=4)
            
                # if _num > 1:
                #     maxarea = np.argmax(_stats[-1, 1:])
                #     print("\nargmax:", maxarea)
                #     cv2.imshow('', preds2rgb(carries_range))
                #     cv2.waitKey()
                #     cv2.destroyAllWindows()
                #     _label[_label != maxarea] = 0

                    # exit()


            # print(np.bincount(edge.flatten()))
            # print(np.argwhere(edge == 255).shape)
            # cv2.imshow('', preds2rgb(preds_image_t))
            # cv2.imshow('', np.hstack((radio, edge)))


            
            # carries_map = eliminate_noise(carries_map)
            # a = np.hstack((preds2rgb(carries_map), preds2rgb(eliminate_noise(carries_map))))
            # cv2.imwrite('./Predict_result_iso_cv/{}.jpg'.format(count), a)

            # cv2.imshow('', np.hstack((preds2rgb(carries_map), preds2rgb(eliminate_noise(carries_map)))))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            

            # print(np.bincount(preds_image_t.flatten()))
            preds_image_e = preds_image_t
            carries_map[radio < 25] = 0
        # gt_img = copy.deepcopy(groundtruth[ix].astype(np.uint8))
        # gt_img[gt_img != 6] = 0

        # num_GT, label_GT, stats_GT, centroids_GT = cv2.connectedComponentsWithStats(gt_img, connectivity=4)
        # num_GT, label_GT = retrieve_carries(gt_img, fullteeth=fullteeth)

            num_carries, label_carries = retrieve_carries(carries_map, fullteeth=fullteeth, carries=1, carries_threshold=0)
            
            
            
            num_Pred, label_Pred = num_carries, copy.deepcopy(label_carries)
        # print("Carries:", np.bincount(label_Pred.flatten()))
        # print("GT:", np.bincount(label_GT.flatten()))
        # num_Pred, label_Pred = retrieve_carries(preds_image_e, 0)
        # match, unmatch = labeling(num_GT, label_GT, num_Pred, label_Pred)
            iso_img = iso_img_array[ix]
            preds_image_e[preds_image_e == 6] = 0
            preds_image_e[label_Pred != 0] = 6
            # print(radio)
            # print(radio.shape)
            iso_img[label_Pred != 0] = [42, 42, 128]


            # cv2.imwrite(f'./123/{fname}_{ix}.jpg', iso_img)

            # 預測完拼回原圖
            top, left, h_, w_ = keypoint[ix]
            x_min, y_min, x_max, y_max = bboxes[ix]
            temp = np.zeros((h_, w_), dtype=np.uint8)
            temp = preds_image_e[top:top+h_, left:left+w_]
            temp = cv2.resize(temp, (x_max-x_min, y_max-y_min), interpolation=cv2.INTER_NEAREST)
            print(x_min, x_max, y_min, y_max)
            
            # cv2.imshow("", temp)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            temp_temp = np.zeros((label.shape[0], label.shape[1]), dtype=int)
            temp_temp[y_min:y_max, x_min:x_max] = temp
            pred_label[temp_temp != 0] = temp_temp[temp_temp != 0]

            # pred_label[y_min:y_max, x_min:x_max] = temp
        

        # 整張BW做評估
        
        num_Pred, label_Pred = retrieve_carries(pred_label, carries_threshold=0)
        num_GT, label_GT = retrieve_carries(label, carries_threshold=0)
        # gt_img[gt_img == 6] = 0
        # gt_img[label_GT != 0] = 6
        match_list = labeling(num_GT, label_GT, num_Pred, label_Pred)

        num_Pred -= 1
        num_GT -= 1
        print("Num_pred:", num_Pred, " Num_GT:", num_GT)

        total_pred += num_Pred
        total_gt += num_GT
       
        """
            match: [idx_pred]  
        """
        print("match:", match_list)
        for idx_GT, match in enumerate(match_list):
            Pred_list = match['Pred']
            
            if len(Pred_list) != 0:
                TP += 1
                for idx_pred in Pred_list:
                    print(idx_pred)
                    pred_label[label_Pred == idx_pred] = 8

                total_pred -= (len(Pred_list) - 1)
                num_Pred -= (len(Pred_list) - 1)
        
        print("Num_pred:", num_Pred, " Num_GT:", num_GT)
        """
        ### 舊評估(hugarian algorithm)
        for _match in match_list:
            print("match:", _match)
            pred_idx, gt_idx = _match
            pred_idx += 1
            gt_idx += 1
            # print(pred_idx, gt_idx)
            # gt = copy.deepcopy(label_GT)
            gt = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=int)
            gt[label_GT == gt_idx] = 1
            # gt[label_GT != gt_idx] = 0
            # gt = label_GT[label_GT == gt_idx]
            # print(gt.shape)
            # gt[gt != 0] = 1
            # pred = copy.deepcopy(label_Pred)
            pred = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=int)
            pred[label_Pred == pred_idx] = 1
            # pred[label_Pred != pred_idx] = 0
            # pred = label_Pred[label_Pred == pred_idx]
            # print(pred.shape)
            # pred[pred != 0] = 1
            iou = IOU(pred, gt)
            print("IOU:", iou)
            if iou > 0:
                # y_true.append(0)
                TP += 1
                preds_image_e[pred == 1] = 8
            else:
                # y_true.append(1)
                FP += 1
                # y_predict.append(iou)
            # if num_Pred > num_GT:
            #     FP += (num_Pred - num_GT)
            # if num_GT > num_Pred:
            #     FN += (num_GT - num_Pred)
        """
        # print(pred_label.shape)
        # print(np.bincount(pred_label.flatten()))
        preds_image_rgb = preds2rgb(pred_label)
        pred_img = copy.deepcopy(img)
        pred_img[pred_label == 8, :] = [128, 0, 75]
        preds_image_rgb[pred_label == 8, :] = [128, 0, 75]
        pred_img[pred_label == 6] = [42, 42, 128]

        # preds_image_rgb[pred_label == 6] = [42, 42, 128]
        # preds_image_rgb[carries_map == 1] = [42, 42, 128]
        GT_rgb = preds2rgb(label)
        gt_img = copy.deepcopy(img)
        gt_img[label == 6, :] = [255, 0, 0]
        # preds_image_rgb = cv2.resize(preds_image_rgb, (ori_size[start][1], ori_size[start][0]), interpolation=cv2.INTER_NEAREST)

        # rgb_predict = preds_image_rgb
        rgb_name = os.path.splitext(os.path.basename(test_ids[n]))[0]
        print(rgb_name, '\n')

        compare = np.concatenate([gt_img, pred_img, preds_image_rgb], axis=1)

        if predict_path:

            # cv2.imwrite(predict_path+ rgb_name + '_p_gt{}_pred{}.jpg'.format(num_GT, num_Pred), compare)
            cv2.imwrite(predict_path+ rgb_name + '_p.jpg', compare)

        # start += 1

    print(TP, total_gt, total_pred)
    # print(TP, FP, total_pred, total_gt)
    if TP == 0:
        precision = 0  
        recall = 0 
    else:
        precision = TP / total_pred
        recall = TP / total_gt
    print('Precision:', precision)
    print('Recall:', recall, '\n')
    print('F1_score:', 2*(precision*recall)/(precision+recall))
    print("Num Teeth:", count)
    return TP, total_gt, total_pred, count
#In[]
# filedata = open('./weights/cv/cut_cross_data.pkl', 'rb')
# cross_dictData = pickle.load(filedata)
# filedata.close()
# len(cross_dictData[0])

#In[]

if __name__ == '__main__':
    jpgpattern = './Data/CutImage'
    lblfilepath = './Data/CutImage_label'        # GT label Path
    model_weight_path = './weights/cv_alldata/100epochs_iso_baseline_multifocalloss_maskrcnn/unet_BW_cut_cross0.h5'
    # model_weight_path = './weights/cv_alldata/unet_BW_cross0.h5'
    
    predict_path = './Predict_result_iso_cv_alldata/maskrcnn/'
    # predict_path = './Predict_result_iso_cv_alldata/maskrcnn/100epochs_aug_weightedloss/'
    # predict_path = './Predict_result_iso_cv_alldata/maskrcnn/'
    cross_data_path = './weights/cv_alldata/N8_cross_data.pkl'

    filedata = open(cross_data_path, 'rb')
    _, val_idx = pickle.load(filedata)
    # filedata.close()
    # val_idx = cross_dictData['val_idx']

    sum_TP = 0
    sum_GT = 0
    sum_Pred = 0
    sum_teeth = 0

    for i in range(0, 4):
        print("{}_fold:".format(i))
        model_weight_path = model_weight_path[:-4] + '{}.h5'.format(i)
        # jpg_cross_data = [os.path.join(jpgpattern, idx) for idx in val_idx[i]]
        jpg_cross_data = val_idx[i]
        # print(jpg_cross_data)
        print("123:::::::::::", len(jpg_cross_data))
        TP, total_GT, total_Pred, count = pa_test(model_weight_path, jpgpattern, jpg_cross_data, lblfilepath, predict_path=predict_path)
        sum_TP += TP
        sum_GT += total_GT
        sum_Pred += total_Pred
        sum_teeth += count
    print("-------------Summarize-------------")
    print("TP: ", sum_TP," GT: ", sum_GT, " Pred: ", sum_Pred)

    if sum_TP == 0:
        precision = 0 
        recall = 0 
    else:
        precision = sum_TP / sum_Pred
        recall = sum_TP / sum_GT
    print('Num_teeth:', sum_teeth)
    print('Precision:', precision)
    print('Recall:', recall, '\n')
    print('F1_score:', 2*(precision*recall)/(precision+recall))