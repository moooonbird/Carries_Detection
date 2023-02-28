#In[]
from copy import deepcopy
import glob
import os
import shutil
import cv2
import pickle
import numpy as np
import tqdm 
import random
import copy
from skimage.io import imread, imshow

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow as tf
import label_processing_iso_maskrcnn as img_aug

#In[]

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

category_num = 7

cutfilepath = './Data/CutImage/'
jpgfilepath = './Data/Iso_img/'
npyfilepath = './Data/Iso_label/'

select_jpgfilepath = './Data/train_BW/image'
select_npyfilepath = './Data/train_BW/label'

model_path = './weights/cv_all_data/train_214/unet_BW_cross0.h5'
save_split_record_path = './weights/cv_all_data/BW_panoptic_cross_data_214.pkl'

batch_size = 16
epochs = 100



#In[]
def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

'''
    增加蛀牙的loss(Weights_aware_categorical_crossentropy loss)
'''
def weights_categorical_crossentropy_loss(y_true, y_pred):
    weight = 20
    wbce = K.categorical_crossentropy(y_true, y_pred)
    if(y_true.shape[0]):
        weight_map = np.ones((y_true.shape[:-1]), dtype=np.float32)
        weight_map[weight_map[...,-1]==1] = weight
        weight_map = tf.convert_to_tensor(weight_map)
        wbce = wbce * weight_map

    return wbce

def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    # alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[1],[1],[1],[1],[1],[1],[2]], dtype=tf.float32)

    #alpha = tf.constant_initializer(alpha)

    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        print(alpha)
        # print('123')
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed

def preds2rgb(preds_image_e):
    preds_image_rgb = np.zeros((preds_image_e.shape[0], preds_image_e.shape[1], 3), dtype=np.uint8)
    # preds_image_rgb[preds_image_e==0,:] = [0, 0, 0]
    # preds_image_rgb[preds_image_e==1,:] = [34, 139, 34]
    # preds_image_rgb[preds_image_e==2,:] = [255, 255, 0]
    # preds_image_rgb[preds_image_e==3,:] = [255, 0, 0]
    # preds_image_rgb[preds_image_e==4,:] = [255, 228, 181]
    # preds_image_rgb[preds_image_e==5,:] = [128, 128, 128]
    # preds_image_rgb[preds_image_e==6,:] = [255, 255, 255]
    preds_image_rgb[preds_image_e==0,:] = [0, 0, 0] # 背景
    preds_image_rgb[preds_image_e==1,:] = [34, 139, 34] # 法郎值
    preds_image_rgb[preds_image_e==2,:] = [0, 255, 255] # 牙本質
    preds_image_rgb[preds_image_e==3,:] = [0, 0, 255] # 牙隨
    preds_image_rgb[preds_image_e==4,:] = [181, 228, 255] # 人工
    preds_image_rgb[preds_image_e==5,:] = [128, 128, 128] # 齒曹骨
    preds_image_rgb[preds_image_e==6,:] = [255, 0, 0] # 蛀牙
    
    # cv2.imshow('', preds_image_rgb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return preds_image_rgb

#In[]
'''
    Load image,label and make the groundtruth
'''
cross_num = 4
allList = os.listdir(cutfilepath)
idx = list(np.arange(len(allList)))
random.seed(0)
random.shuffle(idx)
split_idx = split(idx, cross_num)
# print("123:::::::", len(split_idx[0]))

# output = open(save_split_record_path, 'wb')
# pickle.dump([allList, split_idx], output)
# output.close()

_input = open(save_split_record_path, 'rb')
split_idx = pickle.load(_input)
# print(split_idx)

for cross_i in range(0, 4):
    # tmp_idx = copy.deepcopy(split_idx)
    # tmp_idx.pop(cross_i)
    # print(tmp_idx)

    # train_idx = []
    # for i in tmp_idx:
    #     train_idx = train_idx + i
    # # print(len(train_idx))
    # ori_train_ids = [allList[i] for i in train_idx]
    ori_train_ids = split_idx['train_idx'][cross_i]
    # print(ori_train_ids)
    
    
    ## Data Augmentation (先省略) ##
    
    if os.path.isdir(select_jpgfilepath):
        shutil.rmtree(select_jpgfilepath)
        shutil.rmtree(select_npyfilepath)
    os.mkdir(select_jpgfilepath)        
    os.mkdir(select_npyfilepath) 

    count = 0
    # print(ori_train_ids)
    # print(len(ori_train_ids))
    
    # Read data
    image_list = []
    for n in range(len(ori_train_ids)):
        fname, ext = os.path.splitext(ori_train_ids[n])
        image_path = os.path.join(jpgfilepath, fname +'*.JPG')
        # print(image_path)
        # print(glob.glob(image_path))
        image_list += glob.glob(image_path)

    # print(len(image_list))    
    print(image_list)
    
    # data augmentation
    for n in tqdm.tqdm(range(len(image_list)), ncols=10):
        fname, ext = os.path.splitext(os.path.split(image_list[n])[1])
        # image_path = os.path.join(jpgfilepath, fname +'*.jpg')
        image_path = image_list[n]
        # print(glob.glob(image_path))
        # print(len(glob.glob(image_path)))
        
        label_path = os.path.join(npyfilepath, fname +'.pkl')
        # print(label_path)
        # 建立數據擴增資料夾，將擴增資料存入資料夾中
        img_aug.produce_image(image_path, label_path, select_jpgfilepath, select_npyfilepath)
        count += 1
        print('finish: %d / %d'%(count, len(image_list)), end='\r')

    train_ids = os.listdir(select_jpgfilepath)

    jpgdata = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    npydata = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, category_num), dtype=np.uint8)

    for n in tqdm.tqdm(range(len(train_ids))):
        img = cv2.imread(os.path.join(select_jpgfilepath, train_ids[n]))
        fname, ext = os.path.splitext(train_ids[n])
        npypath = os.path.join(select_npyfilepath, fname + '.npy')
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = np.load(npypath)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, category_num), dtype=np.uint8)

        for i in range(category_num):
            mask[label == i, i] = 1
        jpgdata[n] = gimg
        npydata[n] = mask
    print(jpgdata.shape[0])
    print(npydata.shape[0])

    '''
        Unet Model
    '''

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)

    # outputs = Conv2D(category_num, (1, 1), activation='sigmoid') (c9)
    outputs = Conv2D(category_num, (1, 1), activation='softmax') (c9)

    '''
        Build Model
    '''
    # ## pretrain model : train BW image ###
    # model = load_model(pretrain_model_path, custom_objects={'weights_categorical_crossentropy_loss': weights_categorical_crossentropy_loss})
    # model.compile(optimizer='adam', loss=weights_categorical_crossentropy_loss, metrics=['accuracy'])
    
    
    # ### no pretrain model
    model = Model(inputs = [inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=weights_categorical_crossentropy_loss, metrics=['accuracy'])
    model.compile(optimizer='adam', loss=[multi_category_focal_loss1(alpha=[1,1,1,1,1,1,2])], metrics=['accuracy'])

    model.summary()

    '''
        Train Model
    '''
    # earlystopper = EarlyStopping(patience=20, verbose=1)
    model_path = model_path[:-4] + '{}.h5'.format(cross_i)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    results = model.fit(jpgdata, npydata, validation_split=0.1, batch_size=batch_size, epochs=epochs, 
                        callbacks=[checkpointer])
#In[]
# x = np.arange(1, 49)
# split(x, 4)