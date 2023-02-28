import numpy as np
import pickle
import cv2
import math
import random
import LabelData
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os
import copy
import matplotlib.pyplot as plt

'''
    將標記的data轉成mask(groundtruth)
'''
def Mask_produce(dictData):
    
    size = dictData['MapPixel2RegionId'].shape
    #print(size)
    
    for k in range(len(dictData['ListRegion'])):
        for i in range(size[0]):
            for j in range(size[1]):
                if dictData['MapPixel2RegionId'][i][j] in dictData['ListRegion'][k].listRegionId:
                    dictData['MapPixel2RegionId'][i][j] = -1 * int(dictData['ListRegion'][k].idROI)
                elif k== len(dictData['ListRegion'])-1 and dictData['MapPixel2RegionId'][i][j] > 0:
                    dictData['MapPixel2RegionId'][i][j] = 0
    
    dictData['MapPixel2RegionId'] = np.abs(dictData['MapPixel2RegionId'])
    
    """
    print(dictData['MapPixel2RegionId'])
    mask = np.array(dictData['MapPixel2RegionId'], dtype=np.uint8)
    cv2.imshow('My Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print(dictData['ListRegion'][0].idROI)
    #print(dictData['ListRegion'][0].listRegionId)
    """
    
    return dictData['MapPixel2RegionId']

def proMask_produce(dictData):
    
    mask = copy.deepcopy(dictData['MapPixel2RegionId'])
    # print(dictData['ListRegion'][1].idROI)
    for k in range(len(dictData['ListRegion'])):
        for id in dictData['ListRegion'][k].listRegionId:

            mask[mask==id] = -1 * int(dictData['ListRegion'][k].idROI)


    mask[mask > 0] = 0
    mask = np.abs(mask)

    return mask

'''
    Draw grid lines on the image
'''
def draw_grid(im, grid_size):
    
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


'''
    1.彈性形變
'''
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image[:,:,:-1] = cv2.warpAffine(image[:,:,:-1], M, shape_size[::-1],flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
    image[:,:,-1] = cv2.warpAffine(image[:,:,-1], M, shape_size[::-1],flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    
# 去除黑邊
crop = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

'''
    2.隨機裁切圖片
'''
def crop_image(img):
    random_seed = random.randint(10,30)
    seed_w = random.randint(0,random_seed)
    w = img.shape[0]-(random_seed-seed_w)-1
    seed_h = random.randint(0,random_seed)
    h = img.shape[1]-(random_seed-seed_h)-1
    #print(seed_h, seed_w, h, w)
    img_cropped = crop(img, seed_w, seed_h, w, h)
    
    return img_cropped
    
'''
    3.旋轉圖片，並決定是否要裁切
'''
def rotate_image(img, crop_flag):
    """
    angle: 旋轉的角度
    crop_flag: 是否需要進行裁減，bool值
    """
    key = random.randint(0,1)
    if key == 0:
        angle = random.randint(0,15)
    else:
        angle = random.randint(345,360)
    h, w = img.shape[:2]
    # 旋轉角度的周期是360°
    angle %= 360
    # 計算訪設變換矩陣
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋轉後的圖
    img[:,:,:-1] = cv2.warpAffine(img[:,:,:-1], M_rotation, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
    img[:,:,-1] = cv2.warpAffine(img[:,:,-1], M_rotation, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    img_rotated = img
    
    # 如果需要去除黑邊
    if crop_flag:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 轉化角度為弧度
        theta = angle_crop * np.pi / 180
        # 計算高寬比
        hw_ratio = float(h) / float(w)
        # 計算裁減邊長係數的分子項
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # 計算分母中和高寬比相關的項
        r = hw_ratio if h > w else 1 / hw_ratio
        # 計算分母項
        denominator = r * tan_theta + 1
        # 最終的邊長係數
        crop_mult = numerator / denominator

        # 得到裁減區域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        
        img_rotated = crop(img_rotated, x0, y0, w_crop, h_crop)
        
    return img_rotated

'''
    4.增加亮度
'''
def intensity_adjust(img):
    #Alpha = 1.2
    #Beta = 5
    
    Alpha = round(random.uniform(1, 1.3), 2)
    Beta = random.randint(0,10)
    img[:,:,0:-1] = np.uint8(np.clip((Alpha * img[:,:,0:-1] + Beta), 0, 255))
    
    return img



'''
    5.增加對比
'''
def contrast_adjust(img, brightness=0, contrast=90):
    import math

    # brightness = 0
    # contrast = -100 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return img
'''
    數據擴增
    (方法:隨機挑2~4種方法來組合成一張新的照片)
'''
def produce_image(image_path, label_path, outimfilepath, outlafilepath):
    
    im_size = 256
    # distort_num:擴增數量
    distort_num = 1   #Aug:3
    rdom_list = [3,4]
    rdom_num = 1
    
    # image
    img = cv2.imread(image_path)
    h_, w_ = img.shape[:2]
    if (im_size / h_) < (im_size / w_) :
            
        new_width = int(w_ * float((im_size / h_)))
        # print(new_width)
        img = cv2.resize(img, (new_width, im_size), cv2.INTER_NEAREST)
            
    else:
        new_height = int(h_ * float((im_size / w_)))
        # print(new_height)
        img = cv2.resize(img, (im_size, new_height), cv2.INTER_NEAREST)
        
    h_, w_ = img.shape[:2]
    top = int((im_size - h_) / 2)
    down = int((im_size - h_ + 1) / 2)
    left = int((im_size - w_) / 2)
    right = int((im_size - w_ + 1) / 2)
    # print(top, down, left, right)
    ori_image = cv2.copyMakeBorder(img, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])
    # ori_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        
        

        # label = cv2.resize(label, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

        # preds2rgb(label)

    # ori_image = cv2.resize(ori_image.astype(np.float32), (im_size, im_size), interpolation=cv2.INTER_NEAREST)
    #fname, ext = os.path.splitext(image)
    #cv2.imwrite('./Train/ori0.jpg', ori_image)

    # mask(npy)
    filepath = open(label_path, 'rb')
    filedata = pickle.load(filepath)
    label = filedata['label'].astype(np.uint8)
    # print(label.shape)
    # label = np.load(label_path).astype(np.uint8)
    
    h_, w_ = label.shape[:2]
    if (im_size / h_) < (im_size / w_) :
                
        new_width = int(w_ * float((im_size / h_)))
        # print(new_width)
        label = cv2.resize(label, (new_width, im_size), cv2.INTER_NEAREST)
                
    else:
        new_height = int(h_ * float((im_size / w_)))
        # print(new_height)
        label = cv2.resize(label, (im_size, new_height), cv2.INTER_NEAREST)

    h_, w_ = label.shape[:2]
    top = int((im_size - h_) / 2)
    down = int((im_size - h_ + 1) / 2)
    left = int((im_size - w_) / 2)
    right = int((im_size - w_ + 1) / 2)

    temp = np.zeros((im_size, im_size))
    temp[top:top+h_, left:left+w_] = label



    
    ori_mask = copy.deepcopy(temp)
    # print(ori_mask.shape)
    # ori_mask = cv2.resize(ori_mask, (im_size, im_size), interpolation=cv2.INTER_NEAREST)


    """
    # mask (pkl)
    fileData = open(label_path, "rb")
    dictData = pickle.load(fileData)
    fileData.close()
    
    dictData['MapPixel2RegionId'] = cv2.resize(dictData['MapPixel2RegionId'].astype(np.float32), (im_size, im_size), interpolation=cv2.INTER_NEAREST)
    ori_mask = proMask_produce(dictData)
    
    cv2.imshow('My Image', ori_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(ori_image.shape)
    print(ori_mask.shape)
    """

    flip_name = ['ori', 'hf']
    _, fullflname = os.path.split(image_path)
    fname,_ = os.path.splitext(fullflname)
    im_segname = os.path.join(outimfilepath, fname) + '_'
    la_segname = os.path.join(outlafilepath, fname) + '_'
    
    for i in range(1):  #Aug:2
        # Original
        if i==0:
            pro_image = ori_image
            pro_mask = ori_mask
            
        # Flipped Horizontally 水平翻轉
        elif i == 1:
            pro_image = cv2.flip(ori_image, 1)
            pro_mask = cv2.flip(ori_mask, 1)
        
        
        cv2.imwrite(im_segname + flip_name[i] + str(0) + '.jpg', pro_image)
        np.save(la_segname + flip_name[i] + str(0) + '.npy', pro_mask)
        
            
        for j in range(1,distort_num):
            
            mix = np.concatenate((pro_image, pro_mask[:,:,np.newaxis]), axis = 2)
            
            random.shuffle(rdom_list)
            # rdom_num = random.randint(1,2)
            # for k in range(len(rdom_list)):
            for k in range(2):
                if rdom_list[k] == 1:
                    mix = elastic_transform(mix, im_size * 2, im_size * 0.08, im_size * 0.08)
                elif rdom_list[k] == 2:
                    mix = crop_image(mix)
                elif rdom_list[k] == 3:
                    mix = rotate_image(mix, True)
                elif rdom_list[k] == 4:
                    mix = intensity_adjust(mix)
                elif rdom_list[k] == 5:
                    mix = contrast_adjust(mix)
                mix = cv2.resize(mix, (im_size, im_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(im_segname + flip_name[i] + str(j) + '.jpg', mix[:,:,0:-1])
            np.save(la_segname + flip_name[i] + str(j) + '.npy', np.squeeze(mix[:,:,-1]))
                

    return True

'''
    注意: 這邊的label檔是原始的pkl檔
'''
if __name__ == '__main__':
    
    # imfilepath = './dataset/Ori_image_BW'
    # lafilepath = './dataset/Ori_label_BW'
    # outimfilepath = './dataset/train_image_BW'
    # outlafilepath = './dataset/train_label_BW'

    # imfilepath = './dataset/non_cut_BW/image'
    # lafilepath = './datasenon_cut_t/non_cut_BW/label'
    # outimfilepath = './dataset/train_BW/image'
    # outlafilepath = './dataset/train_BW/label'
    
    imfilepath = './Data/Iso_img/04561074_20190502_27_0.JPG'
    lafilepath = './Data/Iso_label/04561074_20190502_27_0.npy'
    outimfilepath = './Data/train_BW/image'
    outlafilepath = './Data/train_BW/label'
    produce_image(imfilepath, lafilepath, outimfilepath, outlafilepath)

    # allList = os.listdir(imfilepath)
    # count = 0
    # img = cv2.imread("./Data/Ori_img/P0008389_14.JPG")
    # contrast_adjust(img)
    # for image in allList:
    #     fname, ext = os.path.splitext(image)
    #     image_path = os.path.join(imfilepath, image)
    #     label_path = os.path.join(lafilepath, fname) + '.pkl'
    #     produce_image(image_path, label_path, outimfilepath, outlafilepath)
    #     count += 1
    #     print('finish: %d / %d'%(count, len(allList)), end='\r')
        
        