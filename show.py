
# coding=gbk
import numpy as np
import os 
import nibabel as nib
import imageio 
import random
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import math




def gaussian_noise(img, mean, sigma):
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out, noise

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def raffin(img):
    ima_raffin = img
    return ima_raffin

def strength_1(img):
    seq = iaa.Sequential([
        #裁剪
        iaa.Crop(px=(50,100),keep_size=False),
        #50%的概率水平翻转
        iaa.Fliplr(0.5),
        #50%的概率垂直翻转
        iaa.Flipud(0.5),
        #高斯模糊,使用高斯核的sigma取值范围在(0,3)之间
        #sigma的随机取值服从均匀分布
        iaa.GaussianBlur(sigma=(0,3.0))
    ])
    #可以内置的quokka图片,设置加载图片的大小
    # example_img = ia.quokka(size=(224,224))
    #这里我们使用自己的图片
    example_img = img
    
    aug_example_img = seq.augment_image(image=example_img)
    return aug_example_img

def transport_log(img):
    output = np.zeros(img.shape,np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = 0 + math.log(img[i,j] + 1)/(0.06 * math.log(2))
    return output

def transport_linear(img):
    output = np.zeros(img.shape,np.uint8)
    img = (img - 42) * (255 - 0) / (232 - 42)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 0:
                img[i, j] = 0
            elif img[i, j] > 255:
                img[i, j] = 255
    return img

def transport_para_linear(img):
    for i in range(img.shape[0]):
      for j in range(0, img.shape[1]):
        if 0 <= img[i, j] < 100:
            img[i, j] = 0 + img[i, j] * (50 / 100)
        elif 100 <= img[i, j] < 200:
            img[i, j] = 50 + ( img[i, j] - 100 ) * (200 - 50) / (200 - 100)
        elif 200 <= img[i,j] <255:
            img[i,j] = 200 + ( img[i,j] - 200 )* (255 - 200) / (255 - 200)
 
    for i in range(0, img.shape[0]):
      for j in range(0, img.shape[1]):
        if img[i,j] <= 0 :
            img[i,j] = 0
        elif img[i,j] >= 255:
            img[i,j] = 255
    return img


def transform_to_list(img):
    list1 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            list1[i][j] = img[i][j]
    return list1

def enhance_contrast(image):
    gray = image
    gray = cv2.convertScaleAbs(gray)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 对灰度图像应用CLAHE算法
    enhanced = clahe.apply(gray)
    
    return enhanced



def normalize_image(image):
    """
    将图像归一化到0到1之间。
 :param image: 输入图像，可以是任何形状的numpy数组。
    :return: 归一化后的图像。
    """
    # 将图像换为浮点数类型
    image = image.astype(np.float32)
    # 将像素值缩放到0到1之间
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def nii_to_image(niifile):
    filenames = os.listdir(filepath)  #读取nii文件
    slice_trans = []
    result = []
    arr_3d = np.random.rand(3240, 135, 189)
    arr_3d_index = 0
    for f in filenames:
        #开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  #读取nii
        img_fdata = img.get_fdata()

        fname = f.replace('.nii', '') 
        img_f_path = os.path.join(imgfile, fname+'-slice_transport_linear')
        # 创建nii对应图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  #新建文件夹
        
        #开始转换图像
        (x,y,z) = img.shape
        for i in range(85,z-60,1):   
            slice = img_fdata[:, :, i] 
            slice = 256*normalize_image(slice) 
            slice_sp_noise = sp_noise(slice, 0.1)
            # print(slice)
            # slice = strength_1(slice)
            slice_transport_log = transport_log(slice)
            # print(slice)
            slice_transport_linear = transport_linear(slice)
            slice_transport_para_linear = transport_para_linear(slice)
            slice_enhance_contrast = enhance_contrast(slice)
            # slice_enhance_contrast = gaussian_noise(slice)
            # slice_enhance_contrast = enhance_led(slice)
            
            


            slice = slice.astype(np.uint8)
            slice_transport_linear = slice_transport_linear.astype(np.uint8)
            slice_transport_para_linear = slice_transport_para_linear.astype(np.uint8)
            slice_transport_log = slice_transport_log.astype(np.uint8)
            slice_enhance_contrast = slice_enhance_contrast.astype(np.uint8)
            slice_sp_noise = slice_sp_noise.astype(np.uint8)

            
            
            # np.save(('./slice_transport_linear.npy'), slice_transport_linear)
            # file = np.load(('slice_transport_linear.npy'), allow_pickle=True)
            # print(file[0])
            
            
            # slice = slice.tolist()
            # slice_transport_linear = slice_transport_linear.tolist()
            # slice_transport_para_linear = slice_transport_para_linear.tolist()
            # slice_transport_log = slice_transport_log.tolist()
            # slice_enhance_contrast = slice_enhance_contrast.tolist()
            # slice_sp_noise = slice_sp_noise.tolist()

            

            # result.append(slice)
            # # np.save(('./ram.npy'), slice_transport_log)
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)


            # result.append(slice_transport_log)
            # # np.save(('./ram.npy'), slice_transport_log)
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)

            # result.append(slice_transport_linear)
            # # np.save(('./ram.npy'), slice_transport_log)
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)

            # result.append(slice_transport_para_linear)
            # # np.save(('./ram.npy'), slice_transport_log)
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)

            # result.append(slice_enhance_contrast)
            # # np.save(('./ram.npy'), slice_transport_log)
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)

            # result.append(slice_sp_noise)
            # # np.save(('./ram.npy'), slice_transport_log)
            
            # # file = np.load(('ram.npy'), allow_pickle=True)
            # # result.append(file)

            # arr_3d[arr_3d_index, :, :] = slice
            # arr_3d_index = arr_3d_index + 1
            # arr_3d[arr_3d_index, :, :] = slice_transport_linear
            # arr_3d_index = arr_3d_index + 1
            # arr_3d[arr_3d_index, :, :] = slice_transport_para_linear
            # arr_3d_index = arr_3d_index + 1
            # arr_3d[arr_3d_index, :, :] = slice_transport_log
            # arr_3d_index = arr_3d_index + 1
            # arr_3d[arr_3d_index, :, :] = slice_enhance_contrast
            # arr_3d_index = arr_3d_index + 1
            # arr_3d[arr_3d_index, :, :] = slice_sp_noise
            # arr_3d_index = arr_3d_index + 1

            # print(slice)

            # result.append(slice)
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), slice_transport_linear)
            



    
    # print(arr_3d[1])
    # print(result.shape)
    
    # print(result.shape)
    # print(result[1])
    # result_total = np.array(result)
    # np.save(('./arr_3d.npy'), arr_3d)
    
    
    





if __name__ == '__main__':
    filepath = 'D:\\CS\\image processing\\nii'
    imgfile = 'D:\\CS\\image processing\\image'
    nii_to_image(filepath)
