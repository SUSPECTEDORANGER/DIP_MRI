import numpy as np
import os 
import nibabel as nib
import imageio 
import random
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import math




def nii_to_image(filepath):
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
    # np.save(('./arr_3d.npy'), arr_3d