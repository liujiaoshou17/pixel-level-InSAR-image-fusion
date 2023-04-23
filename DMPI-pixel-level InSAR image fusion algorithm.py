# -*- coding: utf-8 -*-
# @Time : 2022/3/22 10:40
# @Author : 73486
# @Email : 734861423@qq.com
# @File : fuse_1.py
# @Project : Image_Process
import random
from collections import Counter
from osgeo import  gdal
import  numpy as  np
import  os
import cv2
import matplotlib.pyplot as plt
from PIL import Image



#Read tif image function
def Read_gray(path):
    # open image
    dataset = gdal.Open(path)
    # Number of rows of the raster matrix
    width = dataset.RasterXSize
    # The number of columns of the raster matrix
    height = dataset.RasterYSize
    # Get the number of bands of the raster matrix
    bands = dataset.RasterCount
    # Obtain affine matrix
    geotrans = dataset.GetGeoTransform()
    # Get map projection information
    proj = dataset.GetProjection()
    # Write the data as an array, corresponding to the raster matrix
    data= dataset.ReadAsArray(0, 0, width, height)
    return  dataset,width, height,bands, geotrans,proj,data
# Determine whether two images are collocated functions
def Judge(width_gray1, height_gray1,bands_gray1, geotrans_gray1,proj_gray1,
          width_gray2, height_gray2,bands_gray2, geotrans_gray2,proj_gray2):
    x1 = round(geotrans_gray1[0],2)
    y1 = round(geotrans_gray1[3],2)

    x2 = round(geotrans_gray2[0],2)
    y2 = round(geotrans_gray2[3],2)

    #Information marker
    flag=0
    if (width_gray1 != width_gray2 and height_gray2 != height_gray1 and bands_gray1 != bands_gray2):
        print("There is a problem with the basic parameters of the image")
    elif (proj_gray1 != proj_gray2 ):
        print("Image coordinate system is not the same")
        flag=1
    else:
        print("Images are aligned")
    return  flag

#Selecting the main image function - using grayscale images
def ChooseMainImage(data1,data2):
    d1_num=Counter(data1)
    d2_num=Counter(data2)

    #Use Counter to get the number of zeros in each row
    d1_num = [Counter(data1[:, i]).most_common(1)[0] for i in range(int(data1.shape[1]))]
    d2_num = [Counter(data2[:, i]).most_common(1)[0] for i in range(int(data2.shape[1]))]
    d1_No = np.sum((d1_num),axis=0)
    d2_No = np.sum((d2_num),axis=0)
    #Should use that image, direct comparison
    if d1_No[1]>=d2_No[1]:
        return '2'
    else:
        return '1'

def ReadImage():
    dataset_gray1,width_gray1, height_gray1,bands_gray1, geotrans_gray1,proj_gray1,data_gray_gray1=Read_gray(path_gray_1)
    dataset_gray2,width_gray2, height_gray2,bands_gray2, geotrans_gray2,proj_gray2,data_gray_gray2=Read_gray(path_gray_2)
    #读取rgb图像
    dataset_rgb1,width_rgb1, height_rgb1,bands_rgb1, geotrans_rgb1,proj_rgb1,data_rgb1=Read_gray(path_rgb_1)
    dataset_rgb2,width_rgb2, height_rgb2,bands_rgb2, geotrans_rgb2,proj_rgb2,data_rgb2=Read_gray(path_rgb_2)

def Choosegray():
    if main_image_name=='1':
        gray_img_1 = np.array(data_gray_gray1,copy=True)
        gray_img_2 = np.array(data_gray_gray2, copy=True)
    elif main_image_name=='2':
        gray_img_1 = np.array(data_gray_gray2,copy=True)
        gray_img_2 = np.array(data_gray_gray1, copy=True)
    else:
        print("ERROR: the master gray image ")

# Createmetric
def Createmetric():
    if equal==0 and data_gray_gray2.shape==data_gray_gray1.shape :
        threshold=0.009
        w=decision_metric.shape[0]
        h=decision_metric.shape[1]
        for i in range(w):
            for j in range(h):
                if abs(gray_img_1[i][j]) == 0 and abs(gray_img_2[i][j]) != 0:
                    decision_metric[i][j] = 2
                elif abs(gray_img_1[i][j]) == 0 and abs(gray_img_2[i][j]) == 0:
                    decision_metric[i][j] = 3
                if abs(gray_img_1[i][j]) != 0 and abs(gray_img_2[i][j]) == 0:
                    decision_metric[i][j] = 1
                elif abs(gray_img_1[i][j]) > threshold and abs(gray_img_2[i][j]) > threshold and (
                        abs(gray_img_2[i][j]) - abs(gray_img_1[i][j])) >= 0.1:
                    decision_metric[i][j] = 2  # Increase brightness
                elif abs(gray_img_1[i][j]) > threshold and abs(gray_img_2[i][j]) > threshold:
                    decision_metric[i][j] = 1
                elif abs(gray_img_1[i][j]) > threshold and abs(gray_img_2[i][j]) < threshold:
                    decision_metric[i][j] = 1
                elif abs(gray_img_1[i][j]) <= threshold and abs(gray_img_2[i][j]) > threshold:
                    decision_metric[i][j] = 2
                elif abs(gray_img_1[i][j]) <= threshold and abs(gray_img_2[i][j]) <= threshold:
                    decision_metric[i][j] = 1
                else:
                    print("ERROR:the logic of Decision matrix ")
        print("Decision matrix is success")
    else:
        print("ERROR:Can't create the Decision matrix")

##nodata replace
if main_image_name=='1':
    result_img=np.array(data_rgb1_c,copy=True)
    data_rgb_result = np.array(data_rgb2_c, copy=True)
elif main_image_name=='2':
    result_img=np.array(data_rgb2_c,copy=True)
    data_rgb_result=np.array(data_rgb1_c,copy=True)
#CreateFusion:
def CreateFusion():
    if equal == 0 and result_img.shape == data_rgb_result.shape:
        w = result_img.shape[0]
        h = result_img.shape[1]
        for i in range(w):
            for j in range(h):
                if decision_metric[i, j] == 1:
                    pass
                elif decision_metric[i, j] == 2:
                    result_img[i, j, 0] = data_rgb_result[i, j, 0]
                    result_img[i, j, 1] = data_rgb_result[i, j, 1]
                    result_img[i, j, 2] = data_rgb_result[i, j, 2]
                elif decision_metric[i, j] == 3:
                    result_img[i, j, 0] = 255
                    result_img[i, j, 1] = 255
                    result_img[i, j, 2] = 255
                else:
                    print("Problem with merging image assignment")
        print("The target image has been generated")
    else:
        print("Incorrect data does not generate a merged image")

    print("data", result_img)
####Filter
def FilterImage(result_img):

    img=np.array(result_img,copy=True)
    img_mean = cv2.blur(img, (5,5))
    img_Guassian = cv2.GaussianBlur(img,(5,5),0)
    img_median = cv2.medianBlur(img, 5)
    img_bilater = cv2.bilateralFilter(img,9,75,75)
    titles = ['srcImg','mean', 'Gaussian', 'median', 'bilateral']
    imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

    from PIL import Image
    for i in range(5):
        plt.subplot(2,3,i+1)
        #this is similar to matlab in that there are no zeros and the array subscript starts at 1
        plt.imshow(imgs[i])
        name="C:\ly_something\pythonProgram\Image_Process/filter/"+"23+12"+titles[i]+"test4.png"
        img = Image.fromarray(imgs[i])
        img.save(name)
        plt.title(titles[i])
    plt.show()
####SaveRGB
def SaveRGB(result_img) :
    img = Image.fromarray(result_img.astype(np.uint8))
    img.save("C:\ly_something\pythonProgram\Image_Process/%3s.png"%str(random.randint(50,100)))
####SaveTif
def SaveTIT(result_img,result_path) :
    result_img_8=result_img
    r1=result_img_8[:,:,0]
    r2=result_img_8[:,:,1]
    r3=result_img_8[:,:,2]
    # create dataset and driver
    driver = gdal.GetDriverByName("GTiff")
    #datatype
    dataset = driver.Create(result_path,width_rgb2, height_rgb2,3, gdal.GDT_Byte)
    if dataset !=None:
        #GeoTransform
        dataset.SetGeoTransform(geotrans_rgb2)
        #Projection
        dataset.SetProjection(proj_rgb2)
        dataset.GetRasterBand(1).WriteArray(r1)
        dataset.GetRasterBand(2).WriteArray(r2)
        dataset.GetRasterBand(3).WriteArray(r3)

def mainFlow():
    # Folder path
    dir_path = r"C:\ly_something\arcgisProgram\three River\fusion\three_fusion\tif"
    # Binary image path
    path_gray_1 = os.path.join(dir_path, "iw121.tif")
    path_gray_2 = os.path.join(dir_path, "iw231.tif")
    # Color image path
    path_rgb_1 = os.path.join(dir_path, "D_RGB.tif")
    path_rgb_2 = os.path.join(dir_path, "A_RGB.tif")
    decision_metric = np.empty(shape=(data_gray_gray2.shape[0], data_gray_gray2.shape[1]))
    equal = Judge(width_gray1, height_gray1, bands_gray1, geotrans_gray1, proj_gray1,
                  width_gray2, height_gray2, bands_gray2, geotrans_gray2, proj_gray2)
    data_rgb1_c = data_rgb1.transpose(1, 2, 0)
    data_rgb2_c = data_rgb2.transpose(1, 2, 0)
    equal = Judge(width_rgb1, height_rgb1, bands_rgb1, geotrans_rgb1, proj_rgb1,
                  width_rgb2, height_rgb2, bands_rgb2, geotrans_rgb2, proj_rgb2)
    main_image_name = ChooseMainImage(data_gray_gray1, data_gray_gray2)
    result_img=CreateFusion()
    FilterImage(result_img)
    result_path=""
    SaveTIT(result_img, result_path)
if __name__ == '__main__':
    print(" ")