import cv2
import math
import numpy as np


def read_image(filename): 
    image = cv2.imread(filename) #serve para ler a imagem
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #vira um array da numpy
    return image
    
def resize_image(image,width,height):
    rows, cols, channels = image.shape
    w = int(math.ceil(cols*width/100))
    h = int(math.ceil(rows*height/100))
    new_size = (w,h)
    image = cv2.resize(image,new_size)
    return image

def average_filter(image,kernel_size):
    #blur = média simples
    image = cv2.blur(image,(kernel_size, kernel_size)) #repete o kernel_size pra ficar tipo 3x3, 4x4 e etc
    return image

#FILTROS DE SUAVIZAÇÃO (BLUR)
def gaussian_filter(image,kernel_size):
    standard_deviation = 0
    #Gaussiano = média ponderada, mais leve que o blur
    image = cv2.GaussianBlur(image,(kernel_size,kernel_size),standard_deviation)
    return image

def median_filter(image,kernel_size):
    image = cv2.medianBlur(image,kernel_size)
    return image

def salt_and_pepper_noise(image):
    #h (rows), w (cols)
    h,w, c = image.shape
    noise = np.zeros((h,w),np.uint8) #noise é uma imagem que só tem ruídos
    cv2.randu(noise,0,255)
    image[noise <= 5] = 0
    image[noise >= 250] = 255
    return image

#FILTROS DE AGUÇAMENTO/NITIDEZ (realça ruídos também)
def sobel_filter(image):
   
    return image
    

def laplacian_filter(image):
    
    return image

def highboost_filter(image,a):
   
    return image
