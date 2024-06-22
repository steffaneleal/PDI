import cv2
import math
import numpy as np

def read_image(filename): 
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
    
def resize_image(image,width,height):
    rows, cols, channels = image.shape
    w = int(math.ceil(cols*width/100))
    h = int(math.ceil(rows*height/100))
    new_size = (w,h)
    image = cv2.resize(image,new_size)
    return image

def average_filter(image,kernel_size):
    image = cv2.blur(image,(kernel_size,kernel_size))
    return image

def gaussian_filter(image,kernel_size):
    standard_deviation = 0
    image = cv2.GaussianBlur(image,(kernel_size,kernel_size),standard_deviation)
    return image

def median_filter(image,kernel_size):
    image = cv2.medianBlur(image,kernel_size)
    return image

def salt_and_pepper_noise(image):
    #h (rows), w (cols)
    h,w, c = image.shape
    noise = np.zeros((h,w),np.uint8)
    cv2.randu(noise,0,255)
    image[noise <= 5] = 0
    image[noise >= 250] = 255
    return image

def sobel_filter(image):
    if len(image.shape) > 2: #imagem eh colorida
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
    else:
        gray_image = image
    #detectar bordas na direcao x (verticais)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F,1,0,ksize=3)
    #detectar bordas na direcao y (horizontais)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F,0,1,ksize=3)

    #magnitude dos gradientes
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    #normalizar para o intervalo [0,255]
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    edge_image = np.uint8(gradient_magnitude)
 
    if len(image.shape) > 2: #imagem eh colorida
        edge_image = cv2.cvtColor(edge_image,cv2.COLOR_GRAY2RGB) 
    
    #image = cv2.add(image,edge_image)
    image = cv2.addWeighted(image, 0.7, edge_image, 0.3, 0)
 
    return image
    

def laplacian_filter(image):
    image = median_filter(image,3)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    laplacian = cv2.filter2D(image,-1,kernel)
    image = cv2.add(image,laplacian)
    return image

def highboost_filter(image,a):
    image = median_filter(image,3)
    kernel = np.array([[0, -1, 0],
                       [-1, 4+a, -1],
                       [0, -1, 0]])
    laplacian = cv2.filter2D(image,-1,kernel)
    image = cv2.add(image,laplacian)
    return image
