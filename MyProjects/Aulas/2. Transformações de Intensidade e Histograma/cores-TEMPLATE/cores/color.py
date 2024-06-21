import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
    
def grayscale_image(image):
    
    return image

def negative_image(image):
     
    return image

def log_transform(image): 
    
    return image

def gamma_correction(image,gamma):
    
    return image

def gray_histogram(image):
    num_bins = 256
    hist = cv2.calcHist([image],[0],None,[num_bins],[0,num_bins])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.arange(0,num_bins,1)
    ax.bar(xs,hist.flatten(),zs=0,zdir='y',color='black',ec='black',alpha=0.8)
    ax.set_xlabel('Níveis de intensidade')
    ax.set_yticks([0])
    ax.set_zlabel('Quantidade de pixels',labelpad=10)
    plt.show()      

def color_histogram(image):
    num_bins = 256
    hist_r = cv2.calcHist([image],[0],None,[num_bins],[0,num_bins])
    hist_g = cv2.calcHist([image],[1],None,[num_bins],[0,num_bins])
    hist_b = cv2.calcHist([image],[2],None,[num_bins],[0,num_bins])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.arange(0,num_bins,1)
    ax.bar(xs,hist_r.flatten(),zs=0,zdir='y',color='red',ec='red',alpha=0.8)
    ax.bar(xs,hist_g.flatten(),zs=10,zdir='y',color='green',ec='green',alpha=0.8)
    ax.bar(xs,hist_b.flatten(),zs=20,zdir='y',color='blue',ec='blue',alpha=0.8)
    ax.set_xlabel('Níveis de intensidade')
    ax.set_yticks([0,10,20])
    ax.set_yticklabels(['Red','Green','Blue'])
    ax.set_zlabel('Quantidade de pixels',labelpad=10)
    plt.show()

def show_histogram(image):
    if (np.ndim(image) > 2): #entao a imagem possui 3 canais (colorida)
        color_histogram(image)
    else:
        gray_histogram(image)

def gray_contrast_stretch(image,max,min):
    
    return image

def color_contrast_stretch(image,max,min):
    
    return image

def contrast_stretch(image,max,min):
    if (np.ndim(image) > 2): #entao a imagem possui 3 canais (colorida)
        image = color_contrast_stretch(image,max,min)
    else:
        image = gray_contrast_stretch(image,max,min)
    return image

def gray_histogram_equalization(image):
    
    return image

def color_histogram_equalization(image):
    
    image = cv2.merge([r,g,b])
    return image

def histogram_equalization(image):
    if (np.ndim(image) > 2): #entao a imagem possui 3 canais (colorida)
        image = color_histogram_equalization(image)
    else:
        image = gray_histogram_equalization(image)
    return image
