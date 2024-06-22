import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
import time

def main():
    mainStartTime = time.time()
    trainImagePath = 'C:/Users/paupi/Documentos/Projeto_Python_Codigo_Completo_Extrac_Classif_Classico/classificacao/images_split/train/'
    testImagePath = 'C:/Users/paupi/Documentos/Projeto_Python_Codigo_Completo_Extrac_Classif_Classico/classificacao/images_split/test/'
    trainFeaturePath = 'C:/Users/paupi/Documentos/Projeto_Python_Codigo_Completo_Extrac_Classif_Classico/classificacao/features_labels/sift/train/'
    testFeaturePath = 'C:/Users/paupi/Documentos/Projeto_Python_Codigo_Completo_Extrac_Classif_Classico/classificacao/features_labels/sift/test/'
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractSIFTFeatures(trainImages)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractSIFTFeatures(testImages)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):   
            if len(filenames) > 0:  # it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels, dtype=object)

def extractSIFTFeatures(images):
    sift = cv2.SIFT_create()
    bar = Bar('[INFO] Extracting SIFT features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if len(image.shape) > 2:  # color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            features = np.mean(descriptors, axis=0)
            featuresList.append(features)
        else:
            featuresList.append(np.zeros(128))
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = f'{labels=}'.split('=')[0] + '.csv'
    feature_filename = f'{features=}'.split('=')[0] + '.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0] + '.csv'
    os.makedirs(path, exist_ok=True)
    np.savetxt(os.path.join(path, label_filename), labels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path, feature_filename), features, delimiter=',')  # float does not need format
    np.savetxt(os.path.join(path, encoder_filename), encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
