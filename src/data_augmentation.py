#Creazione Dataset

import skimage.transform
import numpy as np
import random
import pickle


clean_data = np.load("../data/clean_data.npy")
height = 64
width = 64
shape = [height, width, 1]

##Ciclo per sporcare tutte le immagini

num_images_per_letter = 1

npy_file = []
#j = 0
for letter in clean_data:
    #print(j)
    #j += 1
    for i in range(num_images_per_letter):
        
        #insert clean letter
        if i == 0:
            npy_file.append(letter/255)
    
        #Rotation
        angle = random.randint(-35, 35)
        output_rotation = skimage.transform.rotate(letter, angle= angle, mode='constant',cval=255, preserve_range= True)
        #show_rotation = np.resize(output, (64, 64))
        npy_file.append(output_rotation/255)
    
        #scale from clean
        two_d_matrix = np.resize(letter, (64, 64))
        works = False
        while works==False:
            scale = random.randint(10, 20)/10
            scaling = skimage.transform.rescale(two_d_matrix, scale= scale, mode='constant', cval=255, preserve_range=True)
            if scaling.shape[0] % 2 == 0:
                works = True
        if scale < 1:
            put_in = int((64 - scaling.shape[0])/2)
            scaling = np.pad(scaling, pad_width = put_in, mode = "wrap")
        elif scale > 1:
            take_out = int((scaling.shape[0] - 64)/2)
            scaling = scaling[take_out:-take_out, take_out:-take_out]
        output_scale_c = np.resize(scaling, (64, 64, 1))
        npy_file.append(output_scale_c/255)
        
        #scale from rotated
        two_d_matrix = np.resize(output_rotation, (64, 64))
        works = False
        while works==False:
            scale = random.randint(10, 20)/10
            scaling = skimage.transform.rescale(two_d_matrix, scale= scale, mode='constant', cval=255, preserve_range=True)
            if scaling.shape[0] % 2 == 0:
                works = True
        if scale < 1:
            put_in = int((64 - scaling.shape[0])/2)
            scaling = np.pad(scaling, pad_width = put_in, mode = "wrap")
        elif scale > 1:
            take_out = int((scaling.shape[0] - 64)/2)
            scaling = scaling[take_out:-take_out, take_out:-take_out]
        output_scale_r = np.resize(scaling, (64, 64, 1))
        npy_file.append(output_scale_r/255)
        
        ##Salt and Pepper from clean
        output_salt_pepper_c = skimage.util.random_noise(letter,mode='salt')
        npy_file.append(output_salt_pepper_c)
        
        ##Salt and Pepper from rotated
        output_salt_pepper_r = skimage.util.random_noise(output_rotation,mode='salt')
        npy_file.append(output_salt_pepper_r)
        
        ##Salt and Pepper from rotated and scale
        output_salt_pepper_r_s = skimage.util.random_noise(output_scale_r,mode='salt')
        npy_file.append(output_salt_pepper_r_s)
        


#####create y_trains
#1 initial clean images(one for every position)
#6 number of transformations
#4 random transformation
number_of_images_per_char = 1*18 + 6*num_images_per_letter*18

fonts = pickle.load( open( "../data/fonts.p", "rb" ) )
fonts_train = [x for x in fonts for _ in range(number_of_images_per_char)]

char = pickle.load( open( "../data/char.p", "rb" ) )
char_train = [x for x in char for _ in range(number_of_images_per_char)]

bold = pickle.load( open( "../data/bold.p", "rb" ) )
bold_train = [x for x in bold for _ in range(number_of_images_per_char)]

italics = pickle.load( open( "../data/italics.p", "rb" ) )
italics_train = [x for x in italics for _ in range(number_of_images_per_char)]


#remove white images
#identify position of the white images
indexes = []
for e in range(len(npy_file)):
    if np.mean(npy_file[e])==1:
        indexes.append(e)
#replace them with a "W"
num_white = len(indexes)
replacements = ["White"]*num_white
for (index, replacements) in zip(indexes, replacements):
    npy_file[index] = replacements
    fonts_train[index] = replacements
    char_train[index] = replacements
    bold_train[index] = replacements
    italics_train[index] = replacements
#remove them
npy_file = list(filter(lambda a: a != "White", npy_file))
fonts_train = list(filter(lambda a: a != "White", fonts_train))
char_train = list(filter(lambda a: a != "White", char_train))
bold_train = list(filter(lambda a: a != "White", bold_train))
italics_train = list(filter(lambda a: a != "White", italics_train))


#create an array for .npy file
npy_file = np.array(npy_file, dtype=np.float64)
npy_file = np.resize(npy_file, (len(npy_file),64,64,1))
#load X train
np.save("../data/dirty_data.npy",npy_file)

#load Y trains
pickle.dump( fonts_train, open( "../data/fonts_train.p", "wb" ) )
pickle.dump( char_train, open( "../data/char_train.p", "wb" ) )
pickle.dump( bold_train, open( "../data/bold_train.p", "wb" ) )
pickle.dump( italics_train, open( "../data/italics_train.p", "wb" ) )


##Per printare l'immagine in npy_file
'''import matplotlib.pyplot as plt
modifica = np.resize(npy_file[0], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[1], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[2], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[3], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[4], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[5], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[6], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[7], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
modifica = np.resize(npy_file[8], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)
'''
'''import math
np.mean(npy_file[1910])'''