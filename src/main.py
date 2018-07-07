from keras.layers import Dense, Dropout, Flatten,Input,Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from contextlib import redirect_stdout


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#create Y train for fonts
fonts = pickle.load( open( "../data/fonts_train.p", "rb" ) )
#fonts = pickle.load( open( "../data/fonts.p", "rb" ) )
encoder = LabelEncoder()
encoder.fit(fonts)
encoded_fonts = encoder.transform(fonts)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train_font = np_utils.to_categorical(encoded_fonts)

#create Y train for char
char = pickle.load( open( "../data/char_train.p", "rb" ) )
#char = pickle.load( open( "../data/char.p", "rb" ) )
encoder = LabelEncoder()
encoder.fit(char)
encoded_char = encoder.transform(char)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train_char = np_utils.to_categorical(encoded_char)

#create Y train for bold
bold = pickle.load( open( "../data/bold_train.p", "rb" ) )
#bold = pickle.load( open( "../data/bold.p", "rb" ) )
encoder = LabelEncoder()
encoder.fit(bold)
encoded_bold = encoder.transform(bold)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train_bold = np_utils.to_categorical(encoded_bold)

#create Y train for italics
italics = pickle.load( open( "../data/italics_train.p", "rb" ) )
#italics = pickle.load( open( "../data/italics.p", "rb" ) )
encoder = LabelEncoder()
encoder.fit(italics)
encoded_italics = encoder.transform(italics)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train_italics = np_utils.to_categorical(encoded_italics)



X_train = np.load("../data/dirty_data.npy") #load data
#X_train = np.load("../data/clean_data_white.npy") #load data



# define baseline model
input_shape = X_train.shape[1:] # what are the input dimensions?

#lookup the functional API of keras...  new_layer_name = new_layer_type(parameters) (previous_layer)
inp = Input(shape=input_shape)

#take track of the convolutional layers
convs = []
conv_depths = []

conv_depth_1 = 64
conv_depths.append(conv_depth_1)
conv_1 = Convolution2D(filters = conv_depth_1, kernel_size = 3, padding = "same", activation = "relu") (inp)
convs.append(conv_1)
# In questo modo la dimensione del nostro NN sarebbe eccessivamente grande
# Usiamo il pooling in modo tale da ridurre la dimensionalita'
pool_1 = MaxPooling2D(pool_size = 2) (conv_1)

# Aggiungiamo un neurone dropout --> serve a mettere una penalizzazione quando il modello overfitta
# il parametro che si passa e' la probabiita' di "kill a neuron"
drop_1 = Dropout(0.25) (pool_1)

# Dobbiamo fare il flatting --> con il pulling abbiamo ridotto le prime due dimensioni, ma ora dobbiamo ridurre la terza
# L'obiettivo e' avere in output solamente un vettore
flat_1 = Flatten()(drop_1)

#second convolutional layer
conv_depth_2 = 64
conv_depths.append(conv_depth_2)
conv_2 = Convolution2D(filters = conv_depth_2, kernel_size = 3, padding = "same" , activation = "relu") (drop_1)
convs.append(conv_2)
pool_2 = MaxPooling2D(pool_size = 2) (conv_2)
drop_2 = Dropout(0.25) (pool_2)
flat_2 = Flatten()(drop_2)

conv_depth_3 = 64
conv_depths.append(conv_depth_3)
conv_3 = Convolution2D(filters = conv_depth_3, kernel_size = 3, padding = "same" , activation = "relu") (drop_2)
convs.append(conv_3)
pool_3 = MaxPooling2D(pool_size = 2) (conv_3)
drop_3 = Dropout(0.25) (pool_3)
flat_3 = Flatten()(drop_3)

conv_depth_4 = 64
conv_depths.append(conv_depth_4)
conv_4 = Convolution2D(filters = conv_depth_4, kernel_size = 3, padding = "same" , activation = "relu") (drop_3)
convs.append(conv_4)
pool_4 = MaxPooling2D(pool_size = 2) (conv_4)
drop_4 = Dropout(0.25) (pool_4)
flat_4 = Flatten()(drop_4)

conv_depth_5 = 64
conv_depths.append(conv_depth_5)
conv_5 = Convolution2D(filters = conv_depth_5, kernel_size = 3, padding = "same" , activation = "relu") (drop_4)
convs.append(conv_5)
pool_5 = MaxPooling2D(pool_size = 2) (conv_5)
drop_5 = Dropout(0.25) (pool_5)
flat_5 = Flatten()(drop_5)


num_classes_font = Y_train_font.shape[1]
num_classes_char = Y_train_char.shape[1]
num_classes_bold = Y_train_bold.shape[1]
num_classes_italics = Y_train_italics.shape[1]

################font network
#font_1 = Dense(32, activation='softmax')(flat_5)
#font_2 = Dense(16, activation='relu')(font_1)
output_font = Dense(num_classes_font, activation='softmax', name="output_font")(flat_5)

################char network
#char_1 = Dense(128, activation='relu')(font_1)
#char_2 = Dense(16, activation='relu')(char_1)
output_char = Dense(num_classes_char, activation='softmax', name="output_char")(flat_5)

################bold network
#bold_1 = Dense(32, activation='relu')(char_2)
#bold_2 = Dense(16, activation='relu')(bold_1)
output_bold = Dense(num_classes_bold, activation='sigmoid', name="output_bold")(flat_5)

###############italics network
#italics_1 = Dense(32, activation='relu')(bold_2)
#italics_2 = Dense(16, activation='relu')(italics_1)
output_italics = Dense(num_classes_italics, activation='sigmoid', name="output_italics")(flat_5)


#define the model
model = Model(inputs=inp, outputs=[output_font,output_char,output_bold,output_italics])


model.compile(optimizer='adam',
              loss={'output_font': 'categorical_crossentropy','output_char': 'categorical_crossentropy', 'output_bold': 'binary_crossentropy', 'output_italics': 'binary_crossentropy'},
              metrics = ["accuracy"])


# And trained it via:
model.fit(X_train,
          {'output_font': Y_train_font,'output_char': Y_train_char ,'output_bold': Y_train_bold, 'output_italics': Y_train_italics},
          epochs=10, batch_size=256)

#print("Loss, Precision: ", model.evaluate(X_test, Y_test))

#save model in .json file
model_json = model.to_json()
with open("../data_out/model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../data_out/model/model.h5")
#print("Saved model to disk")

#save model summary
with open('../data_out/model/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()




############extra points

for c in range(len(convs)):
    model_new = Model(inputs=inp, outputs=convs[c]) # To define a model, just specify its input and output layers
    
    #do not fit the new model but import some of the weights of the previous trained net
    model_new.layers[0].set_weights(model.layers[0].get_weights())
    # etc... be carefull of DROPOUT layer
    
    #create intermediate image
    single_image = X_train
    convolved_single_images = model_new.predict(single_image)
    
    convolved_single_image = convolved_single_images[0]
    #plot the output of each intermediate filter
    conv_depth = conv_depths[c]
    for i in range(conv_depth):
        filter_image = convolved_single_image[:,:,i]
        plt.subplot(6,int(conv_depth/6)+1,i+1)
        plt.imshow(filter_image,cmap='gray',); plt.axis('off');
    plt.savefig("../data_out/img/image_intermediate_"+str(c)+".png")
        