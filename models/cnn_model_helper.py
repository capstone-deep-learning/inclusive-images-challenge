## Import Required libraries
import os, fnmatch
from random import shuffle
import shutil
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Maximum, ZeroPadding2D, BatchNormalization
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.optimizers import Adam, SGD
from keras import optimizers, regularizers
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, ReLU


#####################################################################################################
## Helper functions for placing images in various folders
#####################################################################################################
# This method splits and copies the images in input folder to 3 different folders 
# (train, validate and test)
def SplitImagesToTrainTestFolders(inputFileFolder, targetFileFolder):    
    ## lop through all the files and sub-folders
    for root, dirs, files in os.walk(inputFileFolder):

        file=[]
        ## Read images and convert the same into array, also fetch/set their label
        for f in files:        
            ## Set the directory elements
            file.append(f)

        if(file!=[]):
            ## Create/update label dictionary
            if(dirs == []):
                imageLabel = os.path.basename(os.path.normpath(root))

            ## Shuffle the image URL's to split in train and test data
            shuffle(file)

            tgtTrainPath = os.path.join(targetFileFolder,"Train",imageLabel)
            tgtValidPath = os.path.join(targetFileFolder,"Validation",imageLabel)
            tgtTestPath = os.path.join(targetFileFolder,"Test",imageLabel)

            copyImages(root, tgtTrainPath, file[:1000])
            copyImages(root, tgtValidPath, file[1000:1100])
            testIdx = min(1120, len(file))
            copyImages(root, tgtTestPath, file[1100:testIdx])


## Copy image from one location to another
def copyImages(srcPath, tgtPath, lsFileNames):
    if not os.path.exists(tgtPath):
        os.makedirs(tgtPath)
            
    for fileName in lsFileNames:
        srcFile = os.path.join(srcPath,fileName)
        tgtFile = os.path.join(tgtPath,fileName)
        shutil.copy2(srcFile, tgtFile)


#####################################################################################################
## Helper functions for model creation
#####################################################################################################

# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), padding='same'):
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides, 
                  padding=padding)(feature_batch)
    act = ReLU()(conv)
    bn = BatchNormalization(axis=3)(act)
    return bn


# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act


# Conv. layers set
def conv_layer_mod(feature_batch, feature_map,layer_name, kernel_size=(3, 3),strides=(1,1), padding='same'):
    
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides, 
                  padding=padding, activation='relu', name=layer_name)(feature_batch)
    
    bn = BatchNormalization(axis=3)(conv)
    return bn



#####################################################################################################
## Some models to be tested
#####################################################################################################
# simple model 
def getSimpleModel(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    conv1 = conv_layer(inp_img, 32)
    mp1 = MaxPooling2D(pool_size=(4, 4))(conv1)
    
    # dense layers
    flt = Flatten()(mp1)
    ds1 = dense_set(flt, 256, activation='relu')
    out = dense_set(ds1, num_classes, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)
    model.compile(loss='categorical_crossentropy',
                   optimizer=myoptim,
                   metrics=['accuracy'])
    model.summary()
    return model


# Creates 10 layer model
def get10LayerModel(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    conv1 = conv_layer(inp_img, 64)
    conv2 = conv_layer(conv1, 64)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    conv3 = conv_layer(mp1, 128)
    conv4 = conv_layer(conv3, 128)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    conv5 = conv_layer(mp2, 256)
    conv6 = conv_layer(conv5, 256)
    conv7 = conv_layer(conv6, 256)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv7)

    # dense layers
    flt = Flatten()(mp3)
    ds1 = dense_set(flt, 128, activation='relu') ## Changed it from 128 to 512
    #ds2 = dense_set(ds1, 512, activation='relu') ## Added this layer
    out = dense_set(ds1, num_classes, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=myoptim, metrics=['accuracy'])
    model.summary()
    return model


def get6LayerModel(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    conv1 = conv_layer(inp_img, 32, padding='same')
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    conv2 = conv_layer(mp1, 64, padding='same')
    mp2 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(conv2)
    conv3 = conv_layer(mp2, 128)
    mp3 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(conv3)
    conv4 = conv_layer(mp3, 256)
    mp4 = MaxPooling2D(pool_size=(7, 7), strides=(2, 2))(conv4)

    # dense layers
    flt = Flatten()(mp4)
    ds1 = dense_set(flt, 64, activation='relu')
    ds2 = dense_set(ds1, 128, activation='relu')
    out = dense_set(ds2, num_classes, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=myoptim, metrics=['accuracy'])
    model.summary()
    return model


def get6LayerModelMod(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    
    conv1 = conv_layer_mod(inp_img, 32, "block1_conv1")
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="block1_pool")(conv1)
       
    conv2 = conv_layer_mod(mp1, 64, "block2_conv1")
    mp2 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2), name="block2_pool")(conv2)
    
    conv3 = conv_layer_mod(mp2, 128, "block3_conv1") #, kernel_size=(5, 5)
    mp3 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2), name="block3_pool")(conv3)
    
    conv4 = conv_layer_mod(mp3, 256, "block4_conv1") # , kernel_size=(5, 5)
    mp4 = MaxPooling2D(pool_size=(7, 7), strides=(2, 2), name="block4_pool")(conv4)

    # dense layers
    flt = Flatten(name="flatten")(mp4)

    dns1 = Dense(64, activation='relu', name="dense1")(flt)
    bn1 = BatchNormalization(axis=-1)(dns1)
    dp1 = Dropout(0.2, name="dropout1")(bn1)
    
    dns2 = Dense(128, activation='relu', name="dense2")(dp1)
    bn2 = BatchNormalization(axis=-1)(dns2)
    dp2 = Dropout(0.2, name="dropout2")(bn2)
    
    out = Dense(num_classes, activation='softmax', name="output")(dp2)

    model = Model(inputs=inp_img, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=myoptim, metrics=['accuracy'])
    model.summary()
    return model


def createTransferModel(base_model, freezeLayers=10, 
                        optimizer = optimizers.SGD(lr=0.0001, momentum=0.9)):
    # Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
    for layer in base_model.layers[:freezeLayers]:
        layer.trainable = False

    #Adding custom Layers 
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    #x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input = base_model.input, output = predictions)

    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", 
                        optimizer = optimizer,    
                        metrics=["accuracy"])

    model_final.summary()
    return model_final


#############################################################################################################
## Models created on 2019-02-17
#############################################################################################################
# Creates 7 layer model
def get7Layer_4Conv_3Dense_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
	
	## Blcok 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR')(inp_img)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)

	## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Blcok 4
    model = Conv2D(256,kernel_size=3,activation='relu', padding='same', name='BLOCK4_CONV1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK4_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK4_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(256, activation='relu', name='DENSE1_LYR')(model)
    model = Dropout(0.4, name='DENSE1_DROPOUT_LYR')(model)
    model = Dense(256, activation='relu', name='DENSE2_LYR')(model)
    model = Dropout(0.4, name='DENSE2_DROPOUT_LYR')(model)

	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT_LYR')(model)
    
    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(loss='categorical_crossentropy', optimizer=myoptim, metrics=['accuracy'])
    modelx.summary()

    return modelx


# Creates 7 layer model
def get7Layer_4Conv_3Dense_BN_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))

    ## Block 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR')(inp_img)
    model = BatchNormalization(name='BLOCK1_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)
    
    ## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR')(model)
    model = BatchNormalization(name='BLOCK2_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR')(model)
    model = BatchNormalization(name='BLOCK3_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Blcok 4
    model = Conv2D(256,kernel_size=3,activation='relu', padding='same', name='BLOCK4_CONV1_LYR')(model)
    model = BatchNormalization(name='BLOCK4_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK4_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK4_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(256, activation='relu', name='DENSE1_LYR')(model)
    model = BatchNormalization(name='DENSE1_BN_LYR')(model)
    model = Dropout(0.4, name='DENSE1_DROPOUT_LYR')(model)
    model = Dense(256, activation='relu', name='DENSE2_LYR')(model)
    model = BatchNormalization(name='DENSE2_BN_LYR')(model)
    model = Dropout(0.4, name='DENSE2_DROPOUT_LYR')(model)
	
	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT_LYR')(model)
	
    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(optimizer=myoptim, loss="categorical_crossentropy", metrics=["accuracy"])
    
    modelx.summary()
    return modelx



# Creates 7 layer model
def get7Layer_4Conv_3Dense_BN_Reg_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))

    ## Blcok 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(inp_img)
    model = BatchNormalization(name='BLOCK1_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)
    
    ## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK2_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK3_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Blcok 4
    model = Conv2D(256,kernel_size=3,activation='relu', padding='same', name='BLOCK4_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK4_BN1_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK4_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK4_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(256, activation='relu', name='DENSE1_LYR', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='DENSE1_BN_LYR')(model)
    model = Dropout(0.4, name='DENSE1_DROPOUT_LYR')(model)
    model = Dense(256, activation='relu', name='DENSE2_LYR', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='DENSE2_BN_LYR')(model)
    model = Dropout(0.4, name='DENSE2_DROPOUT_LYR')(model)
	
	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT_LYR')(model)
	
    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(optimizer=myoptim, loss="categorical_crossentropy", metrics=["accuracy"])
    
    modelx.summary()
    return modelx



# Creates 9 layer model
def get9Layer_6Conv_3Dense_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    
    ## Blcok 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR')(inp_img)
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)

	## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR')(model)
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR')(model)
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(128, activation='relu', name='DENSE1_LYR')(model)
    model = Dropout(0.3, name='DENSE1_DROPOUT_LYR')(model)
    model = Dense(128, activation='relu', name='DENSE2_LYR')(model)
    model = Dropout(0.3, name='DENSE2_DROPOUT_LYR')(model)
	
	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT_LYR')(model)

    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(optimizer=myoptim, loss="categorical_crossentropy", metrics=["accuracy"])
    
    modelx.summary()
    return modelx


# Creates 9 layer model
def get9Layer_6Conv_3Dense_BN_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3))
    
	## Blcok 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR')(inp_img)
    model = BatchNormalization(name='BLOCK1_BN1_LYR')(model)
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV2_LYR')(model)
    model = BatchNormalization(name='Block1_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)

	## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR')(model)
    model = BatchNormalization(name='BLOCK2_BN1_LYR')(model)
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV2_LYR')(model)
    model = BatchNormalization(name='BLOCK2_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR')(model)
    model = BatchNormalization(name='BLOCK3_BN1_LYR')(model)
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV2_LYR')(model)
    model = BatchNormalization(name='BLOCK3_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(128, activation='relu', name='DENSE1')(model)
    model = BatchNormalization(name='DENSE1_BN')(model)
    model = Dropout(0.3, name='DENSE1_DROPOUT')(model)
    model = Dense(128, activation='relu', name='DENSE2')(model)
    model = BatchNormalization(name='DENSE2_BN')(model)
    model = Dropout(0.3, name='DENSE2_DROPOUT')(model)
	
	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT')(model)
    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(optimizer=myoptim, loss="categorical_crossentropy", metrics=["accuracy"])
    
    modelx.summary()
    return modelx



# Creates 9 layer model
def get9Layer_6Conv_3Dense_BN_Reg_Model(num_classes, myoptim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)):
    inp_img = Input(shape=(256, 256, 3), name='INPUT_LYR')
    
	## Blcok 1
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(inp_img)
    model = BatchNormalization(name='BLOCK1_BN1_LYR')(model)
    model = Conv2D(32,kernel_size=3,activation='relu', padding='same', name='BLOCK1_CONV2_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='Block1_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK1_POOL1_LYR')(model)
    #model = Dropout(0.1, name='BLOCK1_DROPOUT1_LYR')(model)

	## Blcok 2
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK2_BN1_LYR')(model)
    model = Conv2D(64,kernel_size=3,activation='relu', padding='same', name='BLOCK2_CONV2_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK2_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK2_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK2_DROPOUT1_LYR')(model)

	## Blcok 3
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV1_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK3_BN1_LYR')(model)
    model = Conv2D(128,kernel_size=3,activation='relu', padding='same', name='BLOCK3_CONV2_LYR'
                   , kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='BLOCK3_BN2_LYR')(model)
    model = MaxPooling2D(strides=2, name='BLOCK3_POOL1_LYR')(model)
    #model = Dropout(0.3, name='BLOCK3_DROPOUT1_LYR')(model)

	## Flatten the inputs
    model = Flatten(name='FLATTEN_LYR')(model)

	## Dense layers
    model = Dense(128, activation='relu', name='DENSE1_LYR', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='DENSE1_BN_LYR')(model)
    model = Dropout(0.3, name='DENSE1_DROPOUT_LYR')(model)
    model = Dense(128, activation='relu', name='DENSE2_LYR', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization(name='DENSE2_BN_LYR')(model)
    model = Dropout(0.3, name='DENSE2_DROPOUT_LYR')(model)
	
	## Final output layer
    model = Dense(num_classes, activation='softmax', name='OUTPUT_LYR')(model)
    modelx = Model(inputs=inp_img, outputs=model)
    modelx.compile(optimizer=myoptim, loss="categorical_crossentropy", metrics=["accuracy"])
    
    modelx.summary()
    
    return modelx


#####################################################################################################
## Helper functions to analyse the models outcome
#####################################################################################################

## Confusion matrix
def createConfusionMatrix(y_test, y_pred, labelCode, imageName=''):
    cr = metrics.classification_report(y_test,y_pred)
    print(cr)

    cm = metrics.confusion_matrix(y_test, y_pred)
    #print(cm)    
    dfCM = pd.DataFrame(cm, index=list(labelCode), columns=list(labelCode))
    plt.figure(figsize=(80,20))
    ax = sns.heatmap(dfCM,vmax=8, square=True, fmt='.2f',annot=True, 
                     linecolor='white', linewidths=0.1)
    
    if(imageName != ''):
        imgPath = '/home/jupyter/glcapstone/Logs/ConfusionMatrix/'+imageName+'.png'
        print('Saving confusion Matrix at:', imgPath)
        plt.savefig(imgPath)
    plt.show()

