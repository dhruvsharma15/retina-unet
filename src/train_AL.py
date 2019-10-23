#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:13:23 2019

@author: dhruv
"""

import configparser

from keras.utils.vis_utils import plot_model as plot

from model import get_unet, get_gnet
import sys
sys.path.insert(0, './lib/')
from lib.help_functions import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


#function to obtain data for training/testing (validation)
from lib.extract_patches import get_data_training, get_data_testing

from datetime import datetime
import numpy as np
from functools import partial


import keras.backend as K
import tensorflow as tf
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from active_learner import ActiveLearner
from strategies.uncertainty import *


def main():
    start=datetime.now()
    
    #========= Load settings from Config file
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    #patch to the datasets
    path_data = config.get('data paths', 'path_local')
    #Experiment name
    name_experiment = config.get('experiment name', 'name')
    #training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    
    try:
        nb_active_epochs = int(config.get('training settings', 'nb_active_epochs'))
    except:
        nb_active_epochs = 10
    
    ###################### active learning hyperparameters #####
    try:
        nb_labeled = int(config.get('AL_params', 'nb_labeled'))
    except:
        nb_labeled = 9000
        
    try:
        nb_iterations = int(config.get('AL_params', 'nb_iterations'))
    except:
        nb_iterations = 5
    
    try:
        nb_annotations = int(config.get('AL_params', 'nb_annotations'))
    except:
        nb_annotations = 600
        
    try:
        strategy = config.get('AL_params', 'query_strategy')
        if(strategy == 'informative_batch_sampling'):
            query_strategy = partial(informative_batch_sampling, n_instances=nb_annotations)
        if(strategy == 'uncertainty_batch_sampling'):
            query_strategy = partial(uncertainty_batch_sampling, n_instances=nb_annotations)
        if(strategy == 'mc_dropout_sampling'):
            query_strategy = partial(mc_dropout_sampling, n_instances=nb_annotations)
        if(strategy == 'spatial_unceratinty_sampling'):
            query_strategy = partial(spatial_unceratinty_sampling, n_instances=nb_annotations)
        if(strategy == 'uncertainty_sampling'):
            query_strategy = uncertainty_sampling
        
    except:
        query_strategy = partial(informative_batch_sampling, n_instances=nb_annotations)
    
    
    #============ Load the data and divided in patches
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = int(config.get('training settings', 'N_subimgs')),
        inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
    )
    

    X_test, Y_test = get_data_testing(
        DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original'),
        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width'))
    )
    
    Y_test = masks_Unet(Y_test)  #reduce memory consumption
    
    ###############################
    patches_imgs_train = patches_imgs_train[:2000]
    patches_masks_train = patches_masks_train[:2000]
    
    #========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0],40)
    visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
    visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()
    
    
    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
    ###############################
    X_patches, X_valid, Y_labels, Y_valid = train_test_split(patches_imgs_train, patches_masks_train, test_size=0.1, random_state=42)
    
    ##################################################
        
#    nb_unlabeled = X_patches.shape[0] - nb_labeled
    initial_idx = np.random.choice(range(len(X_patches)), size=nb_labeled, replace=False)

    ##################################################
    
    # DB definition
    

    X_labeled_train = X_patches[initial_idx]
    y_labeled_train = Y_labels[initial_idx]
    
    
    X_pool = np.delete(X_patches, initial_idx, axis=0)
    y_pool = np.delete(Y_labels, initial_idx, axis=0)
    # (1) Initialize model
    
    
    
    
    #=========== Construct and save the model arcitecture =====
    n_ch = patches_imgs_train.shape[3]
    patch_height = patches_imgs_train.shape[1]
    patch_width = patches_imgs_train.shape[2]
    model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
    print("Check: final output of the network:")
    print(model.output_shape)
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
    
#    print(model.summary())
    
    # Active learner initilization
    learner = ActiveLearner(model = model,
                            query_strategy = query_strategy,
                            X_training = X_labeled_train,
                            y_training = y_labeled_train,
                            weights_path = name_experiment,
                            X_val = X_valid,
                            y_val = Y_valid,
                            verbose = 1, epochs = N_epochs,
                            batch_size = batch_size
                            )
    ############ testing ############
    val_output = learner.evaluate(X_test, Y_test, verbose = 1)
        
    for i in range(len(learner.model.metrics_names)):
        print(learner.model.metrics_names[i], val_output[i])
    ################################
    
    layer_name = 'conv2d_8'
    intermediate_layer_model = Model(inputs=learner.model.input,
                                         outputs=learner.model.get_layer(layer_name).output)
    for idx in range(nb_iterations):
        nb_active_epochs = nb_active_epochs + 2
        
        
#        ## features of the labeled and the unlabeled pool ############
#        print('extracting features from the encoder of the UNet')
#        n_dims = min(512,min(len(learner.X_training), len(X_pool)))
#        
#        print('applying PCA for labeled pool')
#        labeled_inter = intermediate_layer_model.predict(learner.X_training)
#        labeled_inter = labeled_inter.reshape((len(labeled_inter), -1))
#        labeled_inter = StandardScaler().fit_transform(labeled_inter)
#        
#        print('applying PCA for unlabeled pool')
#        unlabeled_inter = intermediate_layer_model.predict(X_pool)
#        unlabeled_inter = unlabeled_inter.reshape((len(unlabeled_inter), -1))
#        unlabeled_inter = StandardScaler().fit_transform(unlabeled_inter)
#        
#        pca = PCA(n_components = min(n_dims, min(labeled_inter.shape)))
#        features_labeled = pca.fit_transform(labeled_inter)
#        
#        pca = PCA(n_components = min(n_dims, min(unlabeled_inter.shape)))
#        features_unlabeled = pca.fit_transform(unlabeled_inter)
        #################################################################
        
        print('Query no. %d' % (idx + 1))
        print('Training data shape', learner.X_training.shape)
        print('Unlabeled data shape', X_pool.shape)
        query_idx, query_instance = learner.query(X_u=X_pool, n_instances = nb_annotations)#,
#                                                  features_labeled = features_labeled, 
#                                                  features_unlabeled = features_unlabeled)
        
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=False,
            verbose=1, epochs = nb_active_epochs, batch_size = batch_size
        )
        # remove queried instance from pool
        print("patches annotated: ", X_pool[query_idx].shape)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        
        print('Training data shape after this query', learner.X_training.shape)
        print('Unlabeled data shape after this query', X_pool.shape)
        
        ####### testing ##############
        val_output = learner.evaluate(X_test, Y_test, verbose=1)
        
        for i in range(len(learner.model.metrics_names)):
            print(learner.model.metrics_names[i], val_output[i])

    
#    #============  Training ==================================
#    checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
#    
#    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
#    print(patches_imgs_train.shape, patches_masks_train.shape)
#    model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpointer])
#    
    
#    ========== Save and test the last model ===================
#    model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#    #test the model
#    score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
#    print('Test score:', score[0])
#    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()

















#
