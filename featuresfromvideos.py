import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
import sys, os
#from autocrop import crop_one
from pathlib import Path
import skvideo.utils
import skvideo.io


def importPreTrainedNet(name):
    """ Imports the model specified by name, with weights trained on imagenet.
    """
    if name == 'VGG16':
        model = applications.VGG16(include_top=False, weights='imagenet')
    elif name == 'MobileNet':
        # MobileNet needs its input shape defined.
        model = applications.MobileNet(include_top=False, weights='imagenet', input_shape=(128, 128, 3), pooling='avg')
    elif name == 'InceptionV3':
        model =   applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    else:
        print('failed to load model')
        sys.exit()
    print('Using ', name, " as base network.")
    return model

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
def extract_features(data_dir, model, crop=False):
    """ Extracts the features of all images present in the folder.
    data_dir should be something like data/positive/
    which in turn contains videos of all positive examples. From these
    features are extracted and saved into a npy file. The same can be done
    for the negative and positive test examples with their respective data folder
    path. Each "set" then will have its own features.
    """
    class_features = []
    #class_features = np.array(class_features)
    # Retrieve each file in directory.
    for file_name in os.listdir(data_dir + "videos/"):
        video_name = data_dir + "videos/" + file_name
        #if(crop==True):
        #    crop_one(video_name)
        videogenerator = skvideo.io.vreader(fname=video_name)
        video_features = []
        for frame in videogenerator:
            # Convert image to useable form.
            img = resize(frame, (128, 128))
            img = img.reshape((1,)+img.shape)
            features = model.predict([img])
            features = features.flatten()
            #print("feature shape", np.array(features).shape)
            video_features.append(features)
        print("features extracted, shape of feature list for video: ", np.array(video_features).shape)
        #print("features.shape", features.shape)
        video_name = video_name[:-4]
        np.save(open(data_dir + "features/" + file_name + "_feature.npy", 'wb'), video_features)
        class_features.extend(video_features)
        print("class_features.shape", np.array(class_features).shape)
        # Save features to file
    np.save(open(data_dir + "features.npy", 'wb'), class_features)



def extract_features_revamped(data_dir, model, crop=False):
    """ Extracts the features of all images present in the folder.
    data_dir should be something like data/positive/
    which in turn contains videos of all positive examples. From these
    features are extracted and saved into a npy file. The same can be done
    for the negative and positive test examples with their respective data folder
    path. Each "set" then will have its own features.
    """
    class_features = []
    #class_features = np.array(class_features)
    # Retrieve each file in directory.
    for cat_name in os.listdir(data_dir):
        if "py" in cat_name: continue
        cat_dir = data_dir + cat_name
        print("for category:", cat_dir, "\n")
        for vid_dir in os.listdir(cat_dir):
            if "npy" in vid_dir: continue
            print("for video:", vid_dir, "\n")
            dir_name = "{}/{}".format(cat_dir, vid_dir)
            for vid_file in os.listdir(dir_name):
                if "npy" in vid_file: continue
                vid_name = "{}/{}".format(dir_name, vid_file)
                videogenerator = skvideo.io.vreader(fname=vid_name)
                video_features = []
                try:
                    for frame in videogenerator:
                        img = resize(frame, (128, 128))
                        img = img.reshape((1,)+img.shape)
                        features = model.predict([img])
                        features = features.flatten()
                        #print("feature shape", np.array(features).shape)
                        video_features.append(features)
                except AssertionError:
                    print("AssertionError encountered, skipping video file.\n")
                    pass
                np.save(open("{}features.npy".format(dir_name), 'wb'), video_features)


model = importPreTrainedNet('MobileNet')
positive_dir = "entrance_inception/train/pos/class1/"
negative_dir = "entrance_inception/test/neg/"
test_positive = "entrance_inception/test/pos/class1/"
if __name__ == "__main__":
    """ With hardcoded paths to a single positive training class,
    a single positive test class, and the negative examples,
    extract features using the specified model.
    """
    #extract_features(positive_dir, model)
    #extract_features(negative_dir, model)
    #extract_features(test_positive, model)
    extract_features_revamped("/mnt/storage/Datasets/UCF_101_remaining/", model)

