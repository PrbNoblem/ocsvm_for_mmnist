from skimage.io import imread
import os, argparse, sys
from PIL import Image
import numpy as np
from occ_svm_video import (tune_params, plot_roc, 
    plot_confusion_matrix, classify_video, test)

from multi_model_master import run_tests


# TODO
# test out pixel data then run the tests and be done with it



def get_pixel_features(parent_dir="pixel_data"):
    """ Needs to return three things:
    * (Positive) Training features, being a dict mapping names
    of classes to features, with features being on the format 
    (nbr_features, feature_len). 
    * Positive testing features on the same format as training features.
    * Negative training features on the format (nbr_features, feature_len)
    To enable coupling with the name of the 'examples' the return format is instead
    {class_name : [(pixels, ex_name)]} or [(pixels, ex_name)]
    """
    train = dict()
    pos_test = dict()

    # Get positive training features
    
    for class_name in os.listdir(os.path.join(parent_dir, "train", "positive")):
        print("Getting class:", class_name)
        class_path = f"{parent_dir}/train/positive/{class_name}"
        pixels = []
        for example in os.listdir(class_path):
            print("Example name:", example)
            example_path = os.path.join(class_path, example)    
            for img in os.listdir(example_path):
                im = Image.open(f"{example_path}/{img}")
                im = im.resize((28,28))
                im_array = np.asarray(im).reshape((784))
                print(f"im_array shape: {im_array.shape}")
                pixels.append((im_array, example))
        train[class_name] = pixels

    # Get positive testing features
    for class_name in os.listdir(os.path.join(parent_dir, "test", "positive")):
        print("Getting class:", class_name)
        class_path = f"{parent_dir}/test/positive/{class_name}"
        pixels = []
        for example in os.listdir(class_path):
            print("Example name:", example)
            example_path = os.path.join(class_path, example)    
            for img in os.listdir(example_path):
                im = Image.open(f"{example_path}/{img}")
                im = im.resize((28,28))
                im_array = np.asarray(im).reshape((784))
                print(f"im_array shape: {im_array.shape}")
                pixels.append((im_array, example))
        pos_test[class_name] = pixels


    # Get negative testing features
    neg_path = os.path.join(parent_dir, 'test', 'negative')
    pixels = []
    for example in os.listdir(neg_path):
        print("Example name:", example)
        example_path = os.path.join(neg_path, example)    
        for img in os.listdir(example_path):
            im = Image.open(f"{example_path}/{img}")
            im = im.resize((28,28))
            im_array = np.asarray(im).reshape((784))
            print(f"im_array shape: {im_array.shape}")
            pixels.append((im_array, example))

    return train, pos_test, pixels



def check_args(args):
    """ Parse arguments."""
    parser = argparse.ArgumentParser(description="Feed me data!")
    parser.add_argument('-parent_dir', default="pixel_data/",
        help='Enter -parent_dir datafolder/ to specify parent folder of data.')
    parser.add_argument('-t', default="False",
        help='-t True | False to specify if models should be retraind.')
    parser.add_argument('-n', default="",
        help='-n to specify a base for the name of the models.')

    results = parser.parse_args(args)
    return results.parent_dir, results.t, results.n


if __name__ == "__main__":
    parent_dir, retrain, model_name = check_args(sys.argv[1:])
    num_pos_classes = len(os.listdir(
        os.path.join(parent_dir, "train", "positive")))


    # Load features for training/ testing
    pos_train_features, pos_test_features, neg_test_features = \
        get_pixel_features(parent_dir)

    # Set up models, retrain if necessary
    

    pos_train_pixels = dict()
    for k, v in pos_train_features.items():
        pixels_for_class = []
        for t in v:
            pixels_for_class.append(t[0])
        pos_train_pixels[k] = np.array(pixels_for_class)

    pos_test_pixels = dict()
    for k, v in pos_test_features.items():
        pixels_for_class = []
        for t in v:
            pixels_for_class.append(t[0])
        pos_test_pixels[k] = np.array(pixels_for_class)

    neg_test_pixels = []
    for t in v:
        neg_test_pixels.append(t[0])
    neg_test_pixels = np.array(neg_test_pixels)


    models = dict()

    for name in pos_train_features:
        if (retrain=="True"):
            models[name], scores = tune_params(pos_train_pixels[name], pos_test_pixels[name],
             neg_test_pixels, retrain=True, model_name=parent_dir[:-1] + model_name +name + '.sav')
        else:
            models[name], scores = tune_params(pos_train_pixels[name], pos_test_pixels[name],
             neg_test_pixels, retrain=False, model_name=parent_dir[:-1] + model_name +name + '.sav')
        
    for k, v in models.items():
        print("Model name: ", k, "type(model):", type(v))

    run_tests(models, pos_test_pixels, neg_test_pixels)
    #get_consecutive_frames(models, test_pos_path, test_neg_path)