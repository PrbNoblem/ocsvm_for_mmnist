from metrics import (calculate_precision, calculate_recall,
    calculate_f_score)
from occ_svm_video import (tune_params, test, load_features,
    plot_distances, plot_confusion_matrix, plot_roc)
from featuresfromvideos import (importPreTrainedNet, extract_features)
import argparse , os, sys
from datetime import datetime
import numpy as np


# Feature loading
def load_feats(path):
    """ Loads features from the given parent directory into three sets;
    The positive training examples divided into their classes, contained in a
    dict, the positive test examples stored in the same manner, and the negative
    examples which will be the same for all positive classes.
    """
    train_pos_path = path + "train/pos/" # Will contain class1 .. classN
    test_pos_path = path + "test/pos/"  
    test_neg_path = path + "test/neg/"
    train_p, test_p= dict(), dict()
    for class_n in os.listdir(train_pos_path):
        train_p[class_n] = load_features(train_pos_path + class_n + "/features.npy")
        print("loading: ",train_pos_path + class_n + "/features.npy")
        print("train shape: ", train_p[class_n].shape)
    for class_n in os.listdir(test_pos_path):
        test_p[class_n] = load_features(test_pos_path + class_n + "/features.npy")
        print("loading: ",test_pos_path + class_n + "/features.npy")
        print("test shape: ", test_p[class_n].shape)
    test_n = load_features(test_neg_path + "/features.npy")

    #print("train_p, test_p, test_n", train_p.shape, test_p.shape, np.array(test_n).shape)
    return train_p, test_p, test_n

def get_consecutive_frames(models, pos_train_path, pos_test_path, neg_test_path):
    

# Prediction / testing
def run_tests(models, pos_test_features, neg_test_features):
    """ Provided a dict of models {name:model}, a dict of positives
    {class:{video:features}} and negatives {video:features}, run
    various test and produce metrics such as ploting distances, creating
    confusion matrices etc. 
    """            
    now = datetime.now()
    for name in pos_test_features:
        print("testing for class:", name)
        tmp = pos_test_features[name]
        max_batch_size = min([len(tmp),len(neg_test_features),1000])

        p_t = tmp[np.random.choice(tmp.shape[0], max_batch_size, replace=False)]
        n_t = neg_test_features[np.random.choice(neg_test_features.shape[0], max_batch_size, replace=False)]
        test_dir_name ="plots/" + str(now.isoformat())
        if not os.path.isdir(test_dir_name):
            os.mkdir(test_dir_name)
        model = models[name]
        pred_on_pos = test(model, p_t)
        pred_on_neg = test(model, n_t)
        plot_confusion_matrix(pred_on_pos, pred_on_neg, name=name
         + "conf_mat", dirname=test_dir_name)
        plot_distances(model, p_t, pred_on_pos,
         name=(name + "_pos"), dirname=test_dir_name)
        plot_distances(model, n_t, pred_on_neg,
         name=(name +"_neg"), dirname=test_dir_name)
        
        #data = pos_test_features[name] + neg_test_features
        data_pos = np.array(p_t)
        data_neg = np.array(n_t)
        data = np.concatenate((data_pos, data_neg))



        # Getting other metrics:
        print("Precision:", calculate_precision(pred_on_pos, pred_on_neg))
        print("Recall:", calculate_recall(pred_on_pos))
        print("F score:", calculate_f_score(pred_on_pos, pred_on_neg))

        all_distances = model.decision_function(data)
        labels_pos = np.array([1 for i in range(len(p_t))])
        labels_neg = np.array([-1 for i in range(len(n_t))])
        labels = np.concatenate((labels_pos, labels_neg))
        #print(labels_pos.shape, " ", labels_neg.shape)
        #print("all_distances:", all_distances.shape)
        #print("labels shape:", labels.shape)
        plot_roc(model, labels, all_distances, name + "AU_ROC", test_dir_name)

def check_args(args):
    """ Parse arguments."""
    parser = argparse.ArgumentParser(description="Feed me data!")
    parser.add_argument('-parent_dir', default="multi_classes2/",
        help='Enter -parent_dir datafolder/ to specify parent folder of data.')
    parser.add_argument('-t', default="False",
        help='-t True | False to specify if models should be retraind.')
    parser.add_argument('-e', default="False",
        help='-e to specify if features should be re-extracted.')
    parser.add_argument('-n', default="",
        help='-n to specify a base for the name of the models.')

    results = parser.parse_args(args)
    return results.parent_dir, results.t, results.e, results.n


if __name__ == "__main__":
    parent_dir, retrain, extract, model_name = check_args(sys.argv[1:])
    train_pos_path = parent_dir + "train/pos/" # Will contain class1 .. classN.
    test_pos_path = parent_dir + "test/pos/"   # Will contain class1 .. classN.
    test_neg_path = parent_dir + "test/neg/"   # All negative classes included.
    num_pos_classes = len(os.listdir(train_pos_path))

    # Import pre-trained network, extract features.
    if (extract == "True"):
        print("Extracting features")
        model = importPreTrainedNet("MobileNet")
        # Extract features for positive training examples.
        for pos_class in os.listdir(train_pos_path):
            extract_features(train_pos_path + pos_class + "/", model, crop=False)
        # Extract features for positive test examples.
        for pos_class in os.listdir(test_pos_path):
            extract_features(test_pos_path + pos_class + "/", model, crop=False)
        # Extract features for negative test examples.
        extract_features(test_neg_path + "/", model, crop=False)

    # Load features for training/ testing
    pos_train_features, pos_test_features, neg_test_features = \
     load_feats(parent_dir)
    #print("pos_train_features ", pos_train_features)

    # Set up models, retrain if necessary
    models = dict()
    for name in pos_train_features:
        if (retrain=="True"):
            models[name], scores = tune_params(pos_train_features[name], pos_test_features[name],
             neg_test_features, retrain=True, model_name=parent_dir[:-1] + model_name +name + '.sav')
        else:
            models[name], scores = tune_params(pos_train_features[name], pos_test_features[name],
             neg_test_features, retrain=False, model_name=parent_dir[:-1] + model_name +name + '.sav')
        
    for k, v in models.items():
        print("Model name: ", k, "type(model):", type(v))

    run_tests(models, pos_test_features, neg_test_features)
    

    # Gather test data from all classes. For all classes, run tests and
    # calculate the scores, plot distances and confusion matrices.


# CAMERA IPS
# IP: 172.25.74.3
# IP: 172.25.78.9