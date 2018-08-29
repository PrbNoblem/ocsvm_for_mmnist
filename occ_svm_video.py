import numpy as np
from sklearn import svm
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# pickle for saving model
import pickle
from datetime import datetime
from metrics import calculate_accuracy, calculate_f_score, calculate_precision, calculate_recall
from sklearn import metrics



import queue
def classify_video(model, video_features, video_name=""):

    q = queue.LifoQueue(25)
    s = 0
    this_round = 0
    max_frames = 0
    for feature in video_features:
        f_p = model.predict(feature.reshape(1, -1))
        if (f_p == -1):
            this_round += 1
            if (this_round > max_frames):
                max_frames = this_round
        else :
            this_round = 0
        
        q.put(f_p)
        s += f_p
        if(q.full()):
            # Check if the queue is full of negative numbers.
            #if(s == -25):
            #    # The last 25 frames were all considered negative.
            #    print("PLS NO")
            # Remove the oldest element.
            s - q.get()
    print("Consecutive frames classified as negative in {}: {} ".format(video_name, max_frames))
    return max_frames

def load_features(path):
    return np.load(open(path, 'rb'))

from numpy import random
# Tunes the parameters for OneClassSVM, nu and gamma.
def tune_params(training_features, positive_test_features,
        negative_test_features, retrain=True, model_name='model2.sav'):
    """ Given training features of positive examples, positive and 
    negative test features, checks if the retrain argument is set to True
    or if the model described by model_name does not exist, and creates a 
    one class svm model. Parameter tuning is performed in a basic way,
    with the use of F1 score as the benchmark. Random samples are selected
    from the test sets to avoid inbalance. 
    If the retrain argument is set to False and the model exists, it is
    instead loaded from file. 
    """
    if (retrain == True or not Path(model_name).exists()):
        print("Fitting new model, model name:", model_name)
        print("Num training feats.", training_features.shape)
        print("Num pos test feats.", positive_test_features.shape)
        print("Num neg test feats.", negative_test_features.shape)
        max_batch_size = min([len(positive_test_features),len(negative_test_features),1500])
        print("Number of used:", max_batch_size, "\n")
        best_nu, best_g = 0, 0
        max_dist = 0
        prec = -1
        recall= 0
        t_p = positive_test_features[np.random.choice(positive_test_features.shape[0], max_batch_size, replace=False)]
        t_n = negative_test_features[np.random.choice(negative_test_features.shape[0], max_batch_size, replace=False)]
        if( "one" in model_name):
            print("features:",  len(t_p))
            print(t_p[0])
        f = 0
        Nus = [0.00126, 0.0025, 0.00375, 0.005, 0.675, 0.0075, 0.01, 0.05, 0.1]
        #Nus = [5*10**(-0.5*i) for i in reversed(range(4, 16))]
        gammas = [0.000001, 0.000005, 0.00001, 0.0001, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015]
        #gammas = [5*10**(-i) for i in reversed(range(4, 16))]
        best_model = None
        best_scores = (0, 0, 0)
        for nu in Nus:
            for g in gammas:
                model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=g)
                model.fit(training_features)
                
                # select random features for testing
                y_p = model.predict(t_p)
                y_n = model.predict(t_n)
                new_f = calculate_f_score(y_p, y_n)
                #if  new_f > f and sum(y_n) < 0:
                #print("new f score:", new_f, "y_p:", y_p, "y_n:", y_n)
                if  new_f > f:
                    f = new_f
                    best_scores = (calculate_precision(y_p, y_n), calculate_recall(y_p), f)
                    best_nu = nu
                    best_g = g
                    #print("for Nu", nu, "and gamma", g, "y_p:", sum(y_p), "and y_n:", sum(y_n), "f score was: ", f)
                    best_model = model
        pickle.dump(best_model, open(model_name, 'wb'))
    else:
        best_scores = (0, 0, 0)
        print("Loading saved model, model name:", model_name)
        best_model = pickle.load(open(model_name, 'rb'))

    return best_model, best_scores


def reduce_dimensions(feature_list, nbr_dimensions):
    """ Conducts a PCA to reduce the dimensions of 'feature_list' to 'nbr_dimensions'.
    """
    # Rescale
    scaler = StandardScaler()
    scaler.fit(feature_list)
    feature_list = scaler.transform(feature_list)
    # PCA
    pca = PCA(n_components=nbr_dimensions, whiten=True)
    pca.fit_transform(feature_list)
    feature_list = pca.transform(feature_list)
    return feature_list



def fit_one_class_svm(positive_features):
    """ Fits a one class svm to the positive features provided with no
    parameter tuning.
    """
    # Increasing 'gamma' reduces the size of the 'include' area, thus making
    # the classifier 'less picky'.
    model = svm.OneClassSVM(nu=1/(len(positive_features)),kernel='rbf',gamma=1/(len(positive_features[1])))
    #model = svm.OneClassSVM(kernel='polynomial')
    model.fit(positive_features)
    return model

# One predictions for each frame
def test(model, features):
    """ Performs a frame-by-frame classification and returns the
    result as a list. 
    """
    results = []
    for i in range(len(features)):
        single_feature = features[i]
        #print("single_feature.shape", single_feature.shape)
        single_feature = single_feature.reshape(1, -1)
        #print("single_feature.shape", single_feature.shape)
        res = model.predict(single_feature)
        results.append(res)
        #print("prediction", i, ":", res)

    return results

def test_on_set(model, features):
    """ Runs predictions on a set of features and returns the avaraged
    sum of the predictions, with predictions being either 1 (positive),
    or -1 (negative).
    """
    r = model.predict(features)
    return sum(r)/float(len(r))

# Try to plot the distances to the separations in the data given.
import matplotlib.pyplot as plt 
def plot_distances(model, data, predicted_classes,
     true_classes=None, name="", dirname="plots/default/"):
    """ For a given model and data to predict on, plot the distances
    to the hyperplane for each frame. Save the produced image into
    the provided dirname.
    """

    savename = dirname + "/" + name + "_" + ".png"
    
    distances = model.decision_function(data)
    max_distance = max(max(distances),abs(min(distances)))
    #norm_distances = [float(i)/max_distance for i in distances]
    norm_distances = distances
    title = "true positives" if ('pos' in name) else "true negatives"
    plt.title("Distances to hyperplane for " + title)
    plt.ylabel('Distance')
    #plt.ylim(-1, 1)
    plt.xlabel('Frame')
    #print("savename:", savename)
    plt.plot(norm_distances)
    plt.plot(predicted_classes, '.')
    plt.axhline(y=0, color='g', linestyle='-')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    plt.close()

    return distances



from sklearn.metrics import confusion_matrix 
import itertools
import seaborn as sn
def plot_confusion_matrix(pred_on_pos, pred_on_neg,
     name="", dirname="plots/default/"):
    """ Plot confusion matrix for tests. Should be run with both positive 
    and negative test data. One run for each is the easiest, which is why
    the data for the cases are provided to this method separetly. Thus the 
    true class of all the predictions in the first argument is 1 and the 
    class of all the preditions in the second argument is -1.
    """
    #print("dirname:", dirname, "name:", name)
    savename = dirname + "/" + name + "_" + ".png"

    y_pos = [1] * len(pred_on_pos)
    y_neg = [-1] * len(pred_on_neg)
    y = y_pos + y_neg
    all_preds = np.array(pred_on_pos + pred_on_neg).reshape(len(y),)

    cm = confusion_matrix(y, all_preds)
    print(cm)

    sn.set(font_scale=1.4)
    ax = sn.heatmap(cm, annot=True, annot_kws={"size":12}, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.title("Confusion matrix")
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_roc(mode, true_labels, values, name, dirname):
    """ Plots the ROC curve and calculates the area under the curve.
    Saves image in the directory specified by dirname.
    """
    savename = dirname + "/" + name + "_" + ".png"
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, values, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.text(-35, -70, "AUC: %.3f" % roc_auc, ha="right")
    print("\nArea under the ROC curve : %f" % roc_auc)
    plt.plot(fpr, tpr, color='darkorange')
    #plt.savefig(savename, bbox_inches='tight')
    plt.show()
    plt.close()

def evaluate_ucf():
    # For each iteration:
    # Dataset with features from UCF101:
    trainpath =  "/home/hannesj/Datasets/UCF_101_val_jpgs/"
    valpath = "/home/hannesj/Datasets/UCF_101_train_jpgs/"

    # Available categories
    categories = [i for i in os.listdir(trainpath) if 'py' not in i and\
        len(list(os.listdir("{}/{}".format(trainpath, i)))) >= 30*3]
    print(categories)
    results = dict()
    results_nbrs_only = dict()

    for _ in range(100):
        print("\n#############################################################\n")
        # Select random category:
        cat = random.choice(categories) 
        # Select a random scene as a positive class
        possible_scenes = [(i, ( len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ])))
            for i in range(30) if len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ])]
        random.shuffle(possible_scenes)
        rand_scene, num_examples = possible_scenes.pop()
        rand_scene = str(rand_scene) if rand_scene > 9 else "0" + str(rand_scene)
        #print("FIRST POSSIBLE SCENES", possible_scenes)

        # Get half (lower bounded) of examples of this scene
        pos_train_scenes = ["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, rand_scene, i)\
            for i in range(1, int(num_examples//2) + 1) if \
            os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, rand_scene, i))]
        # Get the rest of examples of this scene as positive val data
        pos_test_scenes = ["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, rand_scene, i)\
            for i in range(int(num_examples//2) + 1, num_examples + 1) if \
            os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, rand_scene, i))]

        #print("rand_scene!", rand_scene)

        possible_neg_scenes = [(i, ( len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ])))
            for i in range(30) if len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ]) and i != rand_scene]

        #print("possible neg test scenes:", possible_neg_scenes, "\n")
        
        neg_test_scenes = ["{}{}/v_{}_g{}_c{}.avi".format(trainpath, cat, cat, 
            possible_neg_scenes[i][0] if possible_neg_scenes[i][0] > 9 else "0" + str(possible_neg_scenes[i][0] ),
            possible_neg_scenes[i][1] if possible_neg_scenes[i][1] > 9 else "0" + str(possible_neg_scenes[i][1] )  )\
            for i  in [random.choice(range(len(possible_neg_scenes))) for i in range( len(pos_test_scenes) + 1 )]\
            if  os.path.exists("{}{}/v_{}_g{}_c{}.avi".format(trainpath, cat, cat, 
            possible_neg_scenes[i][0] if possible_neg_scenes[i][0] > 9 else "0" + str(possible_neg_scenes[i][0] ),
            possible_neg_scenes[i][1] if possible_neg_scenes[i][1] > 9 else "0" + str(possible_neg_scenes[i][1] )  ))]

        # Get negative testing examples from other category
        #       print("First category:", cat)
        pos_cat = cat
        cat = random.choice(categories)
        #print("Second category:", cat)

        possible_neg_scenes2 = [(i, ( len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ])))
            for i in range(30) if len([j for j in os.listdir(trainpath + cat) \
            if int(j.split('_')[2][1:]) == i ])]
            
        diff_neg_test = [str(i[0]) if i[0] > 9 else "0" + str(i[0]) for i in possible_neg_scenes2]
        indexes = [i for i in range(len(possible_neg_scenes2))] 
        random.shuffle(indexes) 

        #print("diff_neg_test", len(possible_neg_scenes2))
        a = [ f"{trainpath}{cat}/v_{cat}_g{possible_neg_scenes2[i][0]}_c0{possible_neg_scenes2[i][1]}.avi" \
            for i in indexes ]

        neg_test_scenes.extend(["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, 
            possible_neg_scenes2[i][0], possible_neg_scenes2[i][1])\
            for i in [random.choice(range(len(diff_neg_test))) for i in range( len(pos_test_scenes) + 1 )]\
            if os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, 
            possible_neg_scenes2[i][0], possible_neg_scenes2[i][1])) ])
        

        get_random_cat_stuff(neg_test_scenes, categories, pos_test_scenes, trainpath)

        print("pos train:", pos_train_scenes, "\n")
        print("pos test:", pos_test_scenes, "\n")
        print("neg test:", neg_test_scenes, "\n")

        

        # Get all the features:
        pos_train_features = []
        pos_test_features = []
        neg_test_features = []
        for i in range(len(pos_train_scenes)):
            try:
                feature = np.load(open(pos_train_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", pos_train_scenes[i], "\n" )
            pos_train_features.extend(feature)

        for i in range(len(pos_test_scenes)):
            try:
                feature = np.load(open(pos_test_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", pos_test_scenes[i], "\n" )
            pos_test_features.extend(feature)

        for i in range(len(neg_test_scenes)):
            try:
                feature = np.load(open(neg_test_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", neg_test_scenes[i], "\n" )
            neg_test_features.extend(feature)

        pos_train_features = np.array(pos_train_features)
        pos_test_features = np.array(pos_test_features)
        neg_test_features = np.array(neg_test_features)
        

        if pos_train_features.shape[0] == 0 or pos_test_features.shape[0] == 0 or neg_test_features.shape[0] == 0:
            continue
        print(pos_train_features.shape)
        print(pos_test_features.shape)
        print(neg_test_features.shape)

        model, scores = tune_params(pos_train_features, pos_test_features, neg_test_features,
            model_name="testing_multistuff.sav")

        print("\n scores:", scores ,"\n")

        result_string = f"Positive training data: {', '.join(pos_train_scenes)}\nPositive testing data: {', '.join(pos_test_scenes)}\nNegative testing data: {', '.join(neg_test_scenes)}\nPrecision: {scores[0]}\nRecall: {scores[1]}\n F1 score: {scores[2]}\n"
        
        results[pos_cat] = result_string
        key = f"{pos_cat}_g{rand_scene}"
        results_nbrs_only[key] = scores
    
    return results, results_nbrs_only


def get_random_cat_stuff(examples, categories, pos_test_scenes, trainpath):
    cat = random.choice(categories)
    #print("Second category:", cat)

    possible_neg_scenes2 = [(i, ( len([j for j in os.listdir(trainpath + cat) \
        if int(j.split('_')[2][1:]) == i ])))
        for i in range(30) if len([j for j in os.listdir(trainpath + cat) \
        if int(j.split('_')[2][1:]) == i ]) and i != pos_test_scenes]
        
    diff_neg_test = [str(i[0]) if i[0] > 9 else "0" + str(i[0]) for i in possible_neg_scenes2]
    indexes = [i for i in range(len(possible_neg_scenes2))] 
    random.shuffle(indexes) 

    #print("diff_neg_test", len(possible_neg_scenes2))
    a = [ f"{trainpath}{cat}/v_{cat}_g{possible_neg_scenes2[i][0]}_c0{possible_neg_scenes2[i][1]}.avi" \
        for i in indexes ]

    examples.extend(["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, 
        possible_neg_scenes2[i][0], possible_neg_scenes2[i][1])\
        for i in [random.choice(range(len(diff_neg_test))) for i in range( len(pos_test_scenes) + 1 )]\
        if os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, cat, cat, 
        possible_neg_scenes2[i][0], possible_neg_scenes2[i][1])) ])


    return examples

def get_sim_cat_stuff(examples, category, pos_scene, trainpath):

    possible_neg_scenes = [(i, ( len([j for j in os.listdir(trainpath +"/"+ category) \
            if int(j.split('_')[2][1:]) == i ])))
            for i in range(30) if len([j for j in os.listdir(trainpath +"/"+ category) \
            if int(j.split('_')[2][1:]) == i ]) and i != pos_scene]

    neg_test_scenes = ["{}{}/v_{}_g{}_c{}.avi".format(trainpath, category, category, 
            possible_neg_scenes[i][0] if possible_neg_scenes[i][0] > 9 else "0" + str(possible_neg_scenes[i][0] ),
            possible_neg_scenes[i][1] if possible_neg_scenes[i][1] > 9 else "0" + str(possible_neg_scenes[i][1] )  )\
            for i  in [random.choice(range(len(possible_neg_scenes))) for i in range( 1 + 1 )]\
            if  os.path.exists("{}{}/v_{}_g{}_c{}.avi".format(trainpath, category, category, 
            possible_neg_scenes[i][0] if possible_neg_scenes[i][0] > 9 else "0" + str(possible_neg_scenes[i][0] ),
            possible_neg_scenes[i][1] if possible_neg_scenes[i][1] > 9 else "0" + str(possible_neg_scenes[i][1] )  ))]

    #print(neg_test_scenes)
    examples.extend(neg_test_scenes)
    return examples

def variable_amount_training(category, scene):
    """ Supply category and scene as strings, fix everything else.
    " Give on format CategoryName, gXX
    """
    valpath = "/home/hannesj/Datasets/UCF_101_val_jpgs/"
    trainpath = "/home/hannesj/Datasets/UCF_101_train_jpgs/"

    categories = [i for i in os.listdir(trainpath) if 'py' not in i and\
        len(list(os.listdir("{}/{}".format(trainpath, i)))) >= 30*3]

    i_scene = int(scene[1:])
    print("scene:", scene)
    # Possible scenes should only be the one designated


    examples = [i for i in os.listdir(trainpath + category) if int(i.split('_')[2][1:]) == i_scene   ]

    print("examples:", examples, "\n")

    for j in range(1, int(len(examples)//2) + 1):
        # Get j number of training examples.
        pos_train_scenes = ["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, category, category, scene[1:], k)\
            for k in range(1, j+1) if \
            os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, category, category, scene[1:], k))]
        # Get the rest of examples of this scene as positive val data
        pos_test_scenes = ["{}{}/v_{}_g{}_c0{}.avi".format(trainpath, category, category, scene[1:], k)\
            for k in range(j + 1, len(examples) + 1 ) if \
            os.path.exists("{}{}/v_{}_g{}_c0{}.avi".format(trainpath, category, category, scene[1:], k))]

        print(f"With {j} training examples of scene {scene} from category {category}:\n")

        neg_test_scenes = []
        print('AAAAAAAAAAAaa')
        
        for _ in range(2):
            neg_test_scenes.extend(get_sim_cat_stuff(neg_test_scenes, category, scene, trainpath))
        for _ in range(2):
            neg_test_scenes.extend(get_random_cat_stuff(neg_test_scenes, categories, pos_test_scenes, trainpath))

        pos_train_features = []
        pos_test_features = []
        neg_test_features = []
        for i in range(len(pos_train_scenes)):
            try:
                feature = np.load(open(pos_train_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", pos_train_scenes[i], "\n" )
            pos_train_features.extend(feature)

        for i in range(len(pos_test_scenes)):
            try:
                feature = np.load(open(pos_test_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", pos_test_scenes[i], "\n" )
            pos_test_features.extend(feature)

        for i in range(len(neg_test_scenes)):
            try:
                feature = np.load(open(neg_test_scenes[i] + "/features.npy",'rb'))
            except:
                print("Failed to load file in:", neg_test_scenes[i], "\n" )
            neg_test_features.extend(feature)

        pos_train_features = np.array(pos_train_features)
        pos_test_features = np.array(pos_test_features)
        neg_test_features = np.array(neg_test_features)
    
        model, scores = tune_params(pos_train_features, pos_test_features, neg_test_features,
            model_name="testing_multistuff.sav")
        if model is None: continue
        # Check some videos with this model!
        
        frames_in_pos = 0
        frames_in_neg = 0
        corr_pos = 0
        corr_neg = 0
        print("\n classifying positive test examples:")
        for i in range(len(pos_test_scenes)):
            name = pos_test_scenes[i]
            features = np.load(open(name + "/features.npy",'rb'))
            frames = classify_video(model, features, name)
            frames_in_pos += frames
            if frames < 40: corr_pos += 1

        avg_neg_in_pos = frames_in_pos / len(pos_test_scenes)
        
        print("\n classifying negative test examples:")
        # make sure 'hard' negatives are included.
        for i in range(12):
            rand_nbr = random.choice(range(len(neg_test_scenes)))
            name = neg_test_scenes[rand_nbr]
            features = np.load(open(name + "/features.npy",'rb'))
            frames = classify_video(model, features, name)
            frames_in_neg += frames
            if frames >= 40: corr_neg += 1
        
        avg_neg_in_neg = frames_in_neg / 12
        accuracy = (corr_pos + corr_neg)/(12 + len(pos_test_scenes))
        print("\n scores:", scores ,"\n")

        result_string = f"Positive training data: {', '.join(pos_train_scenes)}\nPositive testing data: {', '.join(pos_test_scenes)}\nNegative testing data: {', '.join(neg_test_scenes)}\nPrecision: {scores[0]}\nRecall: {scores[1]}\n F1 score: {scores[2]}\n"
        
    #model, scores = tune_params()


    return avg_neg_in_pos, avg_neg_in_neg, accuracy


    
    
    #test_cases = [('JavelinThrow', 'g06'), ('PlayingPiano', 'g24'), ('Rowing', 'g25'), ('HulaHoop', 'g21'),
    #   ('PizzaTossing', 'g19'), ('CricketShot', 'g05'), ('SalsaSpin', 'g01')]
    scenes = ["CricketShot_g03",
    "Shotput_g10",  
    "IceDancing_g15",  
    "Rowing_g10",  
    "PizzaTossing_g06",  
    "Diving_g23",  
    "BandMarching_g04",  
    "PizzaTossing_g25",  
    "JugglingBalls_g08",  
    "ShavingBeard_g09",  
    "IceDancing_g08",  
    "Diving_g01",  
    "SalsaSpin_g02",  
    "Rowing_g06",  
    "JumpRope_g22",  
    "CleanAndJerk_g10",  
    "IceDancing_g09",  
    "JugglingBalls_g24",  
    "JumpingJack_g02",  
    "JavelinThrow_g12",  
    "JumpingJack_g06",  
    "GolfSwing_g18",  
    "Shotput_g14",  
    "JugglingBalls_g21",  
    "JumpRope_g14",  
    "JumpRope_g12",  
    "HulaHoop_g08",  
    "Diving_g24",  
    "WritingOnBoard_g11",  
    "SalsaSpin_g24",  
    "IceDancing_g16",  
    "JumpRope_g17",  
    "IceDancing_g03",  
    "IceDancing_g01",  
    "PizzaTossing_g10",  
    "Diving_g15",  
    "SkateBoarding_g07",  
    "ShavingBeard_g20",  
    "SkateBoarding_g17",  
    "CricketShot_g22",  
    "BandMarching_g23",  
    "JumpRope_g03",  
    "Rowing_g17",  
    "PlayingPiano_g25",  
    "JumpRope_g08",  
    "JumpRope_g11",  
    "CricketBowling_g22",  
    "SkateBoarding_g11",  
    "BandMarching_g19",  
    "JumpingJack_g15"]  

    averages = dict()
    test_cases = [ (i.split('_')[0], i.split('_')[1]) for i in scenes  ]
    for case in test_cases:
        print("#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#\n")
        a1, a2, acc = variable_amount_training(case[0], case[1])
        print("#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#\n")
        name = case[0] + "_" +  case[1]
        averages[name] = (a1, a2, acc)
    #m, a1, a2, a3= variable_amount_training("Mixing", "g05")
    #averages['Mixing_g05'] = (a1, a2, a3)

    for name in averages.keys():
        print(f"Training on vid {name}, avarage neg. frames in pos vid: {averages[name][0]}, in neg vid: {averages[name][1]}, accuracy: {averages[name][2]}\n")


    exit()


    print("Using one-class SVM")
    positive_training_features = load_features("data2/positive/features.npy")
    positive_test_features = load_features("data2/testing/positive/features.npy")
    negative_features = load_features("data2/negative/features.npy")
    print("number of positive training features:", len(positive_training_features))
    print("number of positive test features:", len(positive_test_features))

    print("Setting upp One-Class SVM model")
    print(positive_training_features.shape)
    #model = fit_one_class_svm(positive_training_features)
    model, results = tune_params(positive_training_features,
     positive_test_features, negative_features, retrain=False)

    print("Model set up comleted.")

    # Making predictions on the already seen training data for confirmation.
    positives_found_training = test(model, positive_training_features)
    positives_found_training = [i for i in positives_found_training if i == 1]

    # Making predictions on the positive examples. Should produce as many 'ones'
    # as possible. 
    max_batch_size = min([len(positive_test_features),len(negative_features),1000])

    r = np.random.choice(positive_test_features.shape[0], max_batch_size, replace=False)
    positives_found_test_positives = test(model, positive_test_features[r])
    tmp = positives_found_test_positives
    positives_found_test_positives = [i for i in positives_found_test_positives if i == 1]

    # Making predictions on the negative examples. Should produce as few 'ones'
    # as possible.
    positives_found_test_negatives = test(model, negative_features[r])
    tmp2 = positives_found_test_negatives
    positives_found_test_negatives = [i for i in positives_found_test_negatives if i == 1]

    corr_pos = sum(positives_found_test_positives)
    incorr_pos = sum(positives_found_test_negatives)
    # Printing some metrics 
    test_precision = corr_pos / (corr_pos + incorr_pos)
    test_recall = corr_pos / len(positive_test_features)
    test_f1 = (2*test_precision*test_recall)/(test_recall + test_precision)
    print("Precision: ", test_precision)
    print("Recall: ", test_recall)
    print("F-score: ", test_f1)
    print("Accuracy: ", (corr_pos + (max_batch_size - incorr_pos))/ (2 * max_batch_size) )
    print("positives found in positive_test_set:", sum(positives_found_test_positives), "out of ", max_batch_size)
    print("positives found in negative_test_set:", sum(positives_found_test_negatives), "out of ", max_batch_size)

    # Trying to plot stuff
    fake_y = [1] * max_batch_size
    plot_distances(model, positive_test_features, tmp, fake_y)

    fake_y = [-1] * max_batch_size
    plot_distances(model, negative_features, tmp2, fake_y)