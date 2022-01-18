import logging
import abc
from src.mask_processing.assignment_finders import AssignmentFinderInterface
import numpy as np
import time
import sys

# for k-neighbours classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


# TODO: how to save classifiers
class Classification(AssignmentFinderInterface):
    """
       Class to find assignments by classification.
       Has an instance of the ClfAlgoInterface which performs the classification;
       then self does the actual assignment to self.annotations.
    """
    def __init__(self, dataset, *args, **kwargs):
        """
        Sets up a classifier object with given data and algorithm.
        :param dataset: instance of Dataset
        """
        # alternatively, __init__ could be without the algo parameter and could ask the user.
        super(Classification, self).__init__(dataset, *args, **kwargs)
        self.algo = KNNClf(dataset)

    # Todo: clean up args in prepare and find_assignment
    def prepare(self, training_frames, *args, **kwargs):
        """
        Trains the classifier.
        :param training_frames: frames to be used as ground truth for classifier training
        """
        training_segments, training_neurites = self.annotations.get_segs_and_assignments(training_frames)
        real_neurites = self.annotations.real_neurites #MB: real neutit property is in h5data. it is a list from 1 to nb_neurons
        self.algo.train(training_segments, training_neurites, real_neurites,  verbose=True, *args, **kwargs)#MB added verbose=True

    def find_assignment(self, frames, *args, **kwargs):
        """
        Runs the algorithms to find assignments for the segments of frames, and fills self.annotations with resulting assignments.
        :param frames: which frames to find assignments for   # todo: could be string, like 'all'
        """
        assignments = self.algo.classify(frames)
        self.annotations.assign(assignments,update_nb_neurons=True)

class ClfAlgoInterface:
    """
    Interface for different classification algorithms, invisible to main controller.
    Subclasses are only instantiated by Classification.
    """
    # Todo: classification could be called in __init__, or by the __init__ of Classification?
    def __init__(self, features):
        """
        :param features: instance of FeatureData
        """
        self.features = features

    @abc.abstractmethod
    def train(self, training_segments, training_neurites, neurites, verbose=True,*args, **kwargs):
        """
        Trains the classifier.
        :param training_segments: list of (t, s)
        :param training_neurites: list of corresponding assigned neurites
        :param neurites: list of neurites of interest
        """
        raise NotImplementedError

    @abc.abstractmethod
    def classify(self, frames):
        """
        Classify and return assignments.
        :param frames: which frames to classify
        :return: dictionary (t, s) -> neurite of neurite assigned to each segment
        """
        raise NotImplementedError


class KNNClf(ClfAlgoInterface):
    """k-nearest neighbours classification
    Train an ML model (k-neighbours classifier) to re-classify frames, learning from hand-corrected frames.
    Train one classifier per neurite of interest."""
    def __init__(self, features):
        super(KNNClf, self).__init__(features)
        self.logger = logging.getLogger("kNNclf")

    @ignore_warnings(category=UndefinedMetricWarning)
    def train(self, training_segments, training_neurites, neurites, verbose=False, *args, **kwargs):
        """
        Trains one k-nn classifier per neurite. For each neurite:
            - run a small grid search with 5-fold cross-validation for best number of neighbours
            - fit a classifier with best number of neighbours
        Stores the resulting classifiers in a dict self.classifiers: neu -> (trained classifier, precision of the classifier)
        :param training_segments: list of (t, s)
        :param training_neurites: list of corresponding assigned neurites
        :param neurites: list of neurites of interest
        :param verbose: whether to display meta-parameter fitting data by logging as info (otherwise, log as debug)
        """
        self.logger.debug(sys._getframe().f_code.co_name +
                          "with training_segments {} and real neurites {}".format(training_segments, neurites))
        self.neurites = neurites
        self.classifiers = {}
        X_train, segMB  = self.get_scaled_features(training_segments,segments = 1)#MB added seg_list to extract only the features
        print("X_train shape")
        print(X_train.shape) #MB added
        ys = [np.array(training_neurites) == neu for neu in neurites]#MB: it works like a  set of functions(for each neu one function) on matrix of features(data matrix)
        #print(ys)
        w = 'distance'
        algo = 'ball_tree'
        ls = 20
        best_nns = []
        best_accuracies = []
        start = time.time()

        for neu in range(len(self.neurites)):   # for each neurite...
            best = 0
            for nn in np.logspace(np.log10(2), np.log10(25), 7):   # ... small grid search on number of neighbours ...
                nn = int(nn)
                if nn > len(training_neurites):
                    continue
                scores = []
                precisions = []
                for fold in range(5):   # ... with 5-fold cross validation
                    knc = KNeighborsClassifier(n_neighbors=nn, algorithm=algo, weights=w, leaf_size=ls)
                    X_tr, X_te, y_tr, y_te = train_test_split(X_train, ys[neu],test_size=0.2, random_state=fold)#divide the the data ((t,s) sets, actually theircorresponding features and neurit label)
                    knc.fit(X_tr, y_tr.ravel())#MB changed y_tr to y_tr.ravel()
                    y_pre = knc.predict(X_te)
                    scores.append(balanced_accuracy_score(y_te, y_pre))# MB This line causes a warning: 'y_pred contains classes not in y_true'--This can be caused because y_te does not incluse some neurits
                    precisions.append(precision_score(y_te, y_pre))

                m = np.mean(scores)
                if m > best:
                    best = m
                    best_nn = nn
                    best_precision = np.mean(precisions)

            # now use best params to build the classifier for this neurite
            knc = KNeighborsClassifier(n_neighbors=best_nn, algorithm=algo, weights=w, leaf_size=ls)

            knc.fit(X_train, ys[neu].ravel())#MB changed ys[neu] to ys[neu].ravel()

            self.classifiers[self.neurites[neu]] = (knc, best_precision)

            best_nns.append(best_nn)

            best_accuracies.append(best)
        print("Classification training finished")
        #MB: Logger is used for tracking events that occur
        if verbose:
            log_fun = self.logger.info
        else:
            log_fun = self.logger.debug
        log_fun('Total time for hyperparameter optimization: {}'.format(time.time() - start))
        log_fun('Best nb_neighbours: {}'.format(best_nns))
        log_fun('Accuracy on test set: {}'.format(np.mean(best_accuracies)))
        best_sep_disp = {self.neurites[i]: best_accuracies[i] for i in
                         range(len(self.neurites))}
        log_fun('Details of accuracy: {}'.format(best_sep_disp))

    def classify(self, frames, verbose=True):
        """
        :param frames: which frames to classify
        :param verbose: whether to display information such as number of noise segments and disagreeing classifiers
        :return y_pred_allneurites: list of neurites to be assigned to corresponding segments in segments
            (i.e. segment segments[i] is to be assigned to neurite y_pred_allneurites[i]
        """
        self.logger.debug(sys._getframe().f_code.co_name)
        if verbose:
            log_fun = self.logger.info
        else:
            log_fun = self.logger.debug
        X_pred, segments = self.get_scaled_features(frames, times=1)#MB add times in arguments
        y_pred_allneurites = [None for _ in segments]#MB:the array including the neurite assigned to each (t,s) with initial value of None
        for neu in self.neurites:
            knc, precision = self.classifiers[neu]
            y_pred = knc.predict(X_pred)#classifier applied to the set of features
            for seg_idx in range(len(segments)):
                if y_pred[seg_idx]:#if the classifiers output value for this neu is 1:
                    if y_pred_allneurites[seg_idx] is None:#if the segment corresponding to seg_idx is not already assigned a neu
                        y_pred_allneurites[seg_idx] = neu
                    else:
                        log_fun(
                            "seg {} assigned to both {} and {}.".format(segments[seg_idx], y_pred_allneurites[seg_idx], neu))
                        # change only if this accuracy is better than previous assign
                        if precision > self.classifiers[y_pred_allneurites[seg_idx]][1]:
                            log_fun('Reassigning')
                            y_pred_allneurites[seg_idx] = neu
                        else:
                            log_fun('Not reassigning')
        nb_unassigned = sum(np.array(y_pred_allneurites) == None)
        print("Number of unassigned neurits:")#MB added
        print(nb_unassigned)
        print("These segments are assigned background values")#MB added
        log_fun('Noise: {} segments found'.format(nb_unassigned))
        log_fun('Assigned: {} segments found'.format(len(y_pred_allneurites) - nb_unassigned))
        for i in range(len(y_pred_allneurites)):#MB added to avoid unassigned arrors
            if y_pred_allneurites[i]==None:
                y_pred_allneurites[i] = 0#MB added to prevent erro
        return dict(zip(segments, y_pred_allneurites))

    def get_scaled_features(self, frames, segments = None, times = None):#extra arguments was added by MB for classification purpose
        """Gets the normalized array of features for given frames, and the list of (t,s) segment corresponding to each line in the array."""
        if segments is not None:#MB added, because the get_scaled is called by different functions with different format of frames argument
            features, segs_list = self.features.feature_array(segments=frames, segs_list=True)#MB changed time to segments
        else:
            features, segs_list = self.features.feature_array(times=frames, segs_list=True)
        #features, segs_list = self.features.feature_array(times=frames, segs_list=True)#MB removed
        scaled_ftrs = scale(features)
        return scaled_ftrs, segs_list
