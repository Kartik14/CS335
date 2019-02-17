import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"

        n = len(trainingData)
        p = [1.0/n for i in range(n)]
        indices = [i for i in range(n)]
        for l in range(self.num_classifiers):
            sampledIndices = np.random.choice(indices, int(self.ratio*n), True, p)
            sampledTrainingData = [trainingData[i] for i in sampledIndices]
            sampledTrainingLabels = [trainingLabels[i] for i in sampledIndices]
            self.classifiers[l].train(sampledTrainingData, sampledTrainingLabels)

        # util.raiseNotDefined()


    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        "*** YOUR CODE HERE ***"
        guesses = []
        for l in range(self.num_classifiers):
            guesses.append(self.classifiers[l].classify(data))  

        temp = zip(*guesses)
        final_guesses = [1 if sum(t) >= 0 else -1 for t in temp]

        return final_guesses
        # util.raiseNotDefined()
