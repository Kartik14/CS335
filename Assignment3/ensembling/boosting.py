import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"

        n = len(trainingData)
        p = [1.0/n for i in range(n)]
        for l in range(self.boosting_iterations):
            self.classifiers[l].train(trainingData, trainingLabels, p)
            guesses = self.classifiers[l].classify(trainingData)
            error = 0
            for j in range(n):
                if guesses[j] != trainingLabels[j]:
                    error += p[j]
            for j in range(n):
                if guesses[j] == trainingLabels[j]:
                    p[j] = p[j]*(error/(1-error))
            s = sum(p)
            p = [p[i]/s for i in range(len(p))]
            self.alphas[l] = np.log((1-error)/error)

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
        for l in range(self.boosting_iterations):
            temp = self.classifiers[l].classify(data)
            temp = [self.alphas[l]*temp[i] for i in range(len(temp))]
            guesses.append(temp)

        temp = zip(*guesses)
        final_guesses = [1 if sum(t) >= 0 else -1 for t in temp]

        return final_guesses
        # util.raiseNotDefined()