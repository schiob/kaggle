# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    # create the training and test sets
    dataset = genfromtxt('data/train.csv', delimiter=',', dtype='f8', skip_header=1)
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt('data/test.csv', delimiter=',', dtype='f8', skip_header=1)
    
    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
    
    savetxt('data/submission.cvs', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='MoleculeId,PredictedProbability', comments='')

if __name__ == '__main__':
    main()
