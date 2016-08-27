

import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk
import warnings
warnings.filterwarnings("ignore")

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def remove_stop(stopwords,train):
    temp_lis = []
    for tweet in train:
        l = set(tweet)
        temp_lis.append(filter(lambda x: x not in stopwords,l))
    return temp_lis

def atleast_one(train):
    temp_dict = {}
       
    for tweet in train:
        for word in tweet:
           if word in temp_dict:
                temp_dict[word] += 1
           else:
                temp_dict[word] = 1        
    
    return temp_dict

def build_feature(pos,neg,min_pos,min_neg):
    temp_list = []   
    for key in pos.keys():
        ctr_pos = pos[key]
        if key in neg:
            ctr_neg = neg[key]
            if ctr_pos >= 2*ctr_neg and (ctr_pos>=min_pos or ctr_neg>=min_neg):
                temp_list.append(key)
        else:
            if ctr_pos >= min_pos:
                temp_list.append(key)       

    for key in neg.keys():
        ctr_neg = neg[key]
        if key in pos:
            ctr_pos = pos[key]
            if ctr_neg >= 2*ctr_pos and (ctr_pos>=min_pos or ctr_neg>=min_neg):
                temp_list.append(key)
        else:
            if ctr_neg >= min_neg:
                temp_list.append(key)

    
    return temp_list    

def build_vector(data,feature):
    
    temp_list = []
    length = len(feature)
    
    for tweet in data:
        temp_tweet = []
        temp_dict = dict(zip(feature,[0]*length))
        
        for word in tweet:
            if word in temp_dict:    
                temp_dict[word] = 1
         
        for key in temp_dict.keys():
            temp_tweet.append(temp_dict[key])
    
        temp_list.append(temp_tweet)

    return temp_list        


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list has the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    
    
    min_pos = 0.01 * len(train_pos)
    min_neg = 0.01 * len(train_neg)
    
    train_pos = remove_stop(stopwords,train_pos)
    train_neg = remove_stop(stopwords,train_neg)
    
    pos_dict = atleast_one(train_pos)
    neg_dict = atleast_one(train_neg)
    
    feature_words = build_feature(pos_dict,neg_dict,min_pos,min_neg)
    
   

    train_pos_vec = build_vector(train_pos,feature_words)
    train_neg_vec = build_vector(train_neg,feature_words)
    test_pos_vec = build_vector(test_pos,feature_words)
    test_neg_vec = build_vector(test_neg,feature_words)
    
   
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
   
    
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []
    
    for i in range(len(train_pos)):
        lab = 'TRAIN_POS_'+str(i)
        labeled_train_pos.append(LabeledSentence(words = train_pos[i],tags=[lab]))
    for i in range(len(train_neg)):
        lab = 'TRAIN_NEG_'+str(i)
        labeled_train_neg.append(LabeledSentence(words = train_neg[i],tags=[lab]))
    for i in range(len(test_pos)):
        lab = 'TEST_POS_'+str(i)
        labeled_test_pos.append(LabeledSentence(words = test_pos[i],tags=[lab]))
    for i in range(len(test_neg)):
        lab = 'TEST_NEG_'+str(i)
        labeled_test_neg.append(LabeledSentence(words = test_neg[i],tags=[lab]))
    
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
   
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

  
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    for i in range(len(train_pos)):
        lab = 'TRAIN_POS_'+str(i)
        train_pos_vec.append(model.docvecs[lab])

    for i in range(len(train_neg)):
        lab = 'TRAIN_NEG_'+str(i)
        train_neg_vec.append(model.docvecs[lab])

    for i in range(len(test_pos)):
        lab = 'TEST_POS_'+str(i)
        test_pos_vec.append(model.docvecs[lab])

    for i in range(len(test_neg)):
        lab = 'TEST_NEG_'+str(i)
        test_neg_vec.append(model.docvecs[lab])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X,Y)
   
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    
   
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X,Y)
   
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
   
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for tweet in test_pos_vec:
        if model.predict(tweet) == ['pos']:
            tp = tp + 1
        else:
            fn = fn + 1

    for tweet in test_neg_vec:
        if model.predict(tweet) == ['neg']:
            tn = tn + 1
        else:
            fp = fp + 1

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
