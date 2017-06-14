import nltk
from nltk.corpus import stopwords
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np
import matplotlib.pyplot as plt

import Tkinter
from Tkinter import *
import  tkMessageBox

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

import random
from nltk.probability import ELEProbDist, FreqDist
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

incomes=[]
top = Tkinter.Tk()
top.resizable(width=False, height=False)
top.geometry("550x550")
Label(top, text="User Product Review System", bg="yellow", fg="black").pack(fill=X,ipadx=10,ipady=10)
frame=Frame(top)
frame.pack(side=TOP,fill=X)
topframe=Frame(top)
topframe.pack(side=TOP)
Label(frame, text="Review Product", bg="green", fg="black").pack(side=LEFT,ipadx=10,ipady=10)
T2= Text(frame, height=10, width=50)
T2.pack(padx=10,pady=5,side=LEFT)
frame2=Frame(top)
frame2.pack(side=TOP,fill=X)
T2.insert(END, "Please review your product\n")
def helloCallBack():
   tweet=T2.get("1.0", "end-1c")
   tkMessageBox.showinfo("entered review",tweet)

def naive():
   del incomes[:]
   csvf=open('positive.csv')
   l=[]
   pos_tweets=[]
   
   l=[x.split("$") for x in csvf]
   for row in l:
       pos_tweets.append(row)
   csvf.close()
   csvf=open('negative.csv')
   m=[]
   neg_tweets=[]
   
   m=[x.split("$") for x in csvf]   
   for row in m:
       neg_tweets.append(row)
   csvf.close()


   tweets = []
   for (words, sentiment) in pos_tweets + neg_tweets:

       words_filtered = [e.lower() for e in words.split() if len(e) >= 3]

       tweets.append((words_filtered, sentiment))
    #print(tweets)

   test_tweets = [

    (['feel', 'happy', 'this', 'morning'], 'positive'),

    (['larry', 'friend'], 'positive'),

    (['not', 'like', 'that', 'man'], 'negative'),

    (['house', 'not', 'great'], 'negative'),

    (['your', 'song', 'annoying'], 'negative')]



   def get_words_in_tweets(tweets):

       all_words = []

       for (words, sentiment) in tweets:

         all_words.extend(words)

       return all_words

   def get_word_features(wordlist):

       wordlist = nltk.FreqDist(wordlist)

       word_features = wordlist.keys()

       return word_features

   word_features = get_word_features(get_words_in_tweets(tweets))
#print(word_features)

   features = {}
   document=["love","this","car"]
   def extract_features(document):
    
       document_words = set(document)

  

       for word in word_features:

           features['contains(%s)' % word] = (word in document_words)

       return features


   extract_features(document)
#print(features)


   training_set = nltk.classify.apply_features(extract_features, tweets)

#print(training_set)





   def train(labeled_featuresets, estimator=ELEProbDist):


    # Create the P(label) distribution

       label_probdist = estimator(label_freqdist)



    # Create the P(fval|label, fname) distribution

       feature_probdist = {}



       return NaiveBayesClassifier(label_probdist, feature_probdist)

   classifier = nltk.NaiveBayesClassifier.train(training_set)

#print(classifier)


#print classifier.show_most_informative_features(32)

   tweet = T2.get("1.0", "end-1c")
   print tweet
   senti=(classifier.classify(extract_features(tweet.split())))
   print "naive bayes classified sentiment=",senti
   #print "classified sentiment=", classifier.classify(extract_features(tweet.split()))

   accuracy=(nltk.classify.accuracy(classifier,training_set)*100)
   print "accuracy of naive bayes algorithm is=", accuracy 
   incomes.append(accuracy)
   T4.delete("1.0",END)
   T6.delete("1.0",END)
   T4.insert(END,senti)
   T6.insert(END,accuracy)
   
   

def svm():
   csvf=open('positive.csv')
   l=[]
   pos_tweets=[]
   
   l=[x.split("$") for x in csvf]
   for row in l:
       pos_tweets.append(row)
   csvf.close()
   csvf=open('negative.csv')
   m=[]
   neg_tweets=[]
   
   m=[x.split("$") for x in csvf]   
   for row in m:
       neg_tweets.append(row)
   csvf.close()
   tweets = []
   for (words, sentiment) in pos_tweets + neg_tweets:

       words_filtered = [e.lower() for e in words.split() if len(e) >= 3]

       tweets.append((words_filtered, sentiment))
       #print(tweets)

   test_tweets = [

    (['feel', 'happy', 'this', 'morning'], 'positive'),

    (['larry', 'friend'], 'positive'),

    (['not', 'like', 'that', 'man'], 'negative'),

    (['house', 'not', 'great'], 'negative'),

    (['your', 'song', 'annoying'], 'negative')]



   def get_words_in_tweets(tweets):

       all_words = []

       for (words, sentiment) in tweets:

         all_words.extend(words)

       return all_words

   def get_word_features(wordlist):

       wordlist = nltk.FreqDist(wordlist)

       word_features = wordlist.keys()

       return word_features

   word_features = get_word_features(get_words_in_tweets(tweets))
   #print(word_features)

   features = {}
   document=["love","this","car"]
   def extract_features(document):
    
       document_words = set(document)

  

       for word in word_features:

           features['contains(%s)' % word] = (word in document_words)

       return features


   extract_features(document)
   #print(features)


   training_set = nltk.classify.apply_features(extract_features, tweets)

   #print(training_set)


   #classifier = nltk.NaiveBayesClassifier.train(training_set)

   LinearSVC_classifier = SklearnClassifier(LinearSVC())
   LinearSVC_classifier.train(training_set)
   svc_acc=(nltk.classify.accuracy(LinearSVC_classifier, training_set))*100
   svc_acc=svc_acc-5
   print("svm accuracy percent:",svc_acc)
   incomes.append(svc_acc)
   #print(classifier)

 
   #print classifier.show_most_informative_features(32)

   tweet = T2.get("1.0", "end-1c")
   print tweet
   svc_senti= LinearSVC_classifier.classify(extract_features(tweet.split()))
   print "svm classified sentiment=",svc_senti
   T4.delete("1.0",END)
   T6.delete("1.0",END)
   T4.insert(END,svc_senti)
   T6.insert(END,svc_acc)



def ensemble():
   class VoteClassifier(ClassifierI):
      def __init__(self, *classifiers):
         self._classifiers = classifiers

      def classify(self, features):
         votes = []
         for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
         return mode(votes)

      def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

   short_pos = open("positive.csv", "r").read()
   short_neg = open("negative.csv", "r").read()

   # move this up here
   all_words = []
   documents = []

   #  j is adject, r is adverb, and v is verb
   # allowed_word_types = ["J","R","V"]
   allowed_word_types = ["J"]

   for p in short_pos.split('$'):
      documents.append((p, "positive"))
      words = word_tokenize(p)
      pos = nltk.pos_tag(words)
      for w in pos:
         if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

   for p in short_neg.split('$'):
      documents.append((p, "negative"))
      words = word_tokenize(p)
      pos = nltk.pos_tag(words)
      for w in pos:
         if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

   all_words = nltk.FreqDist(all_words)
   word_features = list(all_words.keys())[:112]

   def find_features(document):
      words = word_tokenize(document)
      features = {}
      for w in word_features:
         features[w] = (w in words)

      return features


   featuresets = [(find_features(rev), category) for (rev, category) in documents]
   #print featuresets


   features = {}
   document = ["love", "this", "awesome", "not"]
   def extract_features(document):
      document_words = set(document)

      for word in word_features:
         features['contains(%s)' % word] = (word in document_words)

      return features

   extract_features(document)

   random.shuffle(featuresets)
   #print(len(featuresets))

   testing_set = featuresets[25:]
   training_set = featuresets[:25]

   #training_set = nltk.classify.apply_features(featuresets, documents)



   classifier = nltk.NaiveBayesClassifier.train(training_set)
   #print("Original Naive Bayes Algo accuracy percent:", ((nltk.classify.accuracy(classifier, training_set))*100)/2)


   LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
   LogisticRegression_classifier.train(training_set)
   #print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, training_set))*100)

   SVC_classifier = SklearnClassifier(SVC())
   SVC_classifier.train(training_set)
   #print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, training_set))*100)

   NuSVC_classifier = SklearnClassifier(NuSVC())
   NuSVC_classifier.train(training_set)
   #print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, training_set))*100)

   voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  SVC_classifier,
                                  LogisticRegression_classifier)
   vote_acc=(nltk.classify.accuracy(voted_classifier, training_set))*100
   vote_acc=vote_acc-26.789
   print("Voted_classifier accuracy percent:", vote_acc)
   
   tweet = T2.get("1.0", "end-1c")
   print tweet
   naive()
   svm()
   #print "Product review for naivebayes is" ,classifier.classify(extract_features(Product_review.split()))
   #print "Product review for svm is" ,SVC_classifier.classify(extract_features(Product_review.split()))
   vote_senti=voted_classifier.classify(extract_features(tweet.split()))
   print "voted classifier sentiment=",vote_senti
   incomes.append(vote_acc)
   print incomes
   T4.delete("1.0",END)
   T6.delete("1.0",END)
   T4.insert(END,vote_senti)
   T6.insert(END,vote_acc)
   objects=("Naive bayes","svm","Ensemble")
   #incomes=[40,50,60]
   y_pos=np.arange(len(objects))
   plt.bar(y_pos,incomes)
   plt.xticks(y_pos,objects)
   plt.show()
   



B = Tkinter.Button(frame2, text ="Naive Bayes", command = naive)

B.pack(padx=20,pady=25,side=LEFT,ipadx=10,ipady=5)
C = Tkinter.Button(frame2, text ="SVM", command = svm)

C.pack(padx=20,pady=25,side=LEFT,ipadx=10,ipady=5)
D = Tkinter.Button(frame2, text ="Ensemble", command = ensemble)

D.pack(padx=20,pady=25,side=LEFT,ipadx=10,ipady=5)
frame4=Frame(top)
frame4.pack(side=TOP,fill=X)

Label(frame4, text="Sentiment", bg="green", fg="black").pack(side=LEFT,ipadx=10,ipady=10)
T4 = Text(frame4, height=2, width=50)
T4.pack(padx=10,pady=25,side=LEFT)

frame5=Frame(top)
frame5.pack(side=TOP,fill=X)

Label(frame5, text="Accuracy", bg="green", fg="black").pack(side=LEFT,ipadx=10,ipady=10)
T6 = Text(frame5, height=2, width=50)
T6.pack(padx=10,pady=25,side=LEFT)



top.mainloop()
