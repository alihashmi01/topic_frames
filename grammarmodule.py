#!/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
# This code is available under the MIT License.
#Grammar from this paper http://lexitron.nectec.or.th/public/COLING-2010_Beijing_China/PAPERS/pdf/PAPERS065.pdf
#intuition:  noun-phrases keywords as a rich set of topics 
#Ali H, ML
################################################################################

import couchdb
import nltk
from nltk import Text
from nltk import TextCollection
import math
import re
import time
import string
import sys
import datetime
from datetime import datetime
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import timeit
import gensim
from gensim import corpora, models, similarities
import logging, gensim, bz2

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl



class GrammarModule:

    def __init__(self):
        self.tokenizer = nltk.WordPunctTokenizer()#nltk.RegexpTokenizer("[\w]", flags=re.UNICODE)
        self.stopwords = self.getStopWordList('stop-words-english4.txt')
        #https://gist.github.com/alexbowe/
        self.sentence_re = r'''(?x)
              ([A-Z])(\.[A-Z])+\.?
            | \w+(-\w+)*
            | \$?\d+(\.\d+)?%?
            | \.\.\.
            | [][.,;"'?():-_`]
        '''
        # Grammar from this paper http://lexitron.nectec.or.th/public/COLING-2010_Beijing_China/PAPERS/pdf/PAPERS065.pdf
        self.grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  #

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        self.toks = ""
        self.postoks = ""
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.tree = ""
 
    def leaves(self,tree):
        #NP leaf node of tree
        for childtree in tree.subtrees(filter = lambda t: t.node=='NP'):
            yield childtree.leaves()

    def lemmatize_word(self,word,lemma, stemm, lowercase):
        # this results in unsusual words, thistl instead of thistle
        if lemma == 1:
            word = self.lemmatizer.lemmatize(word)
        if stemm == 1:
            word = self.stemmer.stem_word(word)
        if lowercase == 1:
            word = word.lower()
        return word

    def filterStopWords(self,word):
        blnIncludeWord = bool( len(word) > 1
            and word.lower().decode('utf-8') not in self.stopwords)
        return blnIncludeWord

    def get_terms(self):
        for leaf in self.leaves(self.tree):
            term = [ self.lemmatize_word(w,0,0,0) for w,t in leaf if self.filterStopWords(w)]
            yield term

    #this will generate a noune phrase term string
    def get_str_entities(self,txtPassString):
        self.toks = nltk.regexp_tokenize(txtPassString, self.sentence_re)
        self.postoks = nltk.tag.pos_tag(self.toks)
        self.tree = self.chunker.parse(self.postoks)
        strTermsString = ""
        arrayterms = self.get_terms()
        for singlearray in arrayterms:
            for oneterm in xrange(len(singlearray)):
                strTermsString = strTermsString + " " + re.sub('cq$', '', re.sub('[,.?!\t\n\_\%\$ ]+', '', singlearray[oneterm]))
        #print strTermsString
        return strTermsString

    #to track the history of the run
    def getTimeStamp(self, inputstartend):
        FORMAT = '%Y-%m-%d %H:%M:%S'
        data = "\n" + inputstartend + ' ' + datetime.now().strftime(FORMAT)
        with open("trackhistoryfile.txt", "a") as trackfile:
            trackfile.write(data)

    def getStopWordList(self,stopWordListFileName):
        #read the stopwords file and build a list
        stopWords = []
        stopWords.append('Boston.com')
        stopWords.append('Your Town')
        stopWords.append('Boston Globe')
        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word.decode('utf-8'))
            line = fp.readline()
        fp.close()
        if stopWords:
            stopWords.sort()
            last = stopWords[-1]
            for i in range(len(stopWords)-2, -1, -1):
                if last == stopWords[i]:
                    del stopWords[i]
                else:
                    last = stopWords[i]
        return stopWords

    def getStopWords(self):
        return self.stopwords


