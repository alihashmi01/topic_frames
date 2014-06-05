#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################################
# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# Modified code from (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# Ali H, ML
###################################################################################

import datetime
import corpusvector
import re
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import numpy
import time

class LatentDrichAllocation:
    def __init__(self, topics, alpha, beta, docs, vocabulary_size, smart_initialize=True):
        self.topics = topics
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.vocabulary_size = vocabulary_size
        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), topics)) + alpha     # word count of each document and topic
        self.n_z_t = numpy.zeros((topics, vocabulary_size)) + beta # word count of each topic and vocabulary
        self.n_z = numpy.zeros(topics) + vocabulary_size * beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smart_initialize:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, topics)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        topicsalpha = self.topics * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + topicsalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def lda_generate_learn(lda, iteration, voca):
    # Lower perplexity is better.
    # Perplexity is deﬁned as the geometric mean of the inverse marginal probability of each word in the held-out set of document
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.topics, dtype=int)
    wordcount = [dict() for k in xrange(lda.topics)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    timestring = time.strftime("%Y%m%d")
    fj = open('topicsfile_'+timestring+'_'+str(lda.topics) + '.JSON','w')
    fj.write("{\"topics\": [")
    for k in xrange(lda.topics):
        #print "\n-- topic: %d (%d words)" % (k, zcount[k])
        fj.write("{ \"topic\": \"%s\",\"keywords\":[" % k)
        keywords_str=""
        numc = 0
        for w in numpy.argsort(-phi[k])[:40]:
            #print "%s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0))
            #print "%s" % voca[w]
            keywords_str = keywords_str + ' ' +voca[w]
            numc = numc + 1
            #print "k=%s w=%d  numc=%d  \n" % (k,w,numc)
            if numc == 40 :
                fj.write("{\"keyword\": \"%s\",\"frequency\": \"%d\"}" % (voca[w],wordcount[k].get(w,0)))
            else:
                fj.write("{\"keyword\": \"%s\",\"frequency\": \"%d\"}," % (voca[w],wordcount[k].get(w,0)))
        fj.write("]")
        if k == lda.topics - 1:
            fj.write("}")
        else:
            fj.write("},")
    fj.write("]}")
    fj.close()

def generateTopics(filename, topics,iterations):
    #time stamping for reporting
    start_time = time.time()
    start_stamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    
    corpus=corpusvector.load_file(filename)
    vocab = corpusvector.CorpusVectors()
    docs = [vocab.doc_to_ids(doc) for doc in corpus]
    # alpha = distributions for (per document) topic & beta = (per topic) word distributions
    # smaller alpha = more sparse the distribution
    # intuition: high alpha-value =  documents more similar in terms of what topics the docs contain
    # intuition: high beta-value  = topics more similar in terms of what words they contain
    # The # of iterations increases quality of the topic model
    # see example here, http://www.mblondel.org/journal/2010/08/21/latent-dirichlet-allocation-in-python/
    # we can perhaps provide an interface for these parameters later
    alpha_param = 0.5 #default
    beta_param = 0.5 #default
    smart_init 
    lda_object= LatentDrichAllocation(topics, alpha_param, beta_param, docs, vocab.size(),True)
    x = vocab.size()
    print "vocab size = % d" % x
    lda_generate_learn(lda_object, iterations, vocab)
    
    #time stamping for reporting
    end_time = time.time()
    end_stamp = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    print start_stamp
    print end_stamp
    print "corpus=%d, words=%d, topics=%d iterations=%d" % (len(corpus), len(vocab.vocas), topics, iterations)
    print "elapsed time in seconds=%s " % str(end_time - start_time)


def generateTopicsParams(filename, topics,iterations, alpha_param, beta_param):
    if alpha_param == None:
        alpha_param=0.5
    if beta_param == None:
        beta_param=0.5
        
    #time stamping for reporting
    start_time = time.time()
    start_stamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    
    corpus=corpusvector.load_file(filename)
    vocab = corpusvector.CorpusVectors()
    docs = [vocab.doc_to_ids(doc) for doc in corpus]
    # alpha = distributions for (per document) topic & beta = (per topic) word distributions
    # smaller alpha = more sparse the distribution
    # intuition: high alpha-value =  documents more similar in terms of what topics the docs contain
    # intuition: high beta-value  = topics more similar in terms of what words they contain
    # The # of iterations increases quality of the topic model
    # see example here, http://www.mblondel.org/journal/2010/08/21/latent-dirichlet-allocation-in-python/
    # we can perhaps provide an interface for these parameters later
    lda_object= LatentDrichAllocation(topics, alpha_param, beta_param, docs, vocab.size(),True)
    x = vocab.size()
    print "vocab size = % d" % x
    lda_generate_learn(lda_object, iterations, vocab)
    
    #time stamping for reporting
    end_time = time.time()
    end_stamp = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    print start_stamp
    print end_stamp
    print "corpus=%d, words=%d, topics=%d iterations=%d alpha=%f beta=%f" % (len(corpus), len(vocab.vocas), topics, iterations, alpha_param, beta_param)
    print "elapsed time in seconds=%s " % str(end_time - start_time)
