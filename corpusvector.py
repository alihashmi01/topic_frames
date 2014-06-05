#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################################
# This code is available under the MIT License.
# Modified using 'vocabulary' code from (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# Ali H, ML
###################################################################################


import grammarmodule
import nltk
import re



def load_file(filename):
    corpus = []
    f = open(filename, 'r')
    g = grammarmodule.GrammarModule()
    for line in f:
        str = re.sub('[^A-Za-z0-9 \'\,\.\-]+', '', line)
        str = str.decode("utf8")
        doc = re.findall(r'\w+(?:\'\w+)?',g.get_str_entities(str))
        #doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus

recover_list = {"wa":"was", "ha":"has"}
wl = nltk.WordNetLemmatizer()

def lemmatize(w0):
    w = wl.lemmatize(w0.lower())
    #if w=='de': print w0, w
    if w in recover_list: return recover_list[w]
    return w

class CorpusVectors:
    def __init__(self):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = []      # id to document frequency

    def term_to_id(self, term0):
        term = lemmatize(term0)
        if not re.match(r'[a-z]+$', term): return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
            self.docfreq.append(0)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):
        #print ' '.join(doc)
        list = []
        words = dict()
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                if not words.has_key(id):
                    words[id] = 1
                    self.docfreq[id] += 1
        if "close" in dir(doc): doc.close()
        return list

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

