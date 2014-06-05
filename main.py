#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################################
# This is a wrapper function for generating lda-reduced grammar-based topic set
# This code is available under the MIT License.
# Ali H, ML
###################################################################################

import lda

def main():
    #def generateTopics(filename, topics,iterations):
    # alpha = distributions for (per document) topic & beta = (per topic) word distributions
    # smaller alpha = more sparse the distribution
    # intuition: high alpha-value =  documents more similar in terms of what topics the docs contain
    # intuition: high beta-value  = topics more similar in terms of what words they contain
    # The # of iterations increases quality of the topic model
    # see example here, http://www.mblondel.org/journal/2010/08/21/latent-dirichlet-allocation-in-python/
    # we can perhaps provide an interface for these parameters later
    # def generateTopics(filename, topics,iterations, alpha_param, beta_param):
    lda.generateTopicsParams('BalochHalCorpus.txt',6,25, None, None)
    
    
if __name__ == "__main__":
    main()