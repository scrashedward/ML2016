#! /usr/bin/env python

from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pandas as pd
from stemming.porter2 import stem
from nltk.corpus import stopwords


stopword = stopwords.words('english')

stopword.append(['i', 'how', 'what'])

chars_to_remove = ['.', '!', '?', ',', '[', ']', '(', ')', '/', '%', '#', '@', '=', '+', '&', '*', '\'', '"', '\n']


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--data", dest = "data", help="input data directory")

op.add_option("--output", dest = "output", help = "output file name")

op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(opts.data)

print(opts.output)

print(opts.n_components)

print(opts.minibatch)

print(opts.use_idf)

print(opts.use_hashing)

print(opts.verbose)

f = open(opts.data + "title_StackOverflow.txt", 'r')

dataset = f.readlines()

dataset = [" ".join([word.lower().translate(None, ''.join(chars_to_remove)) for word in sentence.split(" ")]) for sentence in dataset]

dataset = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in dataset]

dataset = [" ".join([word for word in sentence.split(" ") if unicode(word, 'utf-8') not in stopword]) for sentence in dataset]

t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stopwords.words('english'),
                                 use_idf=opts.use_idf)

X = vectorizer.fit_transform(dataset)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=50, init='k-means++', n_init=100,
                         init_size=1000, batch_size=500, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print(type(km.labels_))
print(len(km.labels_))


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(20):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

df = pd.read_csv(opts.data + 'check_index.csv', usecols = [1,2],skiprows = 0)

data = df.as_matrix()

result = km.labels_[data[:,0]] == km.labels_[data[:, 1]]

print(type(result))
print(result.shape)

f = open(opts.output, 'w+')

f.write('ID,Ans\n')

for i in range(result.shape[0]):
	if result[i]:
		f.write(str(i) + ',1\n')
	else:
		f.write(str(i) + ',0\n')
