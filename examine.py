import neural
import genetics
import formula
import pickle
import os
import simpletest

nets = []

for fname in sorted(os.listdir()):
    if fname.startswith('saved_'):
        with open(fname, 'rb') as f:
            nets = [pickle.load(f)] + nets
