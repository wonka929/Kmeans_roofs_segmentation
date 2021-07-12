#!/usr/bin/python

from skimage import io
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import os
import pathlib
import sys, getopt


def main(argv):
    inputfile = ''
    outputfile = ''
    K = 0
    try:
        opts, args = getopt.getopt(argv,"hi:o:K:",["ifile=","ofile=","clusters="])
    except getopt.GetoptError as e:
        print(e)
        print('kmean_segmentation.py -i <inputfile> -K <numero_cluster> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('kmean_segmentation.py -i <inputfile> -K <numero_cluster> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-K", "--clusters"):
            K = arg
    print('Input file is "', inputfile)
    print('Output file is "', outputfile)
    print('Numero di cluster is ', K)
   
    if inputfile.endswith('.jpg'):
        img = io.imread(inputfile)
        img_r = (img / 255.0).reshape(-1,3)
        k_colors = KMeans(n_clusters=int(K), init = 'k-means++').fit(img_r) 
        y_pred=k_colors.predict(img_r)
        newimg=k_colors.cluster_centers_[k_colors.labels_]
        newimg=np.reshape(newimg, (img.shape))
        io.imsave(outputfile + '_' + str(K) + '.jpg', newimg)
   

if __name__ == "__main__":
    main(sys.argv[1:])
