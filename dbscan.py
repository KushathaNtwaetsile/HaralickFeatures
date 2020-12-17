from som_class import *
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
import numpy as np
import mahotas as mh
import pickle
import random
import hdbscan
from skimage import io
import os
import glob
import time

#get a list of all the mosaic names
mosaics = glob.glob('/beegfs/general/lofar/mosaics/*mosaic.fits')
names = [m.split('/')[-1].split('.')[0] for m in mosaics]

#catalogue file
catalogue = '/data/astroml/kushkelly/Lofar/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits'

#read catalogue
catdat= fits.getdata(catalogue, header=True, ignore_missing_end=True)[0]

#get indexes for sorted flux, in descending order
sorted_by_flux = np.argsort(catdat.Total_flux)[::-1]

#we are going to look at the top_n brightest sources
top_n = 10000
sample = sorted_by_flux[:top_n]


#get the RA, DEC, FLUX, MOSAICID and a catalogue number for sample
RA =catdat.RA[sample]
DEC =catdat.DEC[sample]
FLUX = catdat.Total_flux[sample]
MOS = catdat.Mosaic_ID[sample]
NUM = range(len(RA)) #catdat.Isl_id[sample]

#size of cutout in pixels
size = 64

#empty list to hold feature vectors
vectors = []

#keep a list of the cutouts so we can make montages later
ims = []

#make a dictionary to store the big images once we open them to save time on i/o
done = {}

#loop over sample
for i in range(len(RA)):
 #so we know where we are in the loop
    print(i)

    #check to see if we have already made the cutout - if we havent then extract it
    if not os.path.exists('cuts/%08d.npy'%NUM[i]):
        image = '/beegfs/general/lofar/mosaics/%s-mosaic.fits'%MOS[i]
        w = WCS(image)
        if not MOS[i] in done.keys():
            f = fits.open(image)
            maindata = np.squeeze(f[0].data)
            done[MOS[i]] = maindata
        else:
            maindata = done[MOS[i]]

        #cutout from main image
        position = SkyCoord(RA[i]*u.deg, DEC[i]*u.deg)
        cutout = Cutout2D(maindata, position, size, wcs=w, mode='trim')
        ((ymin,ymax),(xmin,xmax)) = cutout.bbox_original
        data = maindata[ymin:ymax,xmin:xmax]

        #normalize
        data = data-np.min(data)
        data = data/np.max(data)
        data = data*256
        #save as a npy file in ./cuts/
        np.save('cuts/%08d.npy'%NUM[i],data)
    else:
 #otherwise just read from ./cuts/
        data = np.load('cuts/%08d.npy'%NUM[i])

    #append cutout to list
    ims.append(data)

    #calculate haralick features
    haralick_features = mh.features.haralick(data.astype(np.uint8))
    haralick = np.mean(haralick_features, axis=0)

    #append this to list of feature vectors
    vectors.append(haralick)

#eps = 0.3
#set up the HDBSCAN clusterer - see params for more details, but this seems to do okay
clusterer = hdbscan.HDBSCAN(min_cluster_size=40,min_samples=1,algorithm='best',cluster_selection_method='eom', metric='euclidean')

clusterer.fit(vectors)

#clusters = clusterer.single_linkage_tree_.get_clusters(cut_distance=0.8,min_cluster_size=40)

#run the model using the list of vectors
#clusterer.fit(vectors)

#get a list of unique labels and the number of counts of each label
#label = -1 is 'noise' - i.e. couldn't be fit to a cluster
ulab,ucount= np.unique(clusterer.labels_,return_counts=True)

#now loop over the unique labels
for lab in ulab:
    #identify all the indexes in the vector list that correspond to given label
    idxs = clusterer.labels_==lab
    ids = np.array(range(len(ims)))[idxs]

    #each vector also has a probability of belonging to the cluster - get this information
    probs = clusterer.probabilities_[idxs]

    #sort the indexes for that label by probability - top first
    ids_prob_sort = np.argsort(probs)[::-1]

    #make the montages - limit to 16x16 grid
    n=0
    for i in ids_prob_sort:

        #get the correct image from the list of cutouts
        j = ids[i]
        im = ims[j]
        #minmax normalize
        im-=np.min(im)
        im/=np.max(im)
        #create temporary output image
        outim = 'image%03d_%04d_%03d.png'%(lab,1000-1000*clusterer.probabilities_[i],n)
        io.imsave(outim,im)

        #test to label with probability - ignore
        #os.system("""/usr/bin/convert %s -fill white -pointsize 12  -draw "text 16,16 '%04d'" _%s"""%(outim,1000*clusterer.probabilities_[i],outim))
        n+=1
	  #break if we hit 16x16 examples
        if n>=(16*16):
            break

    #now montage the images for a given label
    os.system('/usr/bin/montage image%03d*.png -tile 16x16 -geometry 64x64+1+1 label%03d.png'%(lab,lab))
    #remove individual images
    os.system('rm image%03d_*.png'%lab)
    os.system('rm _image%03d_*.png'%lab)

#print out the labels and their counts
for l,c in zip(ulab,ucount):
    print(l, c)


