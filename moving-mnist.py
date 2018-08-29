from PIL import Image
import sys
import os
import math
import numpy as np
import random

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################
#label_dict = dict()
#lbl = None
# helper functions

def arr_from_img(im,shift=0):
    w,h=im.size
    arr=im.getdata()
    c = np.product(arr.size) / (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255. - shift

def get_picture_array(X, index, shift=0):
    #print("index:", index)
    
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
    if ch == 1:
        ret=ret.reshape(h,w)
    return ret

# loads mnist from web on demand
def load_dataset():
    #global label_dict, lbl
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    
    import gzip

    def load_mnist_images(filename):
        #global label_dict, lbl
        if not os.path.exists(filename):
            download(filename)
        # labels
        if not os.path.exists('train-labels-idx1.ubyte.gz'):
            download('train-labels-idx1.ubyte.gz')

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)

        # labels
        with open('train-labels.idx1-ubyte', 'rb') as f:
            lbl = np.fromfile(f, dtype=np.int8)
            #labels = np.frombuffer(f.read(), np.int32, offset=8)
        #labels = labels.reshape(-1, 32)
        lbl = list(lbl[8:])
        
        #lbl.sort()
        print("labels:", type(lbl), len(lbl))
        print("label[31225]:", lbl[31225])
        label_dict = { j : [i for i in range(len(lbl)) if lbl[i] == j] for j in range(0,10) }
        
        #print("len(label_dict)", len(label_dict))
        #print('label dict 5:', label_dict[5])
        return lbl, label_dict, data / np.float32(255)
    lbl, label_dict, dat = load_mnist_images('train-images-idx3-ubyte.gz')
    return lbl, label_dict, dat

# generates and returns video frames in uint8 array
def generate_moving_mnist(shape=(64,64), seq_len=30, seqs=10000, num_sz=28, nums_per_image=2,\
                          num1=0, num2=9):
    #global label_dict, lbl
    lbl, label_dict, mnist = load_dataset()
    width, height = shape
    lims = (x_lim, y_lim) = width-num_sz, height-num_sz
    dataset = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
    

    for seq_idx in xrange(seqs):
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
        speeds = np.random.randint(5, size=nums_per_image)+2
        veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]
        #print(" mnist.shape[0]:",  mnist.shape[0])
        #mnist_images = [Image.fromarray(get_picture_array(mnist,r,shift=0)).resize((num_sz,num_sz), Image.ANTIALIAS) \
        #       for r in np.random.randint(0, mnist.shape[0], nums_per_image)]

        indexes = [random.choice(label_dict[num1]), random.choice(label_dict[num2])]
        print("Randomly selected indexes:", indexes)
        print("Got index {} and {} for {} and {} respectively.".format(
            indexes[0], indexes[1], num1, num2) )
        i_one, i_two = indexes[0], indexes[1]
        print("i_one:", i_one, "i_two:", i_two)
        #print("part of lbls", lbl[15000:15100])
        print("len lbl:", len(lbl))
        print("lbl[ i_one ]:", lbl[ i_one ])
        print("lbl[ i_two ]:", lbl[ i_two ])
        print("The correspondnig label for these indexes are {} and {}".format(
            lbl[ i_one ], lbl[ i_two ] ) )
        print("\n label for 24097 again:", lbl[24097])

        mnist_images = [Image.fromarray(get_picture_array(mnist,r,shift=0)).resize((num_sz,num_sz), Image.ANTIALIAS) \
               for r in indexes ]
        print("length mnist_images: ", len(mnist_images), mnist_images)
        #exit(1)

        positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in xrange(nums_per_image)]
        for frame_idx in xrange(seq_len):
            canvases = [Image.new('L', (width,height)) for _ in xrange(nums_per_image)]
            canvas = np.zeros((1,width,height), dtype=np.float32)
            for i,canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                canvas += arr_from_img(canv, shift=0)
            # update positions based on velocity
            next_pos = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j]+2:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
            positions = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            # copy additive canvas to data array
            dataset[seq_idx*seq_len+frame_idx] = (canvas * 255).astype(np.uint8).clip(0,255)
    return dataset

def main(dest, filetype='npz', frame_size=64, seq_len=30, seqs=100, num_sz=28, nums_per_image=2,
         num_1=0, num_2=9):
    
    dat = generate_moving_mnist(shape=(frame_size,frame_size), seq_len=seq_len, seqs=seqs, \
                                num_sz=num_sz, nums_per_image=nums_per_image, \
                                num1=num_1, num2=num_2)
    n = seqs * seq_len
    if filetype == 'hdf5':
        import h5py
        from fuel.datasets.hdf5 import H5PYDataset
        def save_hd5py(dataset, destfile, indices_dict):
            f = h5py.File(destfile, mode='w')
            images = f.create_dataset('images', dataset.shape, dtype='uint8')
            images[...] = dataset
            split_dict = dict((k, {'images':v}) for k,v in indices_dict.iteritems())
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()
        indices_dict = {'train': (0, n*9/10), 'test': (n*9/10, n)}
        save_hd5py(dat, dest, indices_dict)
    elif filetype == 'npz':
        np.savez(dest, dat)
    elif filetype == 'jpg':
        for i in xrange(dat.shape[0]):
            Image.fromarray(get_picture_array(dat, i, shift=0)).save(os.path.join(dest, '{}.jpg'.format(i)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dest', type=str, dest='dest')
    parser.add_argument('--filetype', type=str, dest='filetype')
    parser.add_argument('--frame_size', type=int, dest='frame_size')
    parser.add_argument('--seq_len', type=int, dest='seq_len') # length of each sequence
    parser.add_argument('--seqs', type=int, dest='seqs') # number of sequences to generate
    parser.add_argument('--num_sz', type=int, dest='num_sz') # size of mnist digit within frame
    parser.add_argument('--nums_per_image', type=int, dest='nums_per_image') # number of digits in each frame
    parser.add_argument('--num1', type=int, dest='num_1') # first number
    parser.add_argument('--num2', type=int, dest='num_2') # second number
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
