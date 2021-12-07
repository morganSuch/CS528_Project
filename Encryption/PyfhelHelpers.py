from PIL import Image
import numpy as np
import pickle, os, time


def ImgEncrypt(pyfhel, plainimg):
    counter = 0
    cipherimg = np.asarray(plainimg)
    shape = cipherimg.shape
    cipherimg = cipherimg.flatten().tolist()
    for pix in cipherimg:
        counter += 1
        #start = time.time()
        #pyfhel.encryptFrac(pix)
        #end = time.time()
        #print('pixel: ',counter, f'\ntime elapsed {end-start}\n')
    #cipherimg = [#pyfhel.encryptFrac(pix) for pix in cipherimg]
    print('Pixels in image: ', counter)
    return np.asarray(cipherimg).reshape(shape)


def saveEncryptedImg(cipherimg, filename):

    fname = "encrypted-images/train/NORMAL/" + filename
    fstream = open(fname, "wb")
    fstream.write(cipherimg)
    fstream.close()
    print('completed encryption: ', filename)
    b = os.path.getsize(fname)
    print('size of encrypted file: ',b)
    
    #pickle.dump(cipherimg, fstream)
    #for pix in cipherimg:
    #    dump = pickle.dumps(pix)
    #    fstream.write(dump)
    #cipherimg = [pyfhel.encodeInt(pix) for pix in cipherimg]