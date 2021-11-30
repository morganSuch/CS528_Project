from PIL import Image
import numpy as np
import pickle


def ImgEncrypt(pyfhel, plainimg):
    
    cipherimg = np.asarray(plainimg)
    shape = cipherimg.shape
    cipherimg = cipherimg.flatten().tolist()
    cipherimg = [pyfhel.encodeInt(pix) for pix in cipherimg]
    
    return np.asarray(cipherimg).reshape(shape)


def saveEncryptedImg(cipherimg, filename):
    """
    args:
        cipherimg: Encryption of an image
        filename: filename to save encryption (saved under encrypted-images directory)
        
    saves Encryption of image int a file
    """
    # This needs to be updated to correspond to the current directory

    # "encrypted-images/test/NORMAL/"
    # "encrypted-images/test/PNEUMONIA/"
    # "encrypted-images/train/NORMAL/"
    # "encrypted-images/train/PNEUMONIA/"

    filename = "encrypted-images/train/NORMAL/" + filename
    fstream = open(filename, "wb")
    
    #pickle.dump(cipherimg, fstream)
    #for pix in cipherimg:
    #    dump = pickle.dumps(pix)
    #    fstream.write(dump)
    #cipherimg = [pyfhel.encodeInt(pix) for pix in cipherimg]
    
    fstream.write(cipherimg)
    fstream.close()


def loadEncryptedImg(filename):
    """
    args:
        filename: filename of the Encrypted object under encrypted-images directory
        
    returns:
        n-dimensional array containing ecryption of image
    loads Encrypted image object from file
    """
    filename = "encrypted-images/" + filename    
    fstream = open(filename, "rb")
    cipherimg = pickle.load(fstream)
    fstream.close()
    return cipherimg
