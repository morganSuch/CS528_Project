import PIL
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import os
from PIL import Image
import numpy as np
import pickle
from Encryption.PyfhelHelpers import saveEncryptedImg, ImgEncrypt
import time 

HE = Pyfhel()
HE.contextGen(p=65537)
HE.keyGen()             
print(HE)

# This needs to be updated for each of the 4 directories the files lie in

# 'test-images/Pediatric Chest X-ray Pneumonia/test/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/test/PNEUMONIA/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/PNEUMONIA/'
directory = 'test-images/Pediatric Chest X-ray Pneumonia/train/NORMAL/'
image_count = 10
running_time = time.time()

while image_count > 0:
    for file in os.listdir(directory):
        print('processing image',image_count, '...\n')
        join = os.path.join(directory, file)
        image = PIL.Image.open(join, mode='r', formats=None)
        #image_count += 1
        print('size of unencrypted image:', os.path.getsize(join))


    

        start = time.time()
        enc_image= ImgEncrypt(HE, image)
        saveEncryptedImg(enc_image, 'enc_'+file)
        end = time.time()
        #print('completed encryption: ', file)
        print(f'Elapsed time: {end - start}\n')
        image_count -= 1

end_running_time = time.time()
print(f'avg time: {end_running_time- running_time}')
print('done')