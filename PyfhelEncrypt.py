import PIL
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import os
from PIL import Image
import numpy as np
import pickle
from PyfhelHelpers import saveEncryptedImg, ImgEncrypt
import time 

HE = Pyfhel()
HE.contextGen(p=65537)
HE.keyGen()             
print(HE)

directory = 'test-images/Pediatric Chest X-ray Pneumonia/train/PNEUMONIA/'
image_count = 777
for file in os.listdir(directory):
    if image_count != 0:
        join = os.path.join(directory, file)
        image = PIL.Image.open(join, mode='r', formats=None)
        print(image_count)
        image_count -= 1

        start = time.time()
        enc_image= ImgEncrypt(HE, image)
        saveEncryptedImg(enc_image, 'enc_'+file)
        end = time.time()
        print('completed encryption: ', file)
        print(f'Elapsed time: {end - start}')

print('done')



'''
for row in cipherimg:
    row_gen = np.empty(len(row), dtype=PyCtxt)
    
    for i in np.arange(len(row)):
        row_gen[i] = HE.encryptFrac(row[i])
    
    x +=1
    print(x)
    arr_gen[row] = row_gen
'''