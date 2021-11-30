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

# This needs to be updated for each of the 4 directories the files lie in

# 'test-images/Pediatric Chest X-ray Pneumonia/test/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/test/PNEUMONIA/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/PNEUMONIA/'
directory = 'test-images/Pediatric Chest X-ray Pneumonia/train/NORMAL/'
image_count = 0
for file in os.listdir(directory):
    join = os.path.join(directory, file)
    image = PIL.Image.open(join, mode='r', formats=None)
    image_count += 1

    print(image_count)
    
    

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