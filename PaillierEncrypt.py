import PIL
import os
from ImageCryptography import ImgEncrypt, saveEncryptedImg
from Paillier import generate_keys

keys = generate_keys()
public_key = keys[0]
private_key = keys[1]
#print(public_key)
#print(private_key)


# This needs to be updated for each of the 4 directories the files lie in
# 'test-images/Pediatric Chest X-ray Pneumonia/test/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/test/PNEUMONIA/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/NORMAL/'
# 'test-images/Pediatric Chest X-ray Pneumonia/train/PNEUMONIA/'
directory = 'test-images/Pediatric Chest X-ray Pneumonia/test/NORMAL/'
image_count = 0
for file in os.listdir(directory):

    join = os.path.join(directory, file)
    image = PIL.Image.open(join, mode='r', formats=None)

    image_count += 1
    print(image_count)
    enc_image= ImgEncrypt(public_key, image)

    saveEncryptedImg(enc_image, file)
    print('completed encryption: ', file)
    b = os.path.getsize('encrypted_images/test/NORMAL/'+ file)
    print('size: ',b)