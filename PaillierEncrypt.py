import PIL
import os
from ImageCryptography import ImgEncrypt, saveEncryptedImg
from Paillier import generate_keys

keys = generate_keys()
public_key = keys[0]
private_key = keys[1]
#print(public_key)
#print(private_key)



directory = 'test-images/Pediatric Chest X-ray Pneumonia/test/NORMAL/'
image_count = 5
for file in os.listdir(directory):
    if image_count != 0:
        join = os.path.join(directory, file)
        image = PIL.Image.open(join, mode='r', formats=None)

        image_count -= 1

        enc_image= ImgEncrypt(public_key, image)
    # print(enc_image)

        saveEncryptedImg(enc_image, file)
        print('completed encryption: ', file)
        b = os.path.getsize('encrypted_images/test/NORMAL/'+ file)
        print('size: ',b)
    # dec_image = ImgDecrypt(public_key, private_key,enc_image)

    # saveDecryptedImg(dec_image, 'dec_noserialize_lungs')    # test


#print(image_count)