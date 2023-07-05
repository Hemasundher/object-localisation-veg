import tensorflow 
from tensorflow.keras.models import load_model

from utilities import *


model=load_model('model2.h5',compile=False)

#add path to image from local machine
path='C:/Users/hemas/Desktop/localisation -veg/test_images/test_image.jpg'

test_image=cv2.imread(path, cv2.IMREAD_COLOR)


test_image = cv2.resize(test_image, (227,227),
               interpolation = cv2.INTER_LINEAR)



show_result(test_image,model)