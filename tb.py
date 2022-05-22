import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('MODELS/tb_model.h5')

def predict_tb(img_path):
    
    img = image.load_img(img_path, target_size=(224,224,3))
    img_arr = image.img_to_array(img)
    img_expand_dims = tensorflow.expand_dims(img_arr,0)

    p = model.predict(img_expand_dims)

    if p[0][1]>0.3:
        return("TB")
    else:
        return("Normal")

    
def saliency_map(img_path,check):

    img1 = image.load_img(img_path,target_size=(224,224))

    #preprocess image to get it into the right format for the model
    img = image.img_to_array(img1)
    img =  tensorflow.expand_dims(img,0)

    y_pred = model.predict(img)

    images = tensorflow.Variable(img, dtype=float)

    with tensorflow.GradientTape() as tape:
        pred = model(images, training=False)

        #returns index of class with maximum value
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
    
    # differentiating loss wrt images    
    grads = tape.gradient(loss, images)
    grad_abs = tensorflow.math.abs(grads)
    grad_max = np.max(grad_abs, axis=3)[0]

    ## normalize to range between 0 and 1
    pixel_min, pixel_max  = np.min(grad_max), np.max(grad_max)
    grad_eval = (grad_max - pixel_min) / (pixel_max - pixel_min)

    fig, axes = plt.subplots(1,2,figsize=(14,5))

    axes[0].imshow(img1)

    if check=="Normal":
        axes[1].imshow(img1)
    else:
        axes[1].imshow(grad_eval,cmap="jet",alpha=1)


    plt.savefig('static/saliency_img.png',dpi=300)
    plt.close()
