import cv2
import numpy as np
import matplotlib.pyplot as plt

label={'no image':0,'cucumber':1,'eggplant':2,'mushroom':3}
label_to_name={1:'cucumber',2:'eggplant',3:'mushroom',0:'no image'}


def draw_rectangle(img,bounds,color='red'):
    colors={'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255)}
    c=colors.get(color.lower(),(255,0,0))
    return cv2.rectangle(img,(int(bounds[0]),int(bounds[1])),(int(bounds[2]),int(bounds[3])),c)

def add_text_to_boundingbox1(img,cat,bounds,color='red'):
    colors={'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255)}
    c=colors.get(color.lower(),(255,0,0))
    return cv2.putText(img,label_to_name[cat],(int(bounds[0]),int(bounds[1])-5),cv2.FONT_HERSHEY_SIMPLEX, .5,c,1,cv2.LINE_AA)


def add_text_to_boundingbox2(img,bounds,t,color='red'):
    colors={'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255)}
    c=colors.get(color.lower(),(255,0,0))
    
    return cv2.putText(img,t,(int(bounds[0]),int(bounds[1])-5),cv2.FONT_HERSHEY_SIMPLEX, .5,c,1,cv2.LINE_AA)

def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  pred_box
    xmin_true, ymin_true, xmax_true, ymax_true = true_box

    #Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    #Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    #Calculates overlap area and union area.
    overlap_area = np.maximum((xmax_overlap - xmin_overlap),0)  * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    # Defines a smoothing factor to prevent division by 0
    smoothing_factor = 1e-10

    #Updates iou score
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return np.round(iou,3)


def show_result(img,model):
    img=img/255
    pred=(model.predict(img.reshape(1,227,227,3)))

    bound=pred[1][0]
    cat=np.argmax(pred[0][0])
    if cat!=0:

        x= draw_rectangle(img,bound)

        x=add_text_to_boundingbox2(x,bound,'predicted box')
        plt.imshow(x)
        plt.title(label_to_name[cat])
        plt.show()
        
    else:
        plt.imshow(img)
        plt.title(label_to_name[cat])
        plt.show()

