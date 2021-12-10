import streamlit as st
import dlib
import numpy as np
import cv2
import tensorflow as tf
import keras
import os
from PIL import Image


#Page Configs
st.set_page_config(page_title='Face Studio', page_icon = './assets/icon.png', layout = 'wide', initial_sidebar_state = 'auto')
k1, k2, k3 = st.columns(3)
k2.title('Face Studio')
#Loading model for hair segmentation
model = keras.models.load_model('./checkpoints/model/checkpoint.hdf5')
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

#Functions for Lip Coloring

def createBoundBox(img,points,teeth_points, scale=2, masked=False, cropped=True):
    #converting lists to numpy array
    points = np.array(points)
    teeth_points = np.array(teeth_points)

    if masked:
        #creating mask only for lips
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points], (255,255,255))
        mask = cv2.fillPoly(mask,[teeth_points], (0,0,0))
        img = cv2.bitwise_and(img, mask)

    if cropped:
        #getting starting (x,y) coordinate, width and height of image
        x,y,w,h = cv2.boundingRect(points)
        crop = img[y:y+h, x:x+w]
        crop = cv2.resize(crop,(0,0),fx=scale, fy=scale)
        return crop
    else:
        return mask



#Functions for Hair Coloring

def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    pred = model.predict(im)

    mask = pred.reshape((224, 224))

    return mask


def transfer(image, mask):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 2] = mask

    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst


def getHead(hog_face_detector, image):
    faces_hog = hog_face_detector(image, 1)

    heads = []
    
    for face in faces_hog:
        
        head = dict()
        
        head["left"] = max(face.left() - 300, 0)
        head["top"] = max(face.top() - 300, 0)
        head["right"] = min(face.right() + 300, image.shape[0])
        head["bottom"] = min(face.bottom() + 300, image.shape[1])
        
        heads.append(head)

    return heads


#Turning hex code to RGB
def hex2rgb(str1):
    str1 = str1.lstrip('#')
    return tuple(int(str1[i:i+2], 16) for i in (0, 2, 4))

#---------------------------------------------------------------
#Sidebar Configs
side = st.sidebar
side.image('./assets/icon.png')
select = side.selectbox('Select Region', options=('Select','Lips', 'Hair', 'Final Mix'))
reset = side.button('Reset Changes',key=1)
#-----------------------------------------------------------------

#Main Code

img_uploaded = k2.file_uploader('Upload Your Image', type=['png', 'jpg', 'webp'])
if img_uploaded is not None:
    if 'lips_arr' not in st.session_state:
        st.session_state.lips_arr = []
    if 'hair_arr' not in st.session_state:
        st.session_state.hair_arr = []
    if reset:
        st.session_state.lips_arr.clear()
        st.session_state.hair_arr.clear()
        st.session_state.facePoints.clear()
        select = 'Select'
    uploaded_img = Image.open(img_uploaded)
    # k2.image(uploaded_img)
    img = np.array(uploaded_img)
    img = cv2.resize(img, (350,350), cv2.INTER_AREA)
    temp = img.copy()
    new_img = img.copy()
    
    if select == 'Lips':
        col = k2.color_picker('Pick a Color')
        img_copy = new_img.copy()
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_bw)
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            landmarks = predictor(img_bw, face)
            x = [landmarks.part(i).x for i in range(68)]
            y = [landmarks.part(i).y for i in range(68)]
            face_points = [[i,j] for i,j in zip(x,y)]  
            st.session_state.facePoints = face_points
        #Cropping Lips section
        lips = createBoundBox(img, face_points[48:61], face_points[60:69], masked=True, cropped=False)

        #Coloring Lips
        lip_color_img = np.zeros_like(lips)
        r,g,b = hex2rgb(col)

        lip_color_img[:] = r,g,b
        lip_color_img = cv2.bitwise_and(lips, lip_color_img)
        lip_color_img = cv2.GaussianBlur(lip_color_img, (7,7),10)
        st.session_state.lips_arr.append(lip_color_img)
        lip_color_img = cv2.addWeighted(img_copy, 1, lip_color_img,0.4,0)
        # img1 = lip_color_img
        k2.image(Image.fromarray(lip_color_img))
        
    
    elif select =='Hair':
        imgh = new_img.copy()
        heads = getHead(face_detector, imgh)
        for head in heads:
            col = k2.color_picker('Pick a Color')
            imgh = imgh[head["top"]:head["bottom"], head["left"]:head["right"]]
            # img = cv2.resize(img, (350,350))

            mask = np.zeros_like(imgh)
            mask = predict(imgh)
                
            mask[mask > 0.5] = 255
            mask[mask <= 0.5] = 0
                

            mask = cv2.resize(mask, (imgh.shape[1], imgh.shape[0]))
            mask = mask.reshape((imgh.shape[1], imgh.shape[0], 1))
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            mask_img = np.zeros_like(imgh)
            mask_img[:,:] = mask
            mask_n = np.zeros_like(mask)
            r,g,b = hex2rgb(col)
            mask_n[:] = r,g,b
            mask_n = cv2.bitwise_and(mask, mask_n)
            mask_nn = np.zeros_like(imgh)
            mask_nn[:, :] = mask_n
            st.session_state.hair_arr.append(mask_nn)

            alpha = 0.8
            beta = (1.0 - alpha)
            dst = cv2.addWeighted(imgh, 1, mask_nn, 0.4,0, mask_nn)
            # img1 = dst
            k2.image(Image.fromarray(dst))

        
    elif select == 'Final Mix':
        eyeLiner = side.checkbox('Eyeliner')
        img1 = new_img.copy()
        lips_mask = st.session_state.lips_arr[-1]
        hair_mask = st.session_state.hair_arr[-1]
        alpha = 0.4
        beta = (1.0 - alpha)
        fin_out = cv2.addWeighted(img1, alpha, hair_mask, beta,0)
        fin_out = cv2.addWeighted(fin_out, 1, lips_mask,0.4,0)
        if eyeLiner:
            eyefin = fin_out.copy()
            eyefin = cv2.polylines(eyefin, [np.array(st.session_state.facePoints[36:42])], True, (0,0,0),1, cv2.LINE_AA)
            eyefin = cv2.polylines(eyefin, [np.array(st.session_state.facePoints[42:48])], True, (0,0,0),1, cv2.LINE_AA)
            k2.image(Image.fromarray(eyefin))
        else:
            k2.image(Image.fromarray(fin_out))



