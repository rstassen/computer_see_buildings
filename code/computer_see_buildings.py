#import libraries
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import segmentation_models as sm

from PIL import Image
from keras import optimizers

##################################################
# Background Code
##################################################

# image transform function
def image_transform(img):

    # importing image as grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crop image
    x = img.shape[1]
    y = img.shape[0]
    cropped_img = gray[0:min(x,y), 0:min(x,y)]

    # resize image
    resize_img = cv2.resize(cropped_img, (256,256))

    # transform image to 3-channel grayscale, to satisfy model's dimension requirements
    # this code was modified from Mona Jalal
    # https://askubuntu.com/questions/1091493/convert-a-1-channel-image-to-a-3-channel-image
    img_out = np.zeros((256,256,3), dtype='int')
    img_out[:,:,0] = resize_img
    img_out[:,:,1] = resize_img
    img_out[:,:,2] = resize_img
    
    return np.expand_dims(img_out, axis=0)

# import and compile CNN model
model = tf.keras.models.load_model('../saved_models/building_segmentation_v2.h5', compile = False)
model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=sm.metrics.iou_score)


##################################################
# UI
##################################################

st.title("Computer.  See.  Buildings.")
uploaded_image = st.file_uploader('Import .png file', type=['png'])
h_meters = st.number_input('Input horizontal image extent in meters', step=1)

# Columns for image display
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    
  
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Formatted Image</p>',unsafe_allow_html=True)
        converted_img = np.array(img.convert('RGB'))
        formatted_img = image_transform(converted_img)
        st.image(formatted_img, width=300)

    with col2:
        st.markdown('<p style="text-align: center;">Predicted Building Footprints</p>',unsafe_allow_html=True)
        prediction = model.predict(formatted_img)
        prediction_squeezed = np.squeeze(prediction, axis = 0)
        prediction_squeezed = np.squeeze(prediction_squeezed, axis = 2)
        st.image(prediction_squeezed, width=300)
        
    if h_meters > 0:
        # finding % pixels vacant
        positive_class = 0
        negative_class = 0
        for row in prediction_squeezed:
            for pixel in row:
                if pixel > 0.5:
                    positive_class += 1
                else:
                    negative_class += 1

        # finding vacant area in m**2
        in_pixels = converted_img.shape[1]
        out_pixels = min(converted_img.shape[0], converted_img.shape[1])

        h_pix_change = in_pixels - out_pixels
        perc_chg = h_pix_change / in_pixels

        h_meters *= 1 - perc_chg
        pixel_len = h_meters / 256

        total_area = (pixel_len * 256)**2
        vacant_area = (total_area * (negative_class/65536))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Area: ", f"{int(total_area)} sqm")
        col2.metric("Building Coverage Area: ", f"{int(positive_class/(positive_class+negative_class)*100)}%")
        col3.metric("Estimated Vacant Space: ", f"{int(vacant_area/2)} sqm")