import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
import random
import os
import imghdr
import streamlit as st
import pickle as pk
import cv2
import requests
from PIL import Image
from io import BytesIO
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model, save_model


with st.sidebar:
    st.title("Home Page")
    selection=st.radio("select your option",options=["upload an image", "Insert Image url"])

def sort_extension():
    
    data_dir = '/content/Dataset/data'
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image in os.listdir(data_dir):
        image_path = os.path.join(data_dir,image)
    try:
        img = cv2.imread(image_path)
        tip = imghdr.what(image_path)
        if tip not in image_exts:
            print('Image not in ext list {}'.format(image_path))
            os.remove(image_path)
    except Exception as e:
        print('Issue with image {}'.format(image_path))


def saving_into_dataFrame():
    #saving the file into a dataframe

    filenames = os.listdir('/content/Dataset/data')

    categories = []
    choosen_filename=[]
    for filename in filenames:
        category = filename.split('.')[0]
        if 'card' in category.lower():
            categories.append(0)
            choosen_filename.append(filename)
        elif 'met' in category.lower():
            categories.append(1)
            choosen_filename.append(filename)
        elif 'paper' in category.lower():
            categories.append(2)
            choosen_filename.append(filename)
        elif 'plas' in category.lower():
            categories.append(3)
            choosen_filename.append(filename)

    print(len(choosen_filename), len(categories))
    df = pd.DataFrame({'filename': choosen_filename, 'category': categories,})



# def download_and_save_image(image_url, save_path="downloaded_image.png"):
#     response = requests.get(image_url)

#     if response.status_code == 200:
#         image_data = BytesIO(response.content)
#         img = Image.open(image_data)
#         saved_img_path="./"+save_path
#         img.save(saved_img_path)
#         st.success("Image downloaded and saved successfully.")
#     else:
#         st.error(f"Failed to download the image. Status code: {response.status_code}")


# import requests
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.image import resize
# from tensorflow.keras.models import load_model
# import numpy as np
# from io import BytesIO

def download_and_save_image(image_url, save_path="downloaded_image.png"):
    '''response = requests.get(image_url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        img = Image.open(image_data).convert("RGB")
        saved_img_path="./"+save_path
        img.save(saved_img_path)
        resize_img = resize(img, (150, 150))
    
        img_array = img_to_array(resize_img)
        img_array = np.expand_dims(img_array, axis=0)
    
        img_array_copy = img_array.copy()
        img_array_copy /= 255.0
    
        loaded_model = load_model("Waste_Management_Model.h5")
    
        if st.button("Predict"):
            prediction = loaded_model.predict(img_array_copy)
            predicted_class = np.argmax(prediction)
            
            class_labels = {0: 'cardboard', 1: 'metal', 2: 'paper', 3: 'plastic'}
            predicted_category = class_labels[predicted_class]
            
            result = f"This Item is a {predicted_category}"
            st.success(result)
            st.image(img, caption=None)
    
        
    else:
        st.error(f"Failed to download the image. Status code: {response.status_code}")'''

    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    
        image_data = BytesIO(response.content)
        img = Image.open(image_data).convert("RGB")
        saved_img_path = "./" + save_path
        img.save(saved_img_path)
        resize_img = resize(img, (150, 150))
    
        img_array = img_to_array(resize_img)
        img_array = np.expand_dims(img_array, axis=0)
    
        img_array_copy = img_array.copy()
        img_array_copy /= 255.0

        loaded_model = load_model("Waste_Management_Model.h5")

        if st.button("Predict"):
            prediction = loaded_model.predict(img_array_copy)
            predicted_class = np.argmax(prediction)
    
            class_labels = {0: 'cardboard', 1: 'metal', 2: 'paper', 3: 'plastic'}
            predicted_category = class_labels[predicted_class]


            if predicted_category == "cardboard":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "metal":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "plastic":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "paper":
                result=f"This Item is a {predicted_category}, it should be disposed"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)

            else:
                print("")

    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.RequestException as err:
        st.error(f"Failed to download the image, The Image link is not a downloadble link")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def image_url_input():
    st.title("Classification with URL")
    image_url = st.text_input("Enter the image url and press enter", key="akaska")    
    # if st.button("Download Image"):
    if image_url != "":
        download_and_save_image(image_url)










def insert():
    st.title("Upload Image to Classify")
    

    # File uploader widget
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png","bmp"], key="upl")

    if uploaded_file is not None:
        # Convert the uploaded image to RGB
        img = Image.open(uploaded_file).convert("RGB")
        # Resize the image using TensorFlow
        resize_img = resize(img, (150, 150))

        # Convert the resized image to an array
        img_array = img_to_array(resize_img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

        # Make a copy of the array and then rescale
        img_array_copy = img_array.copy()
        img_array_copy /= 255.0  # Rescale to values between 0 and 1


        # To load the model
        loaded_model = load_model("Waste_Management_Model.h5")
        # Make the prediction
        
        if st.button("Predict"):
            prediction = loaded_model.predict(img_array_copy)
            predicted_class = np.argmax(prediction)

            # Map the class label to its corresponding category
            class_labels = {0: 'cardboard', 1: 'metal', 2: 'paper', 3: 'plastic'}
            predicted_category = class_labels[predicted_class]

            

            if predicted_category == "cardboard":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "metal":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "plastic":
                result=f"This Item is a {predicted_category}, it should be recycled"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)
            elif predicted_category == "paper":
                result=f"This Item is a {predicted_category}, it should be disposed"
                # Print the prediction
                st.success(result)
                st.image(img, caption=None)






# def all_predict():
#     st.title("Upload item to predict")
#     # File uploader widget
#     uploaded_file = st.file_uploader("select the folder containing your image", key="jacking")

#     if uploaded_file is not None:
#         print("")



# if selection == "Predict a Single Item":
#     main()

# if selection == "Predict for Multi-Patient":
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     #---------------------------------#
#     # Prediction
#     #--------------------------------
#     #---------------------------------#
#     # Sidebar - Collects user input features into dataframe
#     st.header('Upload your image file here')
#     uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"],key="jjj")
#     #--------------Visualization-------------------#
#     # Main panel
    
#     # Displays the dataset
#     # if uploaded_file is not None:
#     #     #load_data = pd.read_table(uploaded_file).
#     #     multi(uploaded_file)
#     # else:
#     #     st.info('Upload your dataset !!')


if selection== "upload an image":
    insert()

if selection =="Insert Image url":
    image_url_input()

# if __name__ == "__main__":
#     main()
