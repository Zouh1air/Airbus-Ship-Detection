import torch
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
import subprocess

os.chdir("C:\\Users\\hp\\Desktop\\MINI-PROJECT\\deploy\\")

#SegNet Model
# Define the IoU (Intersection over Union) metric function
def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def onehot_to_mask(onehot):
    mask = np.argmax(onehot, axis=-1)
    return mask

# Register the custom metric function when loading the model
segnet_model = load_model("Ship_detection_model.h5", custom_objects={"iou_coef": iou_coef})

#U-Net Model
# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the model input size
    image = image.resize((768, 768))
    # Convert the image to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    # Add an extra dimension to represent batch size
    image = np.expand_dims(image, 0)
    return image

# Function to apply the segmentation model and predict the boat
def predict_boat(image):
    # Apply the segmentation model to the image
    segmentation_mask = Unet_model.predict(image)
    # Extract the boat prediction (class 1) from the segmentation mask
    boat_mask = segmentation_mask[0, :, :, 0]
    return boat_mask

# Load the U-NET model
Unet_model = load_model("fullres_model.h5")


# Yolov5 Model
# Load the YOLOv5 model
weights = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    for detection in results.xyxy[0]:
        bbox = detection[:4].tolist()
        class_index = int(detection[5])
        class_name = model.names[class_index]
        confidence = detection[4].item()
        # Draw the bounding box on the image
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1] - 10), f"Ship: {confidence:.2f}", fill="red")
    return image

def count_folders(directory):
    folder_count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_count += 1
    return folder_count

def video_display(video_path):
    if video_path:
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

# Streamlit app
def main():
    st.title("Ship Detection")
    # Upload image file
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg","mp4"])
    model_selection = st.selectbox("Select the model", ["Yolov5","U-NET", "SegNet"])
    submit_button = st.button("Submit")
    if uploaded_file is not None and submit_button:
        # Read the uploaded image
        if model_selection == "U-NET":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Button to predict the boat
            
            # Apply the model and get the boat mask
            boat_mask = predict_boat(preprocessed_image)

            # Convert the boat mask to a binary mask
            binary_mask = np.where(boat_mask > 0.5, 255, 0).astype(np.uint8)

            # Resize the binary mask to match the size of the input image
            resized_mask = cv2.resize(binary_mask, (image.width, image.height))

            # Apply the binary mask to the input image
            masked_image = cv2.bitwise_and(np.array(image), np.array(image), mask=resized_mask)

            # Display the masked image
            st.image(masked_image, caption="Boat Segmentation Mask", use_column_width=True)
        elif model_selection =="SegNet":
            # Load and preprocess the image
            img = load_img(uploaded_file, target_size=(512, 512))
            tmp_img = img_to_array(img)
            tmp_img = tf.expand_dims(tmp_img, 0) / 255.0

            # Perform prediction
            prediction = segnet_model.predict(tmp_img)
            tmp_mask = onehot_to_mask(prediction[0])

            # Resize the mask to match the size of the input image
            tmp_mask = cv2.resize(tmp_mask.astype(np.uint8), (img.width, img.height))

            # Convert the input image and mask to NumPy arrays
            img_array = img_to_array(img).astype(np.uint8)
            mask_array = np.zeros_like(img_array)

            # Set the color of ship pixels to red (BGR format)
            mask_array[tmp_mask == 1] = (0, 0, 255)

            # Apply the mask on the overlay image
            overlay_img = cv2.addWeighted(img_array, 0.5, mask_array, 0.5, 0)

            # Display the input image
            st.subheader("Input Image")
            st.image(img, caption="Input Image", use_column_width=True)

            # Display the overlay image
            st.subheader("Overlay Image")
            st.image(overlay_img, caption="Overlay Image", use_column_width=True)
        else:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_path = getattr(uploaded_file, "name", None)
            if file_extension != ".mp4":
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Apply object detection on the uploaded image
                results = model(image)

                # Draw bounding boxes on the image
                annotated_image = draw_boxes(image.copy(), results)

                # Display the annotated image
                st.image(annotated_image, caption='Annotated Image', use_column_width=True)
            else :
                command = f'python "C:\\Users\\hp\\Desktop\\MINI-PROJECT\\yolov5\\detect.py" --source "{file_path}"'
                # Run the command and capture the output
                try:
                    # Apply object detection on the uploaded image
                    output = subprocess.check_output(command, shell=True)
                    folder_path = 'C:\\Users\\hp\\Desktop\\MINI-PROJECT\\yolov5\\runs\\detect'
                    num_folders = count_folders(folder_path)
                except subprocess.CalledProcessError as e:
                    num_folders = 0 
                    st.error(f"Error executing command: {e}")
                #Sélectionner le fichier vidéo
                video_path =f'C:\\Users\\hp\\Desktop\\MINI-PROJECT\\yolov5\\runs\\detect\\exp{num_folders}\\{file_path}'
                # Lire la vidéo initiale
                video_bytes = uploaded_file.read()
                st.write('uploaded video')
                st.video(video_bytes)
                # Lire la vidéo resultant
                st.write('video frames')
                # Ouvrir la vidéo à partir du chemin spécifié
                video = cv2.VideoCapture(video_path)
                # Lecture de la vidéo frame par frame
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    #Conversion du cadre en couleurs BGR en RVB pour Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #Affichage du cadre vidéo
                    st.image(frame_rgb, channels="RGB")
                video.release()
    elif submit_button:
        st.error("Please upload an image")
if __name__ == "__main__":
    main()

