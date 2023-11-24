import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from midv500models.pre_trained_models import create_model as midv500_create_model
import segmentation_models_pytorch as smp
import albumentations as albu
from ArabicOcr import arabicocr
from check_orientation.pre_trained_models import create_model as orientation_create_model
from image import *
# Function to load image as a NumPy array
def get_image(img):
    image = Image.open(img)
    return np.asarray(image)

# Function to perform Arabic OCR on the corrected image and get the text result
def arabic_ocr_on_corrected_image(image_path, output_path):
    results = arabicocr.arabic_ocr(image_path, output_path)
    for result in results:
        text = result[1]
        confidence = result[2]
        print(confidence)
        if confidence >= 0.5:
            print(f"Detected Text: {text} - Confidence: {confidence}")

def main():
    st.markdown("<h1 style='text-align: center;position:static; top:-50 ;color: rgb(65, 59, 59);font-size:300%;'>IDs Correction with Text Extraction</h1>",
                unsafe_allow_html=True)
    side_bar = st.sidebar
    image_file = side_bar.file_uploader('', type=['png', 'jpg', 'jpeg'])
    col1, col2 = st.columns(2)
    
    if image_file is not None:
        image = get_image(image_file)
        start_btn = side_bar.button('Start OCR')
        
        if start_btn:
            model_segment = midv500_create_model("Unet_resnet34_2020-05-19")
            model_segment.eval()

            # Image processing steps
            image = cv2.resize(image, (470, 600))
            transform = albu.Compose([albu.Normalize(p=1)], p=1)
            padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
            x = transform(image=padded_image)["image"]
            x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
            with torch.no_grad():
                prediction = model_segment(x)[0][0]
            mask = (prediction > 0).cpu().numpy().astype(np.uint8)
            mask = unpad(mask, pads)
            final_image = extract_idcard(image, mask)  # You need to define extract_idcard function
            
            model_rotate = orientation_create_model("swsl_resnext50_32x4d")
            model_rotate.eval()
            transform_rotate = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
            
            temp = []
            for k in [0, 1, 2, 3]:
                x = transform_rotate(image=np.rot90(final_image, k))["image"]
                temp += [tensor_from_rgb_image(x)]
            
            with torch.no_grad():
                prediction = model_rotate(torch.stack(temp)).numpy()
            
            for i in range(4):
                want = [round(tx, 2) for tx in prediction[i]]
                if want[0] == max(want):
                    corrected_image = np.rot90(final_image, i)
            
            # Define new dimensions and resize the corrected image
            new_width = 2000
            new_height = 1800
            resized_image = cv2.resize(corrected_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Save the corrected image and perform Arabic OCR on it
            cv2.imwrite("corrected_image.jpg", resized_image)
            arabic_ocr_on_corrected_image('corrected_image.jpg', 'out.jpg')
            
            # Display the corrected and OCR annotated image in Streamlit columns
            corrected_image = Image.fromarray(corrected_image)
            #col1.text('Image')

            col1.text('Starting image')
            col1.image(image, use_column_width=True)
           
            col1.text('Image After Correction')
            col1.image('corrected_image.jpg', use_column_width=True)

            col1.text('Ocr on Image')
            col1.image('out.jpg', use_column_width=True)

            

if __name__ == "__main__":
    main()
