import streamlit as st
from streamlit_image_annotation import pointdet
import io
from PIL import Image

# Define label list
label_list = ['negative', 'positive', 'not important']

# Initialize session state to store points and labels if not already present
if 'result_dict' not in st.session_state:
    st.session_state['result_dict'] = {}

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    # Convert to a format that can be used in pointdet
    img_path = "uploaded_image.jpg"
    img.save(img_path)
    
    # Initialize points and labels if the image hasn't been annotated yet
    if img_path not in st.session_state['result_dict']:
        st.session_state['result_dict'][img_path] = {'points': [], 'labels': []}
    
    # Use pointdet to annotate the image
    new_labels = pointdet(
        image_path=img_path,
        label_list=label_list,
        points=st.session_state['result_dict'][img_path]['points'],
        labels=st.session_state['result_dict'][img_path]['labels'],
        use_space=True,
        key=img_path
    )
    
    # Update points and labels in session state if any changes are made
    if new_labels is not None:
        st.session_state['result_dict'][img_path]['points'] = [v['point'] for v in new_labels]
        st.session_state['result_dict'][img_path]['labels'] = [v['label_id'] for v in new_labels]
    
    # # Show the annotated image and the result
    # st.image(img, caption="Annotated Image", use_column_width=True)
    
    # Display the current annotations (points and labels)
    # st.json(st.session_state['result_dict'])
else:
    st.write("Please upload an image.")