import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Title
st.title("Interactive Image Point Marker with Color Selection")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open the uploaded file as an image
    image = Image.open(uploaded_file)
    img_width, img_height = image.size
    img_array = np.array(image)  # Convert to NumPy array

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    

    # Add a color selector
    selected_color = st.radio(
        "Select Point Color",
        options=["Red", "Blue"],
        index=0,
        horizontal=True,
    )

    # Map color choices to actual color codes
    color_map = {
        "Red": "red",
        "Blue": "blue",
    }
    stroke_color = color_map[selected_color]

    # Add interactive canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color
        stroke_width=3,
        stroke_color=stroke_color,  # Use the selected color
        background_image=Image.fromarray(img_array),  # Convert NumPy array back to PIL image
        update_streamlit=True,
        height=img_height,  # Set canvas height
        width=img_width,  # Set canvas width
        drawing_mode="point",  # Enable point marking
        key="canvas",
    )

    # Count and display number of points
    if canvas_result.json_data is not None:
        points = canvas_result.json_data["objects"]
        num_points = len(points) if points else 0
        
        # st.write(f"Number of points clicked: {num_points}")

        # Display clicked coordinates and their colors
        # if points:
        #     st.write("Clicked Points Coordinates and Colors:")
        #     for point in points:
        #         st.write(f"Coordinates: ({point['left']:.1f}, {point['top']:.1f}), Color: {point['stroke']}")