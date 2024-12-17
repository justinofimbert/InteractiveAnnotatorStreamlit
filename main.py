import streamlit as st
from streamlit_image_annotation import pointdet
import io
import csv
from PIL import Image
import numpy as np
import pandas as pd

overlap = 0.5

# Define label list
label_list = ['Positivo', 'Negativo', 'No importante']
actions = ['Agregar', 'Borrar']

# Initialize session state to store points and labels if not already present
if 'result_dict' not in st.session_state:
    st.session_state['result_dict'] = {}

if 'patches' not in st.session_state:
    st.session_state['patches'] = None  # Store generated patches here

if 'patch_index' not in st.session_state:
    st.session_state['patch_index'] = 0  # Store selected patch index

if 'label' not in st.session_state:
    st.session_state['label'] = 0  # Store selected label

if 'action' not in st.session_state:
    st.session_state['action'] = 0  # Store selected action

if 'all_points' not in st.session_state:
    st.session_state['all_points'] = set()  # Set to track unique point

if 'all_labels' not in st.session_state:
    st.session_state['all_labels'] = {}  # Dictionary to track labels for each unique point

# Initialize session state for csv and report data
if 'csv_data' not in st.session_state:
    st.session_state['csv_data'] = b""  # Use empty binary to avoid type errors

if 'report_data' not in st.session_state:
    st.session_state['report_data'] = b""  # Use empty binary to avoid type errors

if 'patch_points' not in st.session_state:
    st.session_state['patch_points'] = []

if 'patch_labels' not in st.session_state:
    st.session_state['patch_labels'] = []


def image_split(image, height=512, width=512, overlap=0):
    """
    Splits an image into patches of the same size (height x width) with an optional overlap.
    """
    imgs = []
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    overlap_height = int(height * overlap)
    overlap_width = int(width * overlap)
    
    vertical_splits = int(np.ceil((image_height - overlap_height) / (height - overlap_height)))
    horizontal_splits = int(np.ceil((image_width - overlap_width) / (width - overlap_width)))

    for y in range(vertical_splits):
        if ((y + 1) * (height - overlap_height)) < image_height:
            y_start = y * (height - overlap_height)
        else:
            y_start = image_height - height

        for x in range(horizontal_splits):
            if ((x + 1) * (width - overlap_width)) < image_width:
                x_start = x * (width - overlap_width)
            else:
                x_start = image_width - width

            split = image[y_start:y_start + height, x_start:x_start + width]
            imgs.append(split)

    return imgs, vertical_splits, horizontal_splits


def update_patch_data(session_state):

    current_patch_index = session_state['patch_index']
    vert_splits = session_state['vert_splits']
    hor_splits = session_state['hor_splits']
    overlap = session_state['overlap']

    current_patch_x_coord = (current_patch_index % hor_splits) * ( patch_width * (1-overlap) )
    current_patch_y_coord = (current_patch_index // vert_splits) * ( patch_height * (1-overlap) )

    all_points = session_state['all_points'] # Set to track unique point
    all_labels = session_state['all_labels'] # Dictionary to track labels for each unique point

    all_points = list(all_points)
    all_labels = [all_labels[point] for point in all_points]

    patch_points = []
    patch_labels = []

    for point, label in zip(all_points, all_labels):
        x, y = point

        x -= current_patch_x_coord
        y -= current_patch_y_coord

        if ( x>=0 and y>=0 and x<patch_width and y<patch_height ):

            point = [x, y]
            patch_points.append(point)
            patch_labels.append(label)


    session_state['patch_points'] = patch_points
    session_state['patch_labels'] = patch_labels


# Sidebar content
st.sidebar.title("Anotación de imágenes")

st.session_state['action'] = st.sidebar.selectbox("Acción:", actions)
st.session_state['label'] = st.sidebar.selectbox("Clase:", label_list)

# Image upload
uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_file is not None:

    uploaded_file_name = uploaded_file.name[:-4]

    # Open the uploaded image using PIL
    complete_image = Image.open(uploaded_file)
    
    # Display the complete image before any processing
    st.image(complete_image, caption="Imagen seleccionada", use_column_width=True)
    
    # Let the user select patch resolution
    col1, col2 = st.columns(2)
    patch_height = col1.number_input("Seleccionar altura de la sub-imagen", min_value=16, max_value=complete_image.size[1], value=512, step=16)
    patch_width = col2.number_input("Seleccionar ancho de la sub-imagen", min_value=16, max_value=complete_image.size[0], value=512, step=16)
    
    if st.button("Generar sub-imágenes"):
        splits, vert_splits, hor_splits = image_split(np.array(complete_image), height=patch_height, width=patch_width, overlap=overlap)
        st.session_state['vert_splits'] = vert_splits
        st.session_state['hor_splits'] = hor_splits
        st.session_state['overlap'] = overlap
        st.session_state['patches'] = splits
        st.session_state['patch_index'] = 0  # Reset to first patch


    if st.session_state['patches'] is not None:

        st.session_state['patch_index'] = st.selectbox(
            "Seleccionar sub-imagen a mostrar", 
            options=list(range(len(st.session_state['patches']))), 
            format_func=lambda x: f"Sub-imagen {x + 1}", 
            index=st.session_state['patch_index']
        )

        col1, col2 = st.columns([6, 6])
        with col1:
            if st.button("Anterior"):
                st.session_state['patch_index'] = max(0, st.session_state['patch_index'] - 1)
        with col2:
            if st.button("Siguiente"):
                st.session_state['patch_index'] = min(len(st.session_state['patches']) - 1, st.session_state['patch_index'] + 1)

        update_patch_data(st.session_state)

        img_path = f"patch_{st.session_state['patch_index']}.jpg"
        img = Image.fromarray(st.session_state['patches'][st.session_state['patch_index']])

        action = st.session_state['action']
        if action == actions[1]:
            mode = 'Del'
        else:
            mode = 'Transform'
                       
        # Use pointdet to annotate the image
        new_labels = pointdet(
            image_path=img_path,
            label_list=label_list,
            points=st.session_state['patch_points'],
            labels=st.session_state['patch_labels'],
            use_space=True,
            key=img_path,
            mode = mode,
            label = st.session_state['label']
        )
        
        # Update points and labels in session state if any changes are made
        if new_labels is not None:

            current_patch_index = st.session_state['patch_index']
            vert_splits = st.session_state['vert_splits']
            hor_splits = st.session_state['hor_splits']
            overlap = st.session_state['overlap']

            current_patch_x_coord = (current_patch_index % hor_splits) * ( patch_width * (1-overlap) )
            current_patch_y_coord = (current_patch_index // vert_splits) * ( patch_height * (1-overlap) )

            all_points = st.session_state['all_points'] # Set to track unique point
            all_labels = st.session_state['all_labels'] # Dictionary to track labels for each unique point

            patch_points = []

            # Add new points
            for v in new_labels:
                x, y = v['point']
                label_id = v['label_id']

                patch_points.append(v['point'])

                x += current_patch_x_coord
                y += current_patch_y_coord

                x = int(x)
                y = int(y)

                point_tuple = (x, y)

                if point_tuple not in all_points:
                    all_points.add(point_tuple)
                    all_labels[point_tuple] = label_id  # Store the label for this point

            # Remove points
            removed_points = []
            for global_point in all_points:
                x, y = global_point

                x -= current_patch_x_coord
                y -= current_patch_y_coord

                remove_flag = False

                if ( x>=0 and y>=0 and x<patch_width and y<patch_height ):

                    remove_flag = True

                    for patch_point in patch_points:

                        x_patch, y_patch = patch_point

                        nequal_flag = not ( (x == x_patch) and (y == y_patch) )

                        remove_flag = remove_flag and nequal_flag

                if remove_flag:
                    removed_points.append(global_point)

            for removed_point in removed_points:
                all_points.remove(removed_point)
                del all_labels[removed_point]  # Remove the corresponding label


            all_points = list(all_points)
            all_labels = [all_labels[point] for point in all_points]

            # Create CSV content
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(["X", "Y", "Label"])
            for point, label in zip(all_points, all_labels):
                csv_writer.writerow([point[0], point[1], label_list[label]])

            # Convert CSV buffer to downloadable file
            csv_data = csv_buffer.getvalue().encode('utf-8')

            # **Generate the Annotation Report**
            num_positive = all_labels.count(0)
            num_negative = all_labels.count(1)

            report_content = f"""
            Reporte de anotación
            ==================
            Nombre de la imagen: {uploaded_file_name}
            Número de puntos positivos: {num_positive}
            Número de puntos negativos: {num_negative}
            """

            # Create file-like object to download the report
            report_buffer = io.StringIO()
            report_buffer.write(report_content)
            report_data = report_buffer.getvalue()

            st.session_state['csv_data'] = csv_data
            st.session_state['report_data'] = report_data



        # Sidebar buttons
        with st.sidebar:
            # **1st Download Button** - CSV Annotations
            st.download_button(
                label="Descargar anotaciones (CSV)",
                data=st.session_state['csv_data'],
                file_name=f"{uploaded_file_name}.csv",
                mime="text/csv"
            )

            # **2nd Download Button** - Annotation Report
            st.download_button(
                label="Descargar Reporte (txt)",
                data=st.session_state['report_data'],
                file_name=f'{uploaded_file_name}.txt',
                mime='text/plain'
            )
