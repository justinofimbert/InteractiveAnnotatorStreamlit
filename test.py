import pandas as pd
import matplotlib.pyplot as plt
import cv2  # For image loading (you can also use PIL if preferred)

def overlay_points_on_image(csv_path, image_path):
    """
    Reads a CSV file with "X", "Y", and "Label" columns and overlays these points on an image.
    
    Args:
    - csv_path (str): Path to the CSV file.
    - image_path (str): Path to the image file.
    """
    # 1. Load the image using OpenCV (convert BGR to RGB for matplotlib)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Read the CSV file
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['X', 'Y', 'Label']):  # Corrected column name
        raise ValueError('CSV file must contain columns "X", "Y", and "Label".')
    
    x_coords = df['X'].values
    y_coords = df['Y'].values
    labels = df['Label'].values  # Corrected column name

    # Map categorical labels to integers (optional)
    unique_labels = list(df['Label'].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_int[label] for label in labels]

    # 3. Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.scatter(x_coords, y_coords, c=numeric_labels, cmap='rainbow', edgecolors='k', s=50, label='Points')
    plt.colorbar(label="Labels")
    plt.title('Overlay of Points from CSV on Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert Y-axis if image coordinates and plot coordinates don't match
    plt.show()


# === TEST VARIABLES ===
csv_path = 'points.csv'  # Replace with the path to your CSV file
image_path = 'uploaded_image.jpg'  # Replace with the path to your image file

# === RUN FUNCTION ===
overlay_points_on_image(csv_path, image_path)