from django.shortcuts import render
import sys
from importlib import import_module
from PIL import Image
import os
from django.conf import settings

# Get the path to the directory containing views.py
current_dir = os.path.dirname(__file__)

# Calculate the path to base.py using the parent directory
base_path = os.path.join(current_dir, "..", "..", "base.py")

# Normalize the path
base_path = os.path.normpath(base_path)

# Extract the directory and module name from the path
base_dir, base_module = os.path.split(base_path)
base_module = os.path.splitext(base_module)[0]

# Add the base directory to the sys.path
sys.path.append(base_dir)

# Import base.py as a module
base = import_module(base_module)
######################################
gen = base.base_code(1)

root_dir = next(gen)
tf = next(gen)
model = next(gen)
generator = next(gen)
num_epochs = next(gen)
annoy_index = next(gen)

# model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# batch_size = 64
# root_dir = os.path.join(settings.MEDIA_ROOT, 'My_DataSet')

# # Create the data generator
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
# generator = datagen.flow_from_directory(root_dir, target_size=(224, 224), batch_size=batch_size, class_mode=None, shuffle=False)

# num_images = len(generator.filenames)
# num_epochs = int(math.ceil(num_images / batch_size))

# Construct the full path to the .ann file
annoy_index_path = os.path.join(os.path.dirname(__file__), '..', 'media', 'My_DataSet', 'annoy_index.ann')

# Load the Annoy index
annoy_index.load(annoy_index_path)

def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # image_file = request.FILES['image']
        # external_image = Image.open(image_file)
        # external_image = external_image.resize((224, 224))
        # external_image_array = preprocess_input(tf.keras.preprocessing.image.img_to_array(external_image))
        external_feature = base.process_single_image(image_file)
        # Retrieve the nearest neighbors for the specific external image
        result = annoy_index.get_nns_by_vector(external_feature, 6, include_distances=True)
        print(result)
        # Get the class and image name for nearest neighbors
        nearest_neighbors_indices = result[0]
        similar_images = []
        for neighbor_index in nearest_neighbors_indices:
            class_name = generator.filenames[neighbor_index].split('/')[0]
            image_name = generator.filenames[neighbor_index].split('/')[1]
            image_url = f"{settings.MEDIA_URL}My_DataSet/{class_name}/{image_name}"
            similar_images.append({
                'image_url': image_url,
            })
            

        return render(request, 'home.html', {'result': similar_images})

    return render(request, 'home.html')
