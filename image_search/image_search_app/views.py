from django.shortcuts import render
from annoy import AnnoyIndex
import tensorflow as tf
import math
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import os
from django.conf import settings

model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

batch_size = 64
root_dir = os.path.join(settings.MEDIA_ROOT, 'My_DataSet')

# Create the data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(root_dir, target_size=(224, 224), batch_size=batch_size, class_mode=None, shuffle=False)

num_images = len(generator.filenames)
num_epochs = int(math.ceil(num_images / batch_size))

# # Obtain the feature vectors using the pre-trained model
# feature_list = model.predict(generator, num_epochs)

# # Flatten the spatial dimensions of the feature vectors
# flatten_feature_list = [feature.flatten() for feature in feature_list]

# # Create and build the Annoy index
# annoy_index = AnnoyIndex(2048, 'angular')
# for i, feature in enumerate(flatten_feature_list):
#     annoy_index.add_item(i, feature)
# annoy_index.build(40)

# Construct the full path to the .ann file
annoy_index_path = os.path.join(os.path.dirname(__file__), '..', 'media', 'My_DataSet', 'annoy_caltech101index.ann')

# Load the Annoy index
annoy_index = AnnoyIndex(2048, 'angular')
annoy_index.load(annoy_index_path)

def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        external_image = Image.open(image_file)
        external_image = external_image.resize((224, 224))
        external_image_array = preprocess_input(tf.keras.preprocessing.image.img_to_array(external_image))

        # Obtain the feature vector for the external image
        external_feature = model.predict(tf.expand_dims(external_image_array, axis=0)).flatten()

        # Retrieve the nearest neighbors for the specific external image
        result = annoy_index.get_nns_by_vector(external_feature, 9, include_distances=True)
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
