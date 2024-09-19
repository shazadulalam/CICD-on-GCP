import os
import sys
sys.path.append(os.path.realpath('.'))

import numpy as np
from cicd_on_gcp.model.train import create_model, load_data

def test_model_prediction():
    (_, _), (test_images, test_labels) = load_data()
    test_images = np.expand_dims(test_images, axis=-1)

    model = create_model()
    small_batch_images = test_images[:10]
    predictions = model.predict(small_batch_images)

    assert predictions.shape == (10, 10)
    assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-6)

def test_model_evaluation():
    # Load the data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Reshape test images to add a channel dimension (required by the model)
    test_images = np.expand_dims(test_images, axis=-1)

    # Create and compile the model
    model = create_model()

    # Evaluate the model on a small batch of data
    loss, accuracy = model.evaluate(test_images[:100], test_labels[:100], verbose=0)

    # Check if the loss and accuracy are reasonable numbers
    assert loss >= 0  # Loss should not be negative
    assert 0 <= accuracy <= 1  # Accuracy should be a probability (between 0 and 1)
