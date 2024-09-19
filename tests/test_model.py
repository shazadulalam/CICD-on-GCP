import numpy as np
from ..model.train import create_model, load_data

def test_model_prediction():
    # Load the data
    (_, _), (test_images, test_labels) = load_data()

    # Reshape test images to add a channel dimension (required by the model)
    test_images = np.expand_dims(test_images, axis=-1)

    # Create the model
    model = create_model()

    # Use a smaller batch for testing
    small_batch_images = test_images[:10]
    small_batch_labels = test_labels[:10]

    # Make predictions
    predictions = model.predict(small_batch_images)

    # Check if predictions have the correct shape
    assert predictions.shape == (10, 10)  # 10 images, 10 possible classes

    # Check if predictions are probabilities (summing close to 1 for each prediction)
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
