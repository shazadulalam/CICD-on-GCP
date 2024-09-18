from model.train import create_model


def test_model_creation():
    model = create_model()
    assert model is not None
    assert len(model.layers) > 0
