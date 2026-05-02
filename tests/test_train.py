import os
import pytest
from src.train import prepare_data, train_model

def test_prepare_data():
    prepare_data()
    assert os.path.exists('data/dataset.csv')

def test_train_model():
    score = train_model()
    assert score > 0.5
