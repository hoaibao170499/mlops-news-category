from unittest.mock import MagicMock, patch

import pytest

@pytest.fixture
def mock_model():
    mock_model = MagicMock()
    # real_model.predict(some_data)
    # def ...
    #   return [2.0]
    mock_model.predict.return_value = ['graphics']
    return mock_model


@pytest.fixture
def mock_pandas():
    mock_pandas = MagicMock()
    mock_pandas.DataFrame.return_value = {"test": ['graphics post',]}
    with patch("scripts.router.predict.pd", mock_pandas):  # type hint
        yield mock_pandas


@pytest.fixture
def mock_mlflow_server(mock_model):
    mock_mlflow_server = MagicMock()
    mock_mlflow_server.sklearn.load_model.return_value = mock_model
    with patch("scripts.router.predict.mlflow", mock_mlflow_server):
        yield mock_mlflow_server


def test_get_model(mock_mlflow_server, mock_model):
    from scripts.router.predict import get_model

    model = get_model()
    assert model == mock_model
    result = model.predict(['graphics post'])
    assert result == ['graphics']


def test_predict(mock_mlflow_server, mock_model, mock_pandas):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scripts.router.predict import housing_router

    app = FastAPI()
    app.include_router(housing_router)
    client = TestClient(app)
    response = client.post(
        "/news/predict",
        json={
            "posts": 'graphics post',
        },
    )
    assert response.status_code == 200
    assert response.json() == {"predicted_posts": 'graphics'}
    mock_pandas.DataFrame.assert_called_once_with(
        {
            "posts": ['graphics post'],
        }
    )