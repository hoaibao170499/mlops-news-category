import mlflow.sklearn
import pandas as pd

model_name = "news_classifier"
model_version = "1"
alias = "the_best"

model_uri = f"models:/{model_name}/{model_version}"
# model_uri = f"models:/{model_name}@{alias}"

model = mlflow.sklearn.load_model(model_uri)


def create_sample_data():
    """
    Create 2 sample data points for prediction

    Returns:
        DataFrame with sample data
    """
    # Sample data points with realistic values for housing features
    post_1 = "Hey friend, let's have some beer tonight. sound good?"
    post_2 = "We offer you a free coupon to buy this machine with just 200 dollar in cash"

    sample_data = {
        'posts': [post_1, post_2]
    }

    df = pd.DataFrame(sample_data)
    return df['posts']

ID_TO_LABEL = {
    0: 'atheism',
    1: 'religion.misc',
    2: 'graphics',
    3: 'space',
}

predictions = model.predict(create_sample_data())
predicted_labels = [ID_TO_LABEL.get(int(p), str(p)) for p in predictions]

print(f"Predicted numeric categories: {predictions}")
print(f"Predicted text labels: {predicted_labels}")