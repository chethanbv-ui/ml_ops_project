import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prepare_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/dataset.csv'):
        np.random.seed(42)
        size = np.random.rand(100, 1) * 100
        price = size * 3000 + np.random.randn(100, 1) * 10000
        df = pd.DataFrame({'Size': size.flatten(), 'Price': price.flatten()})
        df.to_csv('data/dataset.csv', index=False)

def train_model():
    df = pd.read_csv('data/dataset.csv')
    x = df[['Size']]
    y = df['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('house_price_prediction')
        
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        
        mlflow.log_metric('r2_score', score)
        mlflow.sklearn.log_model(model, 'model')
        
        print(f'Model R2 Score: {score}')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return score

if __name__ == '__main__':
    prepare_data()
    train_model()
