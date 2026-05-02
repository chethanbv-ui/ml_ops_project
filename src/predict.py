import pickle
import pandas as pd

def predict(size):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.DataFrame({'Size': [size]})
    prediction = model.predict(df)
    return prediction[0]

if __name__ == '__main__':
    price = predict(150)
    print(f'Predicted Price for size 150: {price}')
