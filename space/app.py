import streamlit as st
import numpy as np
import pickle

# Function to load the model from pickle
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# Define the main function for predicting stock price
def predict_stock_price(open_val, high_val, low_val, close_val, adj_close_val, volume, model):
    try:
        # Prepare the input data as a numpy array
        X = np.array([[open_val, high_val, low_val, close_val, adj_close_val]])

        # Use the loaded model for prediction
        predicted_price = model.predict(X)

        return predicted_price[0]
    except (ValueError, AttributeError) as e:
        st.error(f'Error occurred: {str(e)}')
        return None

# Define the Streamlit app
def main():
    st.title('Stock Prediction')

    # Load your trained model from pickle
    model_path = 'model.pkl'  # Replace with your actual path
    model = load_model(model_path)

    # Input fields for user to enter stock data
    open_val = st.number_input('Open (Range: 0 to 100000)', min_value=0.0, max_value=100000.0, value=0.0)
    high_val = st.number_input('High (Range: 0 to 100000)', min_value=0.0, max_value=100000.0, value=0.0)
    low_val = st.number_input('Low (Range: 0 to 100000)', min_value=0.0, max_value=100000.0, value=0.0)
    close_val = st.number_input('Close* (Range: 0 to 100000)', min_value=0.0, max_value=100000.0, value=0.0)
    adj_close_val = st.number_input('Adj Close** (Range: 0 to 100000)', min_value=0.0, max_value=100000.0, value=0.0)
    volume = st.number_input('Volume (Range: 0 to 1,000,000,000)', min_value=0.0, max_value=1000000000.0, value=0.0)

    # Predict button
    if st.button('Predict'):
        if any([open_val == 0.0, high_val == 0.0, low_val == 0.0, close_val == 0.0, adj_close_val == 0.0, volume == 0.0]):
            st.error('Please enter valid numeric values for all fields.')
        else:
            # Call prediction function
            predicted_price = predict_stock_price(open_val, high_val, low_val, close_val, adj_close_val, volume, model)
            
            if predicted_price is not None:
                st.success(f'The predicted stock price is: {predicted_price:.2f}')
            else:
                st.error('Failed to predict stock price.')

if __name__ == '__main__':
    main()
