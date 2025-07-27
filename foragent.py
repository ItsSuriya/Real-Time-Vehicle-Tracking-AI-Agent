import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# ------------------------------
# 1. Load the Trained Model
# ------------------------------
model = tf.keras.models.load_model("final_route_prediction_model.h5")
print("✅ Model Loaded Successfully!")

# ------------------------------
# 2. Load Encoders (Used During Training)
# ------------------------------
csv_path = "balanced_route_prediction_dataset.csv"
df = pd.read_csv(csv_path)

# Load one-hot encoders
categorical_cols = ["Previous Node", "Current Node", "Direction"]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(df[categorical_cols])

target_encoder = OneHotEncoder(sparse_output=False)
target_encoder.fit(df[["Predicted Next Node"]])

scaler = MinMaxScaler()
scaler.fit(df[["Speed (km/h)"]])

# ------------------------------
# 3. Preprocessing Function
# ------------------------------
SEQUENCE_LENGTH = 5

def preprocess_input(prev_node, curr_node, speed, direction):
    time_seconds = 8 * 3600  # Assume testing is at 08:00:00
    speed_normalized = scaler.transform([[speed]])[0][0]

    # One-hot encode categorical features
    cat_input = [[prev_node, curr_node, direction]]
    cat_encoded = encoder.transform(cat_input)

    # Create input vector
    input_vector = np.concatenate([[time_seconds, speed_normalized], cat_encoded[0]])

    # Repeat it SEQUENCE_LENGTH times to match LSTM input shape
    sequence_input = np.tile(input_vector, (SEQUENCE_LENGTH, 1))

    return np.expand_dims(sequence_input, axis=0)  # Shape: (1, SEQUENCE_LENGTH, num_features)

# ------------------------------
# 4. Function to Predict Next Node
# ------------------------------
def predict_next_node(previous_node, current_node, speed, direction):
    try:
        test_input = preprocess_input(previous_node, current_node, speed, direction)
        predicted_probs = model.predict(test_input)
        predicted_class = np.argmax(predicted_probs)
        predicted_next_node = target_encoder.categories_[0][predicted_class]
        
        return predicted_next_node
    
    except Exception as e:
        return f"⚠️ Error: {e}. Please enter valid inputs!"

# Example usage:
predicted_node = predict_next_node(3, 3, 40.5, "West")
if (predicted_node==7):
    print(f"Predicted Next Node: Blind_spot")
else:
    print(f"Predicted Next Node: {predicted_node}")
