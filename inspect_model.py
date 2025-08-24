import tensorflow as tf

MODEL_PATH = "models/plant_classifier.h5"   # adjust if your model is in another folder

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Print model summary
print("\n🟢 Model loaded successfully!\n")
model.summary()

# Print input shape
print("\n🔍 Model input shape:", model.input_shape)
