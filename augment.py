from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the split dataset
train_dir = r"D:\ALL PROJECTS\medicinal-plant-detection\dataset_split\train"
val_dir = r"D:\ALL PROJECTS\medicinal-plant-detection\dataset_split\val"
test_dir = r"D:\ALL PROJECTS\medicinal-plant-detection\dataset_split\test"

# Image size and batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7,1.3]
)

# Validation & Test: only rescale
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
print("✅ Data generators created.")