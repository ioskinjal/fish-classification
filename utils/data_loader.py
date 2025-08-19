# utils/data_loader.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir='data', img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255)

    train_data = datagen.flow_from_directory(
        directory=f"{base_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = datagen.flow_from_directory(
        directory=f"{base_dir}/val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        directory=f"{base_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data