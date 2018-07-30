from collections import Counter
from pathlib import Path
from re import findall
from time import strftime

from keras import backend as K
from keras import layers
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import get_custom_objects
from keras.utils import print_summary
from keras_applications.mobilenet_v2 import relu6


def custom_accs(class_indices):
    # https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy#41717938

    def A_class_accuracy(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, class_indices["A"]), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    def AW_class_accuracy(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, class_indices["AW"]), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    # def AW_class_recall(y_true, y_pred):
    #     class_id_true = K.argmax(y_true, axis=-1)
    #     class_id_preds = K.argmax(y_pred, axis=-1)
    #     accuracy_mask = K.cast(K.equal(class_id_true, class_indices["AW"]), "int32")
    #     class_recall_tensor = K.cast(K.equal(class_))

    return {"A_class_accuracy": A_class_accuracy, "AW_class_accuracy": AW_class_accuracy}


# def create_model():
#     img_input = layers.Input(shape=(64, 160, 1))
#     # First convolution extracts 16 filters that are 3x3
#     x = layers.Conv2D(16, (2, 5), strides=(1, 3), activation='relu')(img_input)
#     x = layers.MaxPooling2D(2)(x)
#     # Second convolution extracts 32 filters that are 3x3
#     x = layers.Conv2D(32, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(2)(x)
#     # Third convolution extracts 64 filters that are 3x3
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(2)(x)
#     # Flatten feature map to a 1-dim tensor so we can add fully connected layers
#     x = layers.Flatten()(x)
#     # Create a fully connected layer with ReLU activation and 512 hidden units
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     output = layers.Dense(6, activation='softmax')(x)
#     model = Model(img_input, output)
#     return model


# def create_model():
#     img_input = layers.Input(shape=(32, 32, 1))
#     x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_input)
#     x = layers.Conv2D(32, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(2)(x)
#     # x = layers.Dropout(0.25)(x)
#
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.Conv2D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling2D(2)(x)
#     # x = layers.Dropout(0.25)(x)
#
#     x = layers.Flatten()(x)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     output = layers.Dense(6, activation='softmax')(x)
#     model = Model(img_input, output)
#     return model

# def create_model():
#     base_model = MobileNet(
#         input_shape=(32, 32, 1),
#         include_top=False,
#         weights=None
#     )
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = layers.Dense(1024, activation='relu')(x)
#     predictions = layers.Dense(6, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     return model

def create_model(height_width):
    base_model = MobileNetV2(
        input_shape=height_width + (1,),
        include_top=False,
        weights=None,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def generators(training_dir, validation_dir, batch_size, height_width, augment):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            channel_shift_range=2,
            rotation_range=2,
            shear_range=2,
            width_shift_range=2,
            height_shift_range=2
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1 / 255)

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=height_width,  # height, width
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=height_width,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )
    return train_generator, validation_generator


def training_run(
        name,
        base_data_dir,
        height_width,
        epochs,
        augment=False,
        old_model=None,
        batch_size=32,
        data_dir="training_val_data",
        checkpoint_period=1
):
    """
    :param height_width: tuple (height, width)
    :param name: string, overwritten if old_model is specified
    :param base_data_dir: string
    :param epochs:
    :param old_model: string of path of checkpoint
    :param batch_size: overwritten if old_model is specified
    :param data_dir: overwritten if old_model is specified
    :param checkpoint_period:
    :return:
    """

    training_dir = Path(base_data_dir) / data_dir / "train"
    validation_dir = Path(base_data_dir) / data_dir / "val"

    if old_model:
        batch_size = int(findall(r"\d\d", Path(old_model).parent.parent.name.split(".")[1])[0])

    train_generator, validation_generator = generators(training_dir, validation_dir, batch_size, height_width, augment)

    counter = Counter(train_generator.classes)
    print(
        "Counter", "  |   ".join(["{}: {}".format(cls, counter[ind])
                                  for cls, ind in train_generator.class_indices.items()])
    )

    max_val = float(max(counter.values()))
    class_weights = {class_id: min(max_val / num_images, 50) for class_id, num_images in counter.items()}
    print("Class indices:", train_generator.class_indices)
    print(
        "Class weights:", "   |   ".join(
            ["{}: {}".format(cls, class_weights[ind]) for cls, ind in train_generator.class_indices.items()]
        )
    )

    if old_model:
        run_dir = Path(old_model).parent.parent
        full_name = run_dir.name.replace("run_", "")
        get_custom_objects().update(
            {
                "A_class_accuracy": custom_accs(train_generator.class_indices)["A_class_accuracy"],
                "AW_class_accuracy": custom_accs(train_generator.class_indices)["AW_class_accuracy"],
                'relu6': relu6
            }
        )
        model = load_model(old_model)
        initial_epoch = int(Path(old_model).name.split(".")[1].split("-")[0])
    else:  # new start
        initial_epoch = 0
        model = create_model(height_width)
        model.summary()
        full_name = name + "__aug-{}_batch_size{}-start_time{}".format(augment, batch_size, strftime("%Y%m%d-%H%M%S"))
        run_dir = Path(base_data_dir) / "runs" / Path("run_" + full_name)
        run_dir.mkdir()

        with open(run_dir / "model_summary.txt", "w") as f:
            print_summary(model, print_fn=lambda x: f.write(x + "\n"))

        with open(run_dir / "class_indices.txt", "w") as f:
            f.write(str(train_generator.class_indices))

        with open(run_dir / "parameters.txt", "w") as f:
            f.write("name-{}_aug-{}_base_data_dir-{}_height_width-{}_{}_{}_{}_data_dir-{}_{}".format(
                name,
                augment,
                base_data_dir,
                height_width,
                epochs,
                old_model,
                batch_size,
                data_dir,
                checkpoint_period
            ))

        model.compile(
            loss='categorical_crossentropy',
            # optimizer=Adam(),
            optimizer=Adadelta(),
            metrics=[custom_accs(train_generator.class_indices)["A_class_accuracy"],
                     custom_accs(train_generator.class_indices)["AW_class_accuracy"]]
        )

    check_points_dir = run_dir / "checkpoints"
    check_points_dir.mkdir(exist_ok=True)
    check_points = check_points_dir / "checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpointer = ModelCheckpoint(
        filepath=str(check_points),
        verbose=1,
        # save_best_only=True,
        period=checkpoint_period
    )

    tensb_dir = Path(base_data_dir) / "logs" / full_name
    tensboard = TensorBoard(log_dir=str(tensb_dir), batch_size=batch_size, write_graph=True)

    # print(model.evaluate_generator(train_generator))
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        verbose=2,
        class_weight=class_weights,
        callbacks=[checkpointer, tensboard],
        initial_epoch=initial_epoch
    )
    model.save(run_dir / "model_save")

    return history


def main():
    base_data_dir = r"E:\Trackmania Data"
    # hist =
    _ = training_run(
        name="mobilenetV2aug",
        base_data_dir=base_data_dir,
        height_width=(50, 50),
        epochs=111,
        # old_model=r"E:\Trackmania Data\runs\run_mobilenetV2newprocessdata10000.batch_size32-start_time20180728-140902\checkpoints\checkpoint.04-2.22.hdf5",
        # batch_size=32,
        data_dir="processed_data_10000_size-50x50",
        # checkpoint_period=10000000000
        augment=True
    )


if __name__ == "__main__":
    main()
