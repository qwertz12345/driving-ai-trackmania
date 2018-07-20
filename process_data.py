import re
from multiprocessing import Pool
from pathlib import Path
from time import time
from PIL import Image
from random import choices

from PIL.Image import LANCZOS


def load_image(image_path):
    """
    :param image_path: string
    :return: Image object, keys pressed (eg "WA"])
    """
    img = Image.open(image_path)
    # img.load()
    return img, re.sub(r"['\[\], ]", "", image_path.name.split(sep="_")[1].split(sep=".")[0])


def load_images_idx(run_name):
    run_path = Path(r"E:\Trackmania Data\data") / Path(run_name) / Path("screenshots")
    image_paths = list(run_path.glob("*.png"))
    images = [load_image(path) for path in image_paths]
    print("Images loaded.")
    return images


def resize_and_bw(image):
    """
    :param image: tuple (image object, keys)
    :return: also convert to grayscale ("bw")
    """
    edited = image[0].convert("L").resize((32, 32), resample=LANCZOS)     # width, height
    return edited, image[1]


def resizing_bw(images):
    pool = Pool()
    resized = pool.map(resize_and_bw, images)
    pool.close()
    pool.join()
    print("Images resized.")
    return resized


# def velo_digits(image):
#     return (image.crop(box=(740, 401, 17 + 740, 29 + 401)).resize((18, 32)),
#             image.crop(box=(758, 401, 17 + 758, 29 + 401)).resize((18, 32)),
#             image.crop(box=(776, 401, 17 + 776, 29 + 401)).resize((18, 32)))
#
#
# def cropped_digits(images):
#     pool = Pool()
#     cropped = pool.map(velo_digits, images)
#     pool.close()
#     pool.join()
#     return cropped


def flip_image(im):
    flipped_keys = "".join(['A' if e == 'D'
                            else 'D' if e == 'A'
                            else e for e in im[1] if e.isalpha()])
    return im[0].transpose(Image.FLIP_LEFT_RIGHT), flipped_keys


def flip_images_and_key_presses(images):
    pool = Pool()
    flipped = pool.map(flip_image, images)
    pool.close()
    pool.join()
    print("Images flipped.")
    return flipped


def create_training_validation_sets(images, ratio):
    import random
    random.seed()
    random.shuffle(images)
    training_data = images[:int(ratio * len(images))]
    validation_data = images[int(ratio * len(images)):]
    return training_data, validation_data


def save_image(image, path):
    image[0].save(path)


def save_images(train, val, image_index, output_dir):
    train_dir_path = Path(r"E:\Trackmania Data") / output_dir / "training_data"
    val_dir_path = Path(r"E:\Trackmania Data") / output_dir / "validation_data"
    train_dir_path.mkdir(exist_ok=True)
    val_dir_path.mkdir(exist_ok=True)

    keys = set(e[1] for e in (train + val) if "T" not in e[1])
    print("Keys:", keys)
    assert len(keys) < 16
    for u in keys:
        if u == "":
            (train_dir_path / Path("no_key")).mkdir(exist_ok=True)
            (val_dir_path / Path("no_key")).mkdir(exist_ok=True)
        else:
            (train_dir_path / Path(u)).mkdir(exist_ok=True)
            (val_dir_path / Path(u)).mkdir(exist_ok=True)

    ims_to_save = []
    for k, im in enumerate(train):
        if im[1] == "":
            ims_to_save.append((im, train_dir_path / Path("no_key") / Path("{:5d}.png".format(k + image_index))))
        elif "T" in im[1]:
            pass
        else:
            ims_to_save.append((im, train_dir_path / Path(im[1]) / Path("{:5d}.png".format(k + image_index))))
    for k, im in enumerate(val):
        if im[1] == "":
            ims_to_save.append((im, val_dir_path / Path("no_key") / Path("{:5d}.png".format(k + image_index))))
        elif "T" in im[1]:
            pass
        else:
            ims_to_save.append((im, val_dir_path / Path(im[1]) / Path("{:5d}.png".format(k + image_index))))

    pool = Pool(processes=2)
    pool.starmap(save_image, ims_to_save)
    pool.close()
    pool.join()
    # for e in ims_to_save:
    #     save_image(e[0], e[1])
    print("Images saved.")


def save_test_images(test_images):
    test_dir_path = Path(r"E:\Trackmania Data\test_data")
    test_dir_path.mkdir()  # TODO: use try except to give a chance to delete folders with old content

    class_dirs = tuple(
        class_dir.name for class_dir in
        (Path(r"E:\Trackmania Data\training_val_data") / Path("training_data")).iterdir()
        if class_dir.is_dir()
    )
    for d in class_dirs:
        (test_dir_path / Path(d)).mkdir()
    ims_to_save = []
    for k, im in enumerate(test_images):
        if im[1] == "":
            ims_to_save.append((im, test_dir_path / Path("no_key") / Path("{:5d}.png".format(k))))
        elif im[1] in class_dirs:
            ims_to_save.append((im, test_dir_path / Path(im[1]) / Path("{:5d}.png".format(k))))
    pool = Pool(processes=2)
    pool.starmap(save_image, ims_to_save)
    pool.close()
    pool.join()
    # for e in ims_to_save:
    #     save_image(e[0], e[1])
    print("Images saved.")


def train_val_data(output_dir):
    # run_names = ["a01_20180613-210459", "a01_20180630-145845", "a01_20180702-163212", "a01_20180702-162702",
    #              "a01_20180703-191614", "a01_20180706-180526", "a01_20180706-182829", "a01_20180706-183011",
    #              "a01_several_runs_20180706-181921", "a01_20180706-181307", "custom_20180708-160902",
    #              "custom_20180708-161212", "custom_several_20180708-161618"]
    data_dir = Path(r"E:\Trackmania Data\data")
    run_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    image_paths = []
    for run_dir in run_dirs:
        run_path = data_dir / Path(run_dir) / Path("screenshots")
        for p in run_path.glob("*.png"):
            image_paths.append(p)

    # track_names = []
    # for p in image_paths:
    #     track_names.append(p.parts[3].split("_")[0])
    # track_names = set(track_names)

    # image_paths = choices(image_paths, k=100)

    chunk_size = 3000
    print("Number of non-test images:", len(image_paths))
    image_paths_chunks = [image_paths[x:x + chunk_size] for x in range(0, len(image_paths), chunk_size)]
    image_index = 0
    train_val_split_ratio = 0.6
    for chunk in image_paths_chunks:
        ims = [load_image(path) for path in chunk]
        ims = resizing_bw(ims)
        train, val = create_training_validation_sets(ims, train_val_split_ratio)
        train += flip_images_and_key_presses(train)
        save_images(train, val, image_index, output_dir)
        image_index += len(chunk)
        print("Progress:", image_index / len(image_paths))


def test_data():
    start_time = time()
    test_imgs = load_images_idx("a15_20180713-150302")
    cr = resizing_bw(test_imgs)
    save_test_images(cr)
    print("Duration:", time() - start_time)


if __name__ == "__main__":
    train_val_data("training_val_data")
    # test_data()
