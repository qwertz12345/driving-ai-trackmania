import re
from collections import Counter
from pathlib import Path
from random import choices

from PIL import Image
from tqdm import tqdm


def main(output_dir, nr_images_per_class = 2000, size=(50, 50)):
    allowed_key_presses = {"A", "AW", "D", "DW", "W"}

    base_dir = Path(r"E:\Trackmania Data")
    data_dir = base_dir / "data"
    output_dir = "{}_{}_size-{}x{}".format(output_dir, nr_images_per_class, size[0], size[1])
    (base_dir / output_dir).mkdir(exist_ok=True)

    run_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    image_paths = []
    for run_dir in run_dirs:
        run_path = data_dir / run_dir / "screenshots"
        for p in run_path.glob("*.png"):
            image_paths.append(p)

    key_presses = []
    classes = set(list(allowed_key_presses) + ["no_key"])
    paths_in_classes = dict(zip(classes, ([] for e in range(len(classes)))))
    for im_path in image_paths:
        keys = re.sub(r"['\[\], ]", "", im_path.name.split(sep="_")[1].split(sep=".")[0])
        if keys not in allowed_key_presses:
            keys = "no_key"
        key_presses.append(keys)
        paths_in_classes[keys].append(im_path)

    class_counter = Counter(key_presses)
    print(class_counter)
    with open(base_dir / output_dir / "class counter.txt", "w") as f:
        f.write(str(class_counter))


    paths_in_classes = {keys: choices(paths, k=min(nr_images_per_class, len(paths))) for keys, paths in paths_in_classes.items()}
    split = 0.6
    train = ("train", {keys: paths[:int(split*len(paths))] for keys, paths in paths_in_classes.items()})
    val = ("val", {keys: paths[int(split*len(paths)):] for keys, paths in paths_in_classes.items()})

    image_number = 0
    for name, im_dict in (train, val):
        print(name, "data")
        for cls in classes:
            class_path = base_dir / output_dir / name / str(cls)
            class_path.mkdir(parents=True)
        for cls in tqdm(classes):
            class_path = base_dir / output_dir / name / str(cls)
            im_paths_one_class = im_dict[cls]
            for im_path in im_paths_one_class:
                im = Image.open(im_path)
                im = im.convert("L")
                im = im.resize(size, resample=Image.LANCZOS)  # width, height
                im.save(class_path / "{}.png".format(image_number))
                if name == "train":
                    im_flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
                    if "A" in cls:
                        im_flipped.save(class_path.parent / cls.replace("A", "D") / "{}_flipped.png".format(image_number))
                    elif "D" in cls:
                        im_flipped.save(class_path.parent / cls.replace("D", "A") / "{}_flipped.png".format(image_number))
                    else:
                        im.save(class_path / "{}_flipped.png".format(image_number))
                image_number += 1


if __name__ == "__main__":
    main("processed_data", nr_images_per_class=10000, size=(50, 50))
