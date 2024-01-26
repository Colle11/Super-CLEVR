"""This module generates yolo v5 labels.

please refer to the official document
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils import data
from torchvision.transforms import ToTensor
from yaml.loader import SafeLoader


class Shape(IntEnum):
    """Shape classification."""

    AEROPLANE = 0
    BICYCLE = auto()
    BUS = auto()
    CAR = auto()
    MOTORBIKE = auto()


class Colour(IntEnum):
    """Colour classification."""

    RED = 0
    GREEN = auto()
    BLUE = auto()
    YELLOW = auto()
    CYAN = auto()
    PURPLE = auto()
    GRAY = auto()
    BROWN = auto()


class Size(IntEnum):
    """Size classification."""

    SMALL = 0
    LARGE = auto()


class Material(IntEnum):
    """Material classification."""

    RUBBER = 0
    METAL = auto()


class Relation(IntEnum):
    """Object Relation."""

    LEFT = 0
    RIGHT = auto()
    BEHIND = auto()
    FRONT = auto()


@dataclass
class SceneObject:
    """Scene object loaded from the scene json file."""

    bbox: dict[str, int]
    shape: str
    color: str
    size: str
    coords_3d: list[float]
    material: str
    rotation: float
    pixel_coords: list[list[float]]
    texture: str
    parts: dict[str, dict[str, str]]


@dataclass
class Scene:
    """Scene loaded from the scene json file."""

    objects: list[SceneObject]
    relations: list[tuple]


def get_attribute_identifier(attribute_type, attribute_name: str, is_return_id=True) -> int | str:
    """Get class identifier/number from the given classification type.

    Args:
        attribute_type: attribute type, such as Shape, Relation
        attribute_name: attribute name from scene file.
        is_return_id: return class identifier
    Returns:
        Union[int, str]: class identifier/number or enum's name.
    """
    for att_type in attribute_type:
        if str.lower(att_type.name) == str.lower(attribute_name):
            return att_type.value if is_return_id else att_type.name
    raise ValueError(f"Scene json has undefined attribute names {attribute_name}")


@dataclass
class SuperClevrDataset(data.Dataset):
    """Super-Clevr dataset."""

    def __init__(
        self,
        data_root_dir: str,
        cache_file_name: str,
        split="train",
        image_size: tuple[int, int] = (640, 480),
        label_file_extension: str = ".txt",
        image_file_extension: str = ".png",
        augment: bool = True,
    ):
        """Load the file names from the cached file.

        The file caches either all training data, all validation data or all test data.

        PyTorch doesn't need to go over inodes repeatedly if we cache the file list in a
        separate file.

        Args:
            data_root_dir: dataset root directory
            cache_file_name: cache file name of all label files
            split: train
            image_size: image size. Defaults to (320, 240) in (W, H)
            label_file_extension: label file extension, .txt by default
            image_file_extension: image file extension, .png by default
            augment: True - data augmentation

        """
        cached_file_list_path = self.retrieve_cache_file_name(data_root_dir, cache_file_name, split)

        with open(cached_file_list_path) as file:
            self.label_file_names = file.readlines()

        self.image_file_names = [
            path.replace("labels", "images").replace(label_file_extension, image_file_extension)
            for path in self.label_file_names
        ]
        self.image_size = image_size
        self.augment = augment

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get data by index.

        Args:
            index: index in the data

        Returns:
            Sample: [image, labels]
        """
        image_path = self.image_file_names[index % self.__len__()].rstrip()
        label_path = self.label_file_names[index % self.__len__()].rstrip()
        image = ToTensor()(Image.open(image_path).convert("RGB"))
        boxes = torch.from_numpy(np.loadtxt(label_path))

        return image, boxes

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.label_file_names)

    @staticmethod
    def retrieve_cache_file_name(data_root_dir: str, cache_file_name: str, split: str) -> Path:
        """Retrieve the cache file name.

        Cache file will be generated under e.g.
        data_root_dir/train_cache_file_list.txt
        data_root_dir/val_cache_file_list.txt
        data_root_dir/test_cache_file_list.txt

        Args:
            data_root_dir: data root directory
            cache_file_name: cache file name
            split: train/test/val

        Returns:
            Path: _description_
        """
        cached_file_list_path = Path(data_root_dir) / str(split + "_" + cache_file_name)
        return cached_file_list_path


@dataclass
class BBox:
    """Bounding box attribute."""

    def __init__(self, x: float, y: float, width: float, height: float):
        """Constructor of bounding box.

        All attributes values range from 0 to 1.

        Args:
            x: box centre coordinate x normalized by image width
            y: box centre coordinate y normalized by image height
            width: box width normalized by image width
            height: box height normalized by image height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def from_shape_sub_to_super_class(scene_object: dict):
    """Write a single label file.

    Args:
        scene_object: scene object dictionary
    """
    shape = scene_object["shape"]

    if shape in ["airliner", "biplane", "jet", "fighter"]:
        scene_object["shape"] = "aeroplane"
    elif shape in ["utility", "tandem", "road", "mountain"]:
        scene_object["shape"] = "bicycle"
    elif shape in ["articulated", "double", "regular", "school"]:
        scene_object["shape"] = "bus"
    elif shape in ["truck", "suv", "minivan", "sedan", "wagon"]:
        scene_object["shape"] = "car"
    elif shape in ["chopper", "scooter", "cruiser", "dirtbike"]:
        scene_object["shape"] = "motorbike"
    else:
        raise NotImplementedError(f"Unknown shape value: {shape}")


def from_json_to_scene(scene_path: str | Path) -> Scene:
    """Load the json file directly to a Scene.

    An example of the objects and relations stored in a scene json file:
    "objects": [{
        "bbox": {
            "x": 60,
            "height": 70,
            "width": 64,
            "y": 77
        },
        "shape": "cube",
        "color": "brown",
        "size": "large",
        "3d_coords": [
            -0.7836728692054749,
            -1.8821995258331299,
            0.699999988079071
        ],
        "material": "rubber"
    },
    {
        "bbox": {
            "x": 120,
            "height": 68,
            "width": 70,
            "y": 200
        },
        "shape": "sphere",
        "color": "gray",
        "size": "large",
        "3d_coords": [...],
        "material": "metal"
    }
    ...
    ],
    "relationships": {
        "behind": [
            [1],
            []
        ],
        "front": [
            [],
            [0]
        ],
        "right": [
            [],
            [0]
        ],
        "left": [
            [1],
            []
        ]
    }

    For example: scene["relationships"]["behind"][0] = [1] means that object 0 is behind of object 1

    And this file will be loaded as a Scene.

    Args:
        scene_path: the file path of the scene file

    Returns:
        Scene: A Scene per scene json file.
    """
    with open(scene_path) as json_file:
        scene_json = json.load(json_file)
        scene_object_list = []
        for scene_object_dict in scene_json["objects"]:
            if isinstance(scene_object_dict, dict):
                scene_object_dict["coords_3d"] = scene_object_dict.pop("3d_coords")
                from_shape_sub_to_super_class(scene_object_dict)
                scene_object = SceneObject(**scene_object_dict)
                scene_object_list.append(scene_object)

        scene_relation_list = []
        for relation_name in scene_json["relationships"]:
            relation_id = get_attribute_identifier(Relation, relation_name, is_return_id=False)
            for object_id1, object_id_list in enumerate(scene_json["relationships"][relation_name]):
                for object_id2 in object_id_list:
                    object1 = scene_object_list[object_id1]
                    object2 = scene_object_list[object_id2]
                    scene_relation_list.append(tuple((relation_id, object1, object2)))

        return Scene(scene_object_list, scene_relation_list)


# if there are too many files, it might not fit in the memory, then we have to load
# the scene files in multiple batches
def load_scenes(scenes_path: Path) -> dict[str, Scene]:
    """Load multiple scenes.

    Args:
        scenes_path: Path to directory where scene files are stored

    Returns:
        Dict[str, Scene]: {scene filename: a Scene}
            for example, {"CLEVR_SLICES_val_000000.json": Scene}
            we use the scene file name to create the label file name.
    """
    scene_dict = {}
    for file in os.listdir(scenes_path):
        filename = os.fsdecode(file)
        scene_dict[filename] = from_json_to_scene(scenes_path / filename)

    return scene_dict


def normalize_bbox(bbox: dict[str, int], image_size: tuple[int, int]) -> BBox:
    """Normalize a bounding box to yolov5 format.

    The values of box coordinates are between 0 and 1, (0,0) is at the top left corner.
    The box width and box height are between 0 and 1, normalized by the image width and height.
    The yolov5 label is in this format: [box_centre_x, box_centre_y, box_width, box_height]

    Args:
        bbox: bounding box attribute from the scene, e.g.
            { "x": 60, "height": 70, "width": 64, "y": 77}
        image_size: image width, image height

    Returns:
        BBox: normalized bounding box
    """
    image_width, image_height = image_size
    centre_x = (bbox["x"] + bbox["width"] / 2) / image_width
    centre_y = (bbox["y"] + bbox["height"] / 2) / image_height

    norm_bbox_width = bbox["width"] / image_width
    norm_bbox_height = bbox["height"] / image_height
    norm_bbox = BBox(centre_x, centre_y, norm_bbox_width, norm_bbox_height)

    return norm_bbox


def write_label_file(
    labels_path: Path,
    scene_object_list: list[SceneObject],
    image_size: tuple[int, int],
    classification_type=Shape,
):
    """Write a single label file.

    Args:
        labels_path: directory path where label files should be generated
            e.g. clevr_tiny/customized_clevr_example/train/labels/CLEVR_SLICES_train_000000.csv
        scene_object_list: scene object list parsed from a single scene file
        image_size: image width, image height
        classification_type: Shape, Colour, Material, Size, etc. If Shape is given, the
            classification value will be retrieved from the Shape type, e.g. Shape.AEROPLANE is 0,
            Shape.BICYCLE is 1.
    """
    # remove the file addition '.json' from the scene filename and add '.csv' file addition
    with open(labels_path, "w", newline="") as new_label_file:
        writer = csv.writer(new_label_file, delimiter=" ")
        for scene_object in scene_object_list:
            norm_box = normalize_bbox(scene_object.bbox, image_size)
            object_class = get_object_class_from_scene_object_attr(classification_type, scene_object)
            writer.writerow([object_class, norm_box.x, norm_box.y, norm_box.width, norm_box.height])
        print(f"Written labels to {labels_path}")


def get_object_class_from_scene_object_attr(classification_type, scene_object: SceneObject) -> int | str:
    """Get the object class from the scene object's attribute.

    Args:
        classification_type: Shape, Colour, Material, Size
        scene_object: scene object

    Returns:
        Union[int, str]: class number
    """
    if issubclass(classification_type, Shape):
        return get_attribute_identifier(Shape, scene_object.shape)
    elif issubclass(classification_type, Colour):
        return get_attribute_identifier(Colour, scene_object.color)
    elif issubclass(classification_type, Material):
        return get_attribute_identifier(Material, scene_object.material)
    elif issubclass(classification_type, Size):
        return get_attribute_identifier(Size, scene_object.size)
    else:
        raise NotImplementedError(f"Unknown scene object attribute: {classification_type}")


def generate_and_cache_label_files(data_config: dict, split: str = "train", classification_type=Shape):
    """Generate bounding box and classification label files.

    Generate the label files and cache all label file paths in a file for train/val/test,
    so we can use the cached list for loading data during training to avoid duplicate
    inode visits in every epoch -> You Only Look Once! :D

    Args:
        data_config: dataset config file, e.g. cfg/data/dataset/super-clevr.yaml
        split: train, val, or test
        classification_type: Shape, Colour, Material, Size, etc. If Shape is given, the
            classification value will be retrieved from the Shape type, e.g. Shape.AEROPLANE is 0,
            Shape.BICYCLE is 1.
    """
    data_root_dir = data_config["data_root_dir"]
    image_size = data_config["image_size"]
    # labels_path = Path(data_root_dir) / "labels" / split
    # scenes_path = Path(data_root_dir) / "scenes" / split
    labels_path = Path(data_root_dir) / "labels"
    scenes_path = Path(data_root_dir) / "scenes"

    # if not scenes_path.exists():
    #     scenes_path = Path(data_root_dir) / split / "scenes"
    #     labels_path = Path(data_root_dir) / split / "labels"

    if not scenes_path.exists():
        raise FileNotFoundError(f"No images file are found in the given path {data_root_dir}")

    cache_file_path = SuperClevrDataset.retrieve_cache_file_name(data_root_dir, data_config["cache_file_name"], split)

    os.makedirs(labels_path, exist_ok=True)
    scene_dict = load_scenes(scenes_path)
    label_file_path_buffer = []
    for scene_filename, scene in scene_dict.items():
        label_filename = scene_filename.split(".", 1)[0] + ".txt"
        each_label_file_path = labels_path / label_filename
        write_label_file(each_label_file_path, scene.objects, image_size, classification_type)
        label_file_path_buffer.append(each_label_file_path)

    with open(cache_file_path, "w", newline="") as cache_file:
        writer = csv.writer(cache_file, delimiter=" ")
        for label_file_name in label_file_path_buffer:
            writer.writerow([label_file_name])
        print(f"Cached all label file names in {cache_file_path}")


# Execute this script locally, e.g. "python generate_yolov5_labels.py"
if __name__ == "__main__":
    print("Generate label files and cache file list...")
    with open("./super-clevr.yaml") as f:
        data_config = yaml.load(f, Loader=SafeLoader)
        generate_and_cache_label_files(data_config, split="train")
        generate_and_cache_label_files(data_config, split="val")
        generate_and_cache_label_files(data_config, split="test")

    print("Test data loader...")
    split = "train"
    with open("./super-clevr.yaml") as f:
        data_config = yaml.load(f, Loader=SafeLoader)
        dataset = SuperClevrDataset(
            data_root_dir=data_config["data_root_dir"],
            cache_file_name=data_config["cache_file_name"],
            split=split,
        )
        data_loader = data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
        )

        # print the first label file for testing, the 0-th index is image
        print(next(iter(data_loader))[1])
