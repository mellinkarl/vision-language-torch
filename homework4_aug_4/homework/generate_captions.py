from pathlib import Path

import fire
from matplotlib import pyplot as plt
import json

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    info_path = Path(info_path)
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not kart_objects:
        return []

    ego_kart = None
    for obj in kart_objects:
        if obj["is_center_kart"] == True:
            center_kart = obj["kart_name"]
            ego_kart = obj
            break
    if ego_kart is None:
        return []

    captions = []
    base_name = info_path.stem.replace("_info", "")
    image_file = f"{info_path.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    kart_name = center_kart
    # 1. Ego car
    # {kart_name} is the ego car.
    captions.append({
        "caption": f"{kart_name} is the ego car.",
        "image_file": image_file
    })


    num_karts = str(len(kart_objects))
    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append({
        "caption": f"There are {num_karts} karts in the scene.",
        "image_file": image_file
    })

    track_name = extract_track_info(info_path)
    # 3. Track name
    # The track is {track_name}.
    captions.append({
        "caption": f"The track is {track_name}.",
        "image_file": image_file
    })

    kart_names = []
    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    for obj in kart_objects:
        kart_names.append(obj["kart_name"])
        if not obj["is_center_kart"]:
            kart_name = obj["kart_name"]

            if obj["center"][0] < ego_kart["center"][0]:
                left_right = "left"
            else:
                left_right = "right"
            
            if obj["center"][1] < ego_kart["center"][1]:
                front_back = "in front of"
            else:
                front_back = "behind"
            
            captions.extend([{
                "caption": f"{kart_name} is {left_right} of the ego car.",
                "image_file": image_file
            }, {
                "caption": f"{kart_name} is {front_back} the ego car.",
                "image_file": image_file
            }])
    
    kart_names_str = ", ".join(kart_names)
    captions.append({
        "caption": f"The karts in the scene are {kart_names_str}.",
        "image_file": image_file
    })

    return captions

def generate_all():
    captions = []
    data_dir = Path('/content/vision-language-torch/homework4_aug_4/data/train')
    for file_path in data_dir.glob('*_info.json'):
        for view_index in range(10):
            captions.extend(generate_caption(file_path, view_index))
    
    with open(data_dir / 'output_captions.json', 'w') as f:
        json.dump(captions, f)


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()
