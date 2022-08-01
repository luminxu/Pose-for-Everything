import os
import shutil
import argparse

from xtcocotools.coco import COCO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--img_src", required=True)
    parser.add_argument("--write_dir", required=True)
    args = parser.parse_args()

    ann_file = args.ann_file
    img_src = args.img_src
    write_dir = args.write_dir

    coco = COCO(ann_file)

    cat_ids = coco.getCatIds()
    for cat_id in cat_ids:
        category_info = coco.loadCats(cat_id)
        cat_name = category_info[0]['name']
        image_path = os.path.join(write_dir, cat_name)
        os.makedirs(image_path, exist_ok=True)

        img_ids = coco.getImgIds(catIds=cat_id)
        for img_id in img_ids:
            img = coco.loadImgs(img_id)[0]
            file_name = img['file_name']
            file_name_new = str(img['id']) + '.jpg'

            src = os.path.join(img_src, file_name)
            dst = os.path.join(image_path, file_name_new)
            shutil.copyfile(src, dst)
