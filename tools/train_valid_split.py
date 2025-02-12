import os
import shutil as sh
from glob import glob
from tqdm import tqdm

target_path = "preprocessed/"
objs_path = list(filter(os.path.isdir, glob(target_path+"*")))
for obj_idx, obj_path in enumerate(tqdm(objs_path)):
    obj_path += "/"
    file_list = glob(obj_path + "merge/*.jpg")

    if len(file_list) > 0:
        os.makedirs(obj_path+"train/", exist_ok=True)
        os.makedirs(obj_path+"validation/", exist_ok=True)

    for i, img_name in enumerate(tqdm(file_list)):
        lbl_name = img_name.replace('.jpg', '.txt')
        if not (os.path.exists(img_name) and os.path.exists(lbl_name)):
            print("No files", img_name)
            continue

        if i % 10:
            sh.copy(img_name, obj_path + "train/")
            sh.copy(lbl_name, obj_path + "train/")
        else:
            sh.copy(img_name, obj_path + "validation/")
            sh.copy(lbl_name, obj_path + "validation/")

print("done.")
