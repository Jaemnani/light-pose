import os
import shutil as sh
from glob import glob

# datas_path = "/Users/jaemoonye/tmp/light-pose/train_data/pose/"
datas_path = "train_data/pallet/"
file_list = glob(datas_path + "*.jpg")

if len(file_list) > 0:
    os.makedirs(datas_path+"train/", exist_ok=True)
    os.makedirs(datas_path+"validation/", exist_ok=True)

for i, img_name in enumerate(file_list):
    lbl_name = img_name.replace('.jpg', '.txt')
    if not (os.path.exists(img_name) and os.path.exists(lbl_name)):
        print("No files", img_name)
        continue

    if i % 10:
        sh.move(img_name, datas_path + "train/")
        sh.move(lbl_name, datas_path + "train/")
        print("move to Train")
    else:
        sh.move(img_name, datas_path + "validation/")
        sh.move(lbl_name, datas_path + "validation/")
        print("move to Valid")

print("done.")
