# path = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_clobot_pod/annotations/person_keypoints_default.json"
# path = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_techno_pal/annotations/person_keypoints_default.json"

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from glob import glob

color_map = [
    (  0,   0, 255),  #  0/19: (R=255, G=0, B=0) -> BGR=(0,0,255)
    (  7,   0, 248),  #  1/19
    ( 13,   0, 242),  #  2/19
    ( 20,   0, 235),  #  3/19
    ( 27,   0, 228),  #  4/19
    ( 34,   0, 222),  #  5/19
    ( 40,   0, 215),  #  6/19
    ( 47,   0, 208),  #  7/19
    ( 54,   0, 202),  #  8/19
    ( 61,   0, 195),  #  9/19
    ( 67,   0, 188),  # 10/19
    ( 74,   0, 181),  # 11/19
    ( 81,   0, 175),  # 12/19
    ( 88,   0, 168),  # 13/19
    ( 94,   0, 161),  # 14/19
    (101,   0, 155),  # 15/19
    (108,   0, 148),  # 16/19
    (115,   0, 141),  # 17/19
    (121,   0, 135),  # 18/19
    (128,   0, 128),  # 19/19: (R=128, G=0, B=128) -> BGR=(128,0,128)
]

def parse_coco_keypoints(ann_file, img_dir, cat_name='person'):
    coco = COCO(ann_file)

    # 2) 원하는 카테고리 ID 가져오기 (예: 사람)
    cat_ids = coco.getCatIds(catNms=[cat_name])
    
    # 3) 해당 카테고리 이미지 ID 가져오기
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    # 4) 이미지 정보 가져오기
    for img_id in img_ids:
        # 4-1) 이미지 메타 데이터 로드
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        height, width = img_info['height'], img_info['width']
        
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            # 이미지 파일이 없으면 스킵
            continue
        
        # 4-2) 해당 이미지의 person(또는 cat_name) 어노테이션 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        annos = coco.loadAnns(ann_ids)

        # 각 객체의 키포인트, bbox 등
        keypoints_list = []
        bbox_list = []
        for ann in annos:
            if 'keypoints' not in ann:
                # keypoints 없는 객체 스킵
                continue
            # 키포인트: [x1, y1, v1, x2, y2, v2, ...] 길이=3* num_keypoints
            kpts = ann['keypoints']
            # COCO에서 visibility: v=0(불확실), v=1(가려짐), v=2(정확히 보임)
            
            keypoints_list.append(kpts)
            bbox_list.append(ann['bbox'])  # [x, y, w, h]
            
        yield {
            'image_id': img_id,
            'file_name': file_name,
            'img_path': img_path,
            'width': width,
            'height': height,
            'keypoints_list': keypoints_list,
            'bbox_list': bbox_list
        }

# 사용 예시
if __name__ == "__main__":

    target_objects = ["pod", "pallet"]
    target_path = "datasets/"
    target_dirs = list(filter(os.path.isdir, glob(target_path +"*")))

    for obj_idx, obj_name in enumerate(target_objects):

        dst_path = "preprocessed/"+obj_name+"/"
        dbg_path = dst_path + "cocokpts_label_dbg/"
        inzapp_path = dst_path + "inzapp/"
        os.makedirs(dbg_path, exist_ok=True)
        os.makedirs(inzapp_path, exist_ok=True)
        
        # img_save_path = dst_path + "images/"
        # kpt_save_path = dst_path + "keypoints/"
        # os.makedirs(img_save_path, exist_ok=True)
        # os.makedirs(kpt_save_path, exist_ok=True)

        for dir_idx, dir_name in enumerate(tqdm(target_dirs)):
            # anno_dir = dir_name + "/annotations"
            anno_path = dir_name + "/annotations/person_keypoints_default.json"
            img_dir = dir_name + "/images/default"

            for data in parse_coco_keypoints(anno_path, img_dir, cat_name = obj_name):
                img_path = data['img_path']
                kpts_list = data['keypoints_list']  # 여러 person 객체가 있을 수 있음
                bbox_list = data['bbox_list']

                print(f"Image ID: {data['image_id']}, path: {img_path}")
                print(f"Number of {obj_name}: {len(kpts_list)}")

                img = cv2.imread(img_path)
                img_size = img.shape[::-1][1:]

                ori_img = img.copy()
                list_length = len(kpts_list)
                print(f"{obj_name}'s Kpts length {list_length}")
                for kpts_idx in range(list_length):
                    bbox = bbox_list[kpts_idx]
                    if bbox[0] < 0:
                        bbox[2] += bbox[0]
                        bbox[0] = 0
                    if bbox[1] < 0:
                        bbox[3] += bbox[1]
                        bbox[1] = 0
                    if bbox[0] + bbox[2] > img_size[0]:
                        bbox[2] = img_size[0] - bbox[0] - 1
                    if bbox[1] + bbox[3] > img_size[1]:
                        bbox[3] = img_size[1]  - bbox[1] - 1


                    bbox_int = np.round(bbox).astype(int)
                    x,y,w,h = bbox_int
                    print("img_size:" , img_size)
                    print(f"b xywh: {x},{y},{w},{h}")
                    wb = int(bbox_int[2] / 10)
                    hb = int(bbox_int[3] / 10)

                    x = bbox_int[0]-wb if bbox_int[0]>=wb else 0
                    y = bbox_int[1]-hb if bbox_int[1]>=hb else 0
                    w = bbox_int[2]+hb if bbox_int[2]<img_size[0]-1 else img_size[0]-1
                    h = bbox_int[3]+wb if bbox_int[3]<img_size[1]-1 else img_size[0]-1
                    print(f"a xywh: {x},{y},{w},{h}")
                    crop_bbox = np.array([x, y, w, h])
                    
                    # x, y, w, h = crop_bbox
                    crop_img = img.copy()[crop_bbox[1] : crop_bbox[1] + crop_bbox[3], crop_bbox[0] : crop_bbox[0] + crop_bbox[2]]
                    crop_ori_img = crop_img.copy()
                    kpts = np.array(kpts_list[kpts_idx]).reshape(-1,3)
                    kpts = kpts - np.array([x, y, 0])

                    inzapp_label = kpts.copy()
                    cimg_size = crop_img.shape[::-1][1:]
                    conf = ((inzapp_label[:,-1] == 2) * 1).reshape(-1,1)
                    kps = inzapp_label[:,:-1]
                    kps[:, 0] /= cimg_size[0]
                    kps[:, 1] /= cimg_size[1]
                    kps = np.clip(kps, 0, 1)
                    inzapp_label = np.hstack((conf, kps))
                    

                    kpts[:,:2][kpts[:,2]==1] = [-1, -1]
                    kpts = kpts[:, :2]
                    kpts = np.round(kpts).astype(int)

                    for kp_idx, kp in enumerate(kpts):
                        if kp[0] < 0 or kp[1] < 0 or kp[0] > img_size[0] or kp[1] > img_size[1]:
                            continue
                        # kp = np.round(kp).astype(int)
                        crop_img = cv2.circle(crop_img, kp, radius=3, color=color_map[kp_idx], thickness=-1)

                    INZAPP=True
                    save_base_path = os.path.basename(dir_name) +"_"+ obj_name +f"_{kpts_idx:02d}_"
                    if INZAPP:
                        save_file_name = save_base_path + os.path.basename(img_path).replace(".jpg", "_DBG.jpg").replace(".png", "_DBG.jpg").replace(".jpeg", "_DBG.jpg")
                        save_img_name = save_base_path + os.path.basename(img_path).replace(".png", ".jpg").replace(".jpeg", ".jpg")
                        save_label_name = save_base_path + os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
                        cv2.imwrite(dbg_path + save_file_name, crop_img)
                        cv2.imwrite(inzapp_path + save_img_name, crop_ori_img)
                        with open(inzapp_path + save_label_name, 'w') as f:
                            for i in inzapp_label:
                                line = "%.1f %.6f %.6f\n"%(i[0], i[1], i[2])
                                f.write(line)

                    # else:
                    #     save_file_name = save_base_path + os.path.basename(img_path).replace(".jpg", "_DBG.jpg").replace(".png", "_DBG.png")
                    #     save_label_name = save_base_path + os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
                    #     cv2.imwrite(dbg_path + save_file_name, crop_img)
                    #     cv2.imwrite(img_save_path + os.path.basename(img_path).replace(".png", ".jpg"), crop_ori_img)
                    #     with open(kpt_save_path + save_label_name, 'w') as f:
                    #         line = ','.join(str(x) for x in kpts.flatten())
                    #         f.write(line)

    # exit()
