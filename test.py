import os
import cv2
import numpy as np
from glob import glob
from light_pose import LightPose
import datetime
import argparse
from tqdm import tqdm
from tools.common import dict_limb
import tensorflow as tf

def test_trained_model(
                       pretrained_model_path: str,
                       test_image_folder: str,
                       save_result_folder: str,
                       input_shape=(96, 96, 1)      # 모델이 학습된 입력 크기에 맞춰서 수정
                        ):
    model = tf.keras.models.load_model(pretrained_model_path)

    limb_size = model.output_shape[-1] - 3 # x, y, obj_score
    if limb_size == 8:
        mode = "Pod"
    elif limb_size == 14:
        mode = "Human"
    elif limb_size == 16:
        mode = "Pallet"
    else:
        print("unknown model.")
        return -1
    limb = dict_limb[mode]
    del model

    pose_model = LightPose(
        mode=mode,
        train_image_path='',              # 의미 없음(필수 인자이므로 빈 문자열)
        input_shape=input_shape,
        batch_size=1,                     # 의미 없음
        pretrained_model_path=pretrained_model_path,
        validation_split=0.
    )

    test_images = glob(os.path.join(test_image_folder, '**', '*.jpg'), recursive=True)
    if not test_images:
        print(f'해당 폴더에 이미지(.jpg)가 없습니다: {test_image_folder}')
        return

    print(f"테스트할 이미지 개수: {len(test_images)}")

    for img_path in tqdm(test_images):
        color_img = cv2.imread(img_path)
        if color_img is None:
            print(f'이미지를 불러올 수 없습니다: {img_path}')
            continue

        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        pred_img = pose_model.predict(rgb_img, view_size=(rgb_img.shape[:2][::-1]))  

        cv2.imwrite(save_result_folder + os.path.basename(img_path).replace(".jpg", "_result.jpg"), pred_img)

def get_name_from_datetime():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%dT%H%M")
    return formatted_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightPose 모드 선택")
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        # default="checkpoint/pod_no_aug/best_model_18000_iter_0.4121_val_PCK.h5",
        # default="checkpoint/pallet_no_aug/best_model_20000_iter_0.0819_val_PCK.h5",
        help="Tensorflow로 생성된 모델 파일 경로를 입력하세요 (*.h5)"
    )
    parser.add_argument(
        '--test_path',
        type=str,
        required=True,
        # default = "preprocessed/pallet/validation/",
        # default = "preprocessed/pod/validation/",
        help="Test Image 폴더 경로를 입력하세요. default : preprocessed/pallet/validation/"
    )
    args = parser.parse_args()

    target_path = "result/test/"
    save_result_folder = target_path + get_name_from_datetime() + "/"
    os.makedirs(save_result_folder, exist_ok=True)

    # 테스트 실행
    test_trained_model(
        pretrained_model_path=args.model_path,
        test_image_folder=args.test_path,
        input_shape=(96, 96, 1),  # 모델 학습 시 사용했던 입력 크기에 맞춰 변경
        save_result_folder=save_result_folder
    )
