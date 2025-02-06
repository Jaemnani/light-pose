import os
import cv2
import numpy as np
from glob import glob
from light_pose import LightPose
import datetime
import argparse

def test_trained_model(mode,
                       pretrained_model_path: str,
                       test_image_folder: str,
                       save_result_folder: str,
                       input_shape=(96, 96, 1),      # 모델이 학습된 입력 크기에 맞춰서 수정
                       output_tensor_dimension=2,      # 학습 시 사용한 출력 차원(1 or 2)
                       confidence_threshold=0.05):      # 신뢰도 임계값(학습 시 설정)
    """
    폴더 내 이미지를 불러와서 학습된 모델로 추론 후 결과를 시각화합니다.
    ESC(27)을 누르면 종료됩니다.
    """
    # 1) LightPose 인스턴스 생성
    #    - 테스트만 진행할 것이므로 아래 매개변수들은 대부분 의미가 없습니다.
    #    - pretrained_model_path로 학습이 완료된 모델을 불러옵니다.
    pose_model = LightPose(
        mode=mode,
        train_image_path='',              # 의미 없음(필수 인자이므로 빈 문자열)
        input_shape=input_shape,
        lr=1e-4,                          # 의미 없음
        warm_up=0.1,                      # 의미 없음
        momentum=0.9,                     # 의미 없음
        batch_size=1,                     # 의미 없음
        iterations=1,                     # 의미 없음
        decay=5e-4,                       # 의미 없음
        training_view=False,              # 의미 없음
        checkpoint_interval=1000,         # 의미 없음
        pretrained_model_path=pretrained_model_path,
        validation_image_path='',         # 의미 없음
        output_tensor_dimension=output_tensor_dimension,
        confidence_threshold=confidence_threshold,
        validation_split=0.0              # 의미 없음
    )

    # 2) 테스트할 이미지 경로들 불러오기
    test_images = glob(os.path.join(test_image_folder, '**', '*.jpg'), recursive=True)
    if not test_images:
        print(f'해당 폴더에 이미지(.jpg)가 없습니다: {test_image_folder}')
        return

    print(f"테스트할 이미지 개수: {len(test_images)}")

    # 3) 이미지 하나씩 추론 및 시각화
    for img_path in test_images:
        # (1) 이미지 로드
        color_img = cv2.imread(img_path)
        if color_img is None:
            print(f'이미지를 불러올 수 없습니다: {img_path}')
            continue

        # (2) 추론(predict)
        #     - LightPose의 predict 메서드는 (H, W, 3) 형태의 RGB 이미지를 예상하므로
        #       opencv로 불러온 BGR 이미지를 RGB로 변환해서 넣어줍니다.
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        pred_img = pose_model.predict(rgb_img, view_size=(rgb_img.shape[:2][::-1]))  
        # view_size는 결과 확인을 위해 resize하는 크기이므로 적절히 수정 가능합니다.

        cv2.imwrite(save_result_folder + os.path.basename(img_path).replace(".jpg", "_result.jpg"), pred_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightPose 모드 선택")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['Human', 'Pod', 'Pallet'], 
        # required=True, 
        # default="Pod",
        default="Pallet",
        help="실행할 모드를 선택합니다: Human, Pod, Pallet 중 하나"
    )
    args = parser.parse_args()


    # 실제 사용 시 아래 경로를 원하는 값으로 수정하세요.
    pretrained_model_path = 'checkpoint/best_model_12000_iter_0.7136_val_PCK.h5'  # 예: 학습 후 생성된 모델 파일
    test_image_folder = 'train_data/pallet/train/'                   # 예: 테스트할 이미지(.jpg) 폴더 경로
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%dT%H%M")
    save_result_folder = "./result/"+formatted_time+"/"
    os.makedirs(save_result_folder, exist_ok=True)

    # 테스트 실행
    test_trained_model(
        mode=args.mode,
        pretrained_model_path=pretrained_model_path,
        test_image_folder=test_image_folder,
        input_shape=(96, 96, 1),  # 모델 학습 시 사용했던 입력 크기에 맞춰 변경
        output_tensor_dimension=2,  # 학습 시 사용했던 출력 차원(1차원 또는 2차원) 지정
        confidence_threshold=0.05,   # 학습 시 사용했던 신뢰도 임계값에 맞춰 조정
        save_result_folder=save_result_folder
    )
