import openvino as ov
import tensorflow as tf
import argparse
import os
import tf2onnx
import onnx
import onnxruntime
import cv2
import numpy as np
from enum import Enum, auto


class Pod_Limb(Enum):
    FLT = 0
    FLB = auto()
    FRB = auto()
    FRT = auto()
    RLT = auto()
    RLB = auto()
    RRB = auto()
    RRT = auto()

class Human_Limb(Enum):
    HEAD = 0
    NECK = auto()
    RIGHT_SHOULDER = auto()
    RIGHT_ELBOW = auto()
    RIGHT_WRIST = auto()
    LEFT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_HIP = auto()
    RIGHT_KNEE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HIP = auto()
    LEFT_KNEE = auto()
    LEFT_ANKLE = auto()

class Pallet_Limb(Enum):
    FLT = 0
    FLB = auto()
    FRB = auto()
    FRT = auto()
    RLT = auto()
    RLB = auto()
    RRB = auto()
    RRT = auto()

    HLLT = auto()
    HLLB = auto()
    HLRB = auto()
    HLRT = auto()
    
    HRLT = auto()
    HRLB = auto()
    HRRB = auto()
    HRRT = auto()

dict_limb = {"Human":Human_Limb,
             "Pod":Pod_Limb,
             "Pallet":Pallet_Limb}

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

def post_process(y, limb_size, output_shape, output_tensor_dimension=2, confidence_threshold=0.05):
    target_shape = (limb_size, 3)
    if output_tensor_dimension == 1:
        return y.reshape(target_shape)
    else:
        rows, cols = output_shape[:2]
        res = np.zeros(shape=(limb_size, 3), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                for i in range(limb_size):
                    confidence = y[row][col][0]
                    if confidence < confidence_threshold:
                        continue
                    class_index = 0
                    max_class_score = 0.0
                    for j in range(limb_size):
                        class_score = y[row][col][j+3]
                        if class_score > max_class_score:
                            max_class_score = class_score
                            class_index = j
                    confidence *= max_class_score
                    if confidence < confidence_threshold:
                        continue
                    if confidence > res[class_index][0]:
                        x_pos = y[row][col][1]
                        y_pos = y[row][col][2]
                        x_pos = (col + x_pos) / float(cols)
                        y_pos = (row + y_pos) / float(rows)
                        res[class_index][0] = confidence
                        res[class_index][1] = x_pos
                        res[class_index][2] = y_pos
        return res

def draw_skeleton(img, y, mode, limb):
    if mode == "Human":
        img = line_if_valid(img, y[limb.HEAD.value], y[limb.NECK.value])

        img = line_if_valid(img, y[limb.NECK.value], y[limb.RIGHT_SHOULDER.value])
        img = line_if_valid(img, y[limb.RIGHT_SHOULDER.value], y[limb.RIGHT_ELBOW.value])
        img = line_if_valid(img, y[limb.RIGHT_ELBOW.value], y[limb.RIGHT_WRIST.value])

        img = line_if_valid(img, y[limb.NECK.value], y[limb.LEFT_SHOULDER.value])
        img = line_if_valid(img, y[limb.LEFT_SHOULDER.value], y[limb.LEFT_ELBOW.value])
        img = line_if_valid(img, y[limb.LEFT_ELBOW.value], y[limb.LEFT_WRIST.value])

        img = line_if_valid(img, y[limb.RIGHT_HIP.value], y[limb.LEFT_HIP.value])

        img = line_if_valid(img, y[limb.RIGHT_SHOULDER.value], y[limb.RIGHT_HIP.value])
        img = line_if_valid(img, y[limb.RIGHT_HIP.value], y[limb.RIGHT_KNEE.value])
        img = line_if_valid(img, y[limb.RIGHT_KNEE.value], y[limb.RIGHT_ANKLE.value])

        img = line_if_valid(img, y[limb.LEFT_SHOULDER.value], y[limb.LEFT_HIP.value])
        img = line_if_valid(img, y[limb.LEFT_HIP.value], y[limb.LEFT_KNEE.value])
        img = line_if_valid(img, y[limb.LEFT_KNEE.value], y[limb.LEFT_ANKLE.value])
    
    elif mode == "Pod":
        img = line_if_valid(img, y[limb.FLT.value], y[limb.FLB.value])
        img = line_if_valid(img, y[limb.FLT.value], y[limb.RLT.value])
        img = line_if_valid(img, y[limb.FLT.value], y[limb.FRT.value])
        img = line_if_valid(img, y[limb.FLB.value], y[limb.FRB.value])
        img = line_if_valid(img, y[limb.FLB.value], y[limb.RLB.value])
        img = line_if_valid(img, y[limb.FRB.value], y[limb.FRT.value])
        img = line_if_valid(img, y[limb.FRB.value], y[limb.RRB.value])
        img = line_if_valid(img, y[limb.FRT.value], y[limb.RRT.value])
        img = line_if_valid(img, y[limb.RLT.value], y[limb.RLB.value])
        img = line_if_valid(img, y[limb.RLT.value], y[limb.RRT.value])
        img = line_if_valid(img, y[limb.RLB.value], y[limb.RRB.value])
        img = line_if_valid(img, y[limb.RRB.value], y[limb.RRT.value]) 
    
    elif mode == "Pallet":
        img = line_if_valid(img, y[limb.FLT.value], y[limb.FLB.value])
        img = line_if_valid(img, y[limb.FLT.value], y[limb.RLT.value])
        img = line_if_valid(img, y[limb.FLT.value], y[limb.FRT.value])
        
        img = line_if_valid(img, y[limb.FLB.value], y[limb.FRB.value])
        img = line_if_valid(img, y[limb.FLB.value], y[limb.RLB.value])
        
        img = line_if_valid(img, y[limb.FRB.value], y[limb.FRT.value])
        img = line_if_valid(img, y[limb.FRB.value], y[limb.RRB.value])
        
        img = line_if_valid(img, y[limb.FRT.value], y[limb.RRT.value])
        
        img = line_if_valid(img, y[limb.RLT.value], y[limb.RLB.value])
        img = line_if_valid(img, y[limb.RLT.value], y[limb.RRT.value])
        
        img = line_if_valid(img, y[limb.RLB.value], y[limb.RRB.value])
        
        img = line_if_valid(img, y[limb.RRB.value], y[limb.RRT.value])

        img = line_if_valid(img, y[limb.HLLT.value], y[limb.HLLB.value])
        img = line_if_valid(img, y[limb.HLLT.value], y[limb.HLRT.value])
        img = line_if_valid(img, y[limb.HLRB.value], y[limb.HLLB.value])
        img = line_if_valid(img, y[limb.HLRB.value], y[limb.HLRT.value])

        img = line_if_valid(img, y[limb.HRLT.value], y[limb.HRLB.value])
        img = line_if_valid(img, y[limb.HRLT.value], y[limb.HRRT.value])
        img = line_if_valid(img, y[limb.HRRB.value], y[limb.HRLB.value])
        img = line_if_valid(img, y[limb.HRRB.value], y[limb.HRRT.value])
    
    for v in y:
        img = circle_if_valid(img, v)
    return img

def line_if_valid(img, p1, p2, confidence_threshold=0.05):
    if p1[0] > confidence_threshold and p2[0] > confidence_threshold:
        x1 = int(p1[1] * img.shape[1])
        y1 = int(p1[2] * img.shape[0])
        x2 = int(p2[1] * img.shape[1])
        y2 = int(p2[2] * img.shape[0])
        img = cv2.line(img, (x1, y1), (x2, y2), (64, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return img

def circle_if_valid(img, v, confidence_threshold=0.05):
    if v[0] > confidence_threshold:
        x = int(v[1] * img.shape[1])
        y = int(v[2] * img.shape[0])
        img = cv2.circle(img, (x, y), 6, (128, 255, 128), thickness=-1, lineType=cv2.LINE_AA)
        img = cv2.circle(img, (x, y), 3, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
    return img

def set_batch_img(model, test_img_path):
    color_img = cv2.imread(test_img_path)
    if color_img is None:
        print(f"이미지를 불러올 수 없습니다: {test_img_path}")
        return -1
    rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    if model.input_shape[-1] == 1:
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    else:
        img = rgb_img.copy()
    target_size = model.input_shape[1:3][::-1]
    if target_size[0] > img.shape[1] or target_size[1] > img.shape[0]:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    batch_img = resized_img.reshape(((1,) + model.input_shape[1:])).astype('float32') / 255.
    return color_img, batch_img
    
def main(mode, model_path):

    limb_size = 0
    if mode == "Pod":
        limb_size = 8
    elif mode == "Pallet":
        limb_size = 16
    elif mode == "Human":
        limb_size = 14
    limb = dict_limb[mode]

    basename = os.path.basename(model_path)
    filename = os.path.splitext(basename)[0]

    # Load Keras Model
    model = tf.keras.models.load_model(model_path)
    # save to pb
    tf.saved_model.save(model, f"model/{filename}")

    # Save to OPENVINO IR
    ov_model = ov.convert_model(f"model/{filename}")
    ov.save_model(ov_model, f"model/{filename}/{filename}.xml", compress_to_fp16=True)
    
    # Save to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=12)
    # onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path="model.onnx")
    onnx.save(onnx_model, f"model/{filename}/{filename}.onnx")
    onnx_session = onnxruntime.InferenceSession(f"model/{filename}/{filename}.onnx")
    input_name = onnx_session.get_inputs()[0].name

    # Preprocess Test Image
    color_img, batch_img = set_batch_img(model, "./train_data/pod/train/01667_rgb.jpg")
    
    
    # keras forward.
    pred = model(batch_img, training=False)[0]
    result = post_process(pred, limb_size, model.output_shape[1:])
    result_img = draw_skeleton(color_img, result, mode, limb)

    cv2.imwrite("./result_keras.jpg", result_img)

    # OPENVINO FORWARD
    ov_compiled_model = ov.compile_model(ov_model)
    pred_ov = ov_compiled_model(batch_img)[0][0]
    result_ov = post_process(pred, limb_size, model.output_shape[1:])
    result_ov_img = draw_skeleton(color_img, result_ov, mode, limb)
    cv2.imwrite('./result_openvino.jpg', result_ov_img)

    # ONNX FORWARD
    pred_onnx = onnx_session.run(None, {input_name: batch_img})[0][0]
    result_onnx = post_process(pred_onnx, limb_size, model.output_shape[1:])
    result_onnx_img = draw_skeleton(color_img, result_onnx, mode, limb)    
    cv2.imwrite("./result_onnx.jpg", result_onnx_img)

    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Export to OPENVINO IR")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['Human', 'Pod', 'Pallet'], 
        # required=True, 
        default="Pod",
        # default="Pallet",
        help="실행할 모드를 선택합니다: Human, Pod, Pallet 중 하나"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        # required=True, 
        default="checkpoint/_bests/pod_best_model_10000_iter_0.9878_val_PCK.h5",
        # default="checkpoint/_bests/pallet_best_model_12000_iter_0.7136_val_PCK.h5",
        help="Tensorflow로 생성된 모델 파일 경로를 입력하세요 (*.h5)"
    )
    args = parser.parse_args()

    main(args.mode, args.model_path)