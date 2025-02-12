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
from glob import glob
from tools.common import dict_limb, set_batch_img, post_process, draw_skeleton
def main(model_path):
    dst_path = "result/export/"
    os.makedirs(dst_path, exist_ok=True)

    folder_name = os.path.splitext(os.path.basename(model_path))[0]
    same_folder_names = list(filter(os.path.isdir, glob(folder_name + "*")))
    same_folder_names_cnt = len(same_folder_names)

    save_dst_path = f'{dst_path}{folder_name}_{same_folder_names_cnt:03d}/'

    basename = os.path.basename(model_path)
    filename = os.path.splitext(basename)[0]

    # Load Keras Model
    model = tf.keras.models.load_model(model_path)
    
    # Check limb_size
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
    
    # save to pb
    tf.saved_model.save(model, f"{save_dst_path}model/{filename}")

    # Save to OPENVINO IR
    ov_model = ov.convert_model(f"{save_dst_path}model/{filename}")
    ov.save_model(ov_model, f"{save_dst_path}model/{filename}/{filename}.xml", compress_to_fp16=True)
    
    # Save to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=12)
    onnx.save(onnx_model, f"{save_dst_path}model/{filename}/{filename}.onnx")
    onnx_session = onnxruntime.InferenceSession(f"{save_dst_path}model/{filename}/{filename}.onnx")
    input_name = onnx_session.get_inputs()[0].name

    if mode == "Pod":
        test_image_path = "tools/test_images/no123_pod_00_03444_rgb.jpg"
    elif mode == "Pallet":
        test_image_path = "tools/test_images/no119_pallet_00_003189.jpg"
    else:
        print("inset Human image path.")
        test_image_path = ""
    
    if not os.path.exists(test_image_path):
        print('Test Image is not exists.')
        return

    # Preprocess Test Image
    color_img, batch_img = set_batch_img(model, test_image_path)
    
    
    # keras forward.
    pred = model(batch_img, training=False)[0]
    result = post_process(pred, limb_size, model.output_shape[1:])
    result_img = draw_skeleton(color_img, result, mode, limb)

    cv2.imwrite(f"{save_dst_path}result_keras.jpg", result_img)

    # OPENVINO FORWARD
    ov_compiled_model = ov.compile_model(ov_model)
    pred_ov = ov_compiled_model(batch_img)[0][0]
    result_ov = post_process(pred, limb_size, model.output_shape[1:])
    result_ov_img = draw_skeleton(color_img, result_ov, mode, limb)
    cv2.imwrite(f'{save_dst_path}result_openvino.jpg', result_ov_img)

    # ONNX FORWARD
    pred_onnx = onnx_session.run(None, {input_name: batch_img})[0][0]
    result_onnx = post_process(pred_onnx, limb_size, model.output_shape[1:])
    result_onnx_img = draw_skeleton(color_img, result_onnx, mode, limb)    
    cv2.imwrite(f"{save_dst_path}result_onnx.jpg", result_onnx_img)

    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Export to OPENVINO IR")
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        # default="checkpoint/pod_no_aug/best_model_18000_iter_0.4121_val_PCK.h5",
        # default="checkpoint/pallet_no_aug/best_model_20000_iter_0.0819_val_PCK.h5",
        help="Tensorflow로 생성된 모델 파일 경로를 입력하세요 (*.h5)"
    )
    args = parser.parse_args()
    main(args.model_path)