"""
Authors : inzapp

Github url : https://github.com/inzapp/light-pose

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import random
from model import Model
from glob import glob
from time import time
from enum import Enum, auto

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tqdm import tqdm
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ale import AbsoluteLogarithmicError
from keras_flops import get_flops

import matplotlib.pyplot as plt


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

class LightPose:
    def __init__(self,
                 mode,
                 train_image_path,
                 input_shape,
                 lr,
                 warm_up,
                 momentum,
                 batch_size,
                 iterations,
                 decay=5e-4,
                 training_view=False,
                 checkpoint_interval=2000,
                 pretrained_model_path='',
                 validation_image_path='',
                 output_tensor_dimension=2,
                 confidence_threshold=0.05,
                 validation_split=0.2):
        self.mode = mode
        self.limb = dict_limb[self.mode]
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.lr = lr
        self.warm_up = warm_up
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.training_view_flag = training_view
        self.checkpoint_interval = checkpoint_interval
        self.output_tensor_dimension = output_tensor_dimension
        self.confidence_threshold = confidence_threshold
        self.img_type = cv2.IMREAD_COLOR
        self.live_view_time = time()
        assert self.output_tensor_dimension in [1, 2]
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        self.limb_size = len(self.limb)
        if self.output_tensor_dimension == 1:
            self.output_size = self.limb_size * 3
        elif self.output_tensor_dimension == 2:
            self.output_size = self.limb_size + 3
        if pretrained_model_path == '':
            self.model = Model(input_shape=self.input_shape, output_size=self.output_size, decay=decay).build(output_tensor_dimension=self.output_tensor_dimension)
            self.model.save('model.h5', include_optimizer=False)
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
            self.input_shape = self.model.input_shape[1:]
        self.output_shape = self.model.output_shape[1:]

        self.train_image_paths = list()
        self.validation_image_paths = list()
        if self.validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(self.train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(self.validation_image_path)
        elif self.validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(self.train_image_path, self.validation_split)

        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            output_tensor_dimension=self.output_tensor_dimension,
            batch_size=self.batch_size,
            limb_size=self.limb_size)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            output_tensor_dimension=self.output_tensor_dimension,
            batch_size=self.batch_size,
            limb_size=self.limb_size)
        self.lr_scheduler = LRScheduler(iterations=self.iterations, lr=self.lr, warm_up=self.warm_up, policy='step')

        self.loss_history = {
            'total_loss': [],
            'confidence_loss': [],
            'regression_loss': []
        }
        if self.output_tensor_dimension == 2:
            self.loss_history['classification_loss'] = []

    @staticmethod
    def init_image_paths(image_path, validation_split=0.0):
        all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        random.shuffle(all_image_paths)
        num_cur_class_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_cur_class_train_images]
        validation_image_paths = all_image_paths[num_cur_class_train_images:]
        return image_paths, validation_image_paths

    def evaluate(self, model, generator_flow, loss_fn):
        loss_sum = 0.0
        for batch_x, batch_y in tqdm(generator_flow):
            y_pred = model(batch_x, training=False)
            loss_sum += tf.reduce_mean(np.square(batch_y - y_pred))
        return loss_sum / tf.cast(len(generator_flow), dtype=tf.float32) 

    def calculate_pck(self, dataset='validation', distance_threshold=0.1, mode="Human"):  # PCK : percentage of correct keypoints, the metric of keypoints detection model
        assert dataset in ['train', 'validation']
        visible_keypoint_count = 0
        correct_keypoint_count = 0
        invisible_keypoint_count = 0
        head_neck_distance_count = 0
        head_neck_distance_sum = 0.0
        image_paths = self.train_image_paths if dataset == 'train' else self.validation_image_paths
        for image_path in tqdm(image_paths):
            img, path = DataGenerator.load_img(image_path, self.input_shape[-1] == 3)
            img = DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32') / 255.0
            y_pred = np.asarray(self.graph_forward(self.model, x)).reshape(self.output_shape).astype('float32')
            y_pred = self.post_process(y_pred)
            
            # read ground truth file.
            with open(f'{path[:-4]}.txt', 'rt') as f:
                lines = f.readlines()
            y_true = np.zeros(shape=(self.limb_size, 3), dtype=np.float32)
            for i, line in enumerate(lines):
                if i == self.limb_size:
                    break
                confidence, x_pos, y_pos = list(map(float, line.split()))
                y_true[i][0] = confidence
                y_true[i][1] = x_pos
                y_true[i][2] = y_pos

            if mode == "Human":
                # if Head, Neck is Visible,
                if y_true[0][0] == 1.0 and y_true[1][0] == 1.0:
                    x_pos_head = y_true[0][1]
                    y_pos_head = y_true[0][2]
                    x_pos_neck = y_true[1][1]
                    y_pos_neck = y_true[1][2]
                    distance = np.sqrt(np.square(x_pos_head - x_pos_neck) + np.square(y_pos_head - y_pos_neck))
                    head_neck_distance_sum += distance
                    head_neck_distance_count += 1


                for i in range(self.limb_size):
                    if y_true[i][0] == 1.0:
                        visible_keypoint_count += 1
                        if y_pred[i][0] > self.confidence_threshold:
                            x_pos_true = y_true[i][1]
                            y_pos_true = y_true[i][2]
                            x_pos_pred = y_pred[i][1]
                            y_pos_pred = y_pred[i][2]
                            distance = np.sqrt(np.square(x_pos_true - x_pos_pred) + np.square(y_pos_true - y_pos_pred))
                            if distance < distance_threshold:
                                correct_keypoint_count += 1
                    else:
                        invisible_keypoint_count += 1
            # PCKb
            else: # PCKb
                diag = np.sqrt(2)
                dists = np.linalg.norm( (y_pred - y_true)[:, 1:], axis=1)
                for i in range(self.limb_size):
                    if y_true[i][0] == 1.0:
                        visible_keypoint_count+=1
                        if y_pred[i][0] > self.confidence_threshold:
                            dist = dists[i]
                            if dist < distance_threshold:
                                correct_keypoint_count+=1
                    else:
                        invisible_keypoint_count+=1
                pass
        pck = correct_keypoint_count / float(visible_keypoint_count)

        print(f'visible_keypoint_count   : {visible_keypoint_count}')
        print(f'invisible_keypoint_count : {invisible_keypoint_count}')
        print(f'correct_keypoint_count   : {correct_keypoint_count}')
        if mode=="Human":
            head_neck_distance = head_neck_distance_sum / float(head_neck_distance_count)
            print(f'head neck distance : {head_neck_distance:.4f}')
        print(f'{dataset} data PCK@{int(distance_threshold * 100)} : {pck:.4f}')
        return pck
    
    def calculate_oks(self, dataset='validation', distance_threshold=0.001, mode="Human"):  # PCK : percentage of correct keypoints, the metric of keypoints detection model
        assert dataset in ['train', 'validation']
        # visible_keypoint_count = 0
        # correct_keypoint_count = 0
        # invisible_keypoint_count = 0
        # head_neck_distance_count = 0
        # head_neck_distance_sum = 0.0
        average_oks = 0.
        oks_count = 0.
        image_paths = self.train_image_paths if dataset == 'train' else self.validation_image_paths
        for image_path in tqdm(image_paths):
            img, path = DataGenerator.load_img(image_path, self.input_shape[-1] == 3)
            resized_img = DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(resized_img).reshape((1,) + self.input_shape).astype('float32') / 255.0
            y_pred = np.asarray(self.graph_forward(self.model, x)).reshape(self.output_shape).astype('float32')
            y_pred = self.post_process(y_pred)
            
            # read ground truth file.
            with open(f'{path[:-4]}.txt', 'rt') as f:
                lines = f.readlines()
            y_true = np.zeros(shape=(self.limb_size, 3), dtype=np.float32)
            for i, line in enumerate(lines):
                if i == self.limb_size:
                    break
                confidence, x_pos, y_pos = list(map(float, line.split()))
                y_true[i][0] = confidence
                y_true[i][1] = x_pos
                y_true[i][2] = y_pos

            gt_visibility = y_true[:, 0]

            area = img.shape[0] * img.shape[1]

            diff = (y_pred - y_true)[:, 1:]
            diff2 = np.sum(diff **2, axis=1)

            # sigma = np.array(0.25)
            sigma = np.array(distance_threshold)
            denom = 2 * (area * (sigma ** 2)) + 1e-7

            oks_per_keypoint = np.exp(-diff2 / denom)

            valid = gt_visibility > 0

            average_oks = ((average_oks * oks_count) + np.sum(oks_per_keypoint[valid])) / (oks_count + np.sum(valid))
            oks_count += np.sum(valid)

        return average_oks

    def compute_gradient_1d(self, model, optimizer, loss_function, x, y_true, limb_size):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.dtypes.int32)
            y_true = tf.reshape(y_true, (batch_size, limb_size, 3))
            y_pred = tf.reshape(y_pred, (batch_size, limb_size, 3))
            batch_size_f = tf.cast(batch_size, dtype=y_pred.dtype)
            confidence_loss = tf.reduce_sum(loss_function(y_true[:, :, 0], y_pred[:, :, 0])) / batch_size_f
            regression_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, 1:] - y_pred[:, :, 1:]), axis=-1) * y_true[:, :, 0]) / batch_size_f
            loss = confidence_loss + regression_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, confidence_loss, regression_loss

    def compute_gradient_2d(self, model, optimizer, loss_function, x, y_true, limb_size):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            batch_size_f = tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)
            confidence_true = y_true[:, :, :, 0]
            confidence_pred = y_pred[:, :, :, 0]
            confidence_loss = tf.reduce_sum(loss_function(confidence_true, confidence_pred)) / batch_size_f

            cx_true = y_true[:, :, :, 1]
            cy_true = y_true[:, :, :, 2]

            y_true_shape = tf.cast(tf.shape(y_true), y_pred.dtype)
            grid_height, grid_width = y_true_shape[1], y_true_shape[2]

            x_range = tf.range(grid_width, dtype=y_pred.dtype)
            x_offset = tf.broadcast_to(x_range, shape=tf.shape(cx_true))

            y_range = tf.range(grid_height, dtype=y_pred.dtype)
            y_range = tf.reshape(y_range, shape=(1, grid_height, 1))
            y_offset = tf.broadcast_to(y_range, shape=tf.shape(cy_true))

            cx_true = x_offset + (cx_true * 1.0 / grid_width)
            cy_true = y_offset + (cy_true * 1.0 / grid_height)

            cx_pred = y_pred[:, :, :, 1]
            cy_pred = y_pred[:, :, :, 2]

            cx_pred = x_offset + (cx_pred * 1.0 / grid_width)
            cy_pred = y_offset + (cy_pred * 1.0 / grid_height)

            cx_loss = tf.square(cx_true - cx_pred)
            cy_loss = tf.square(cy_true - cy_pred)
            regression_loss = tf.reduce_sum((cx_loss + cy_loss) * confidence_true) / batch_size_f

            class_true = y_true[:, :, :, 3:]
            class_pred = y_pred[:, :, :, 3:]
            classification_loss = tf.reduce_sum(tf.reduce_sum(loss_function(class_true, class_pred), axis=-1) * confidence_true) / batch_size_f
            loss = confidence_loss + classification_loss + regression_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, confidence_loss, regression_loss, classification_loss

    def build_loss_str(self, iteration_count, losses):
        classification_loss = -1.0
        if self.output_tensor_dimension == 2:
            total_loss, confidence_loss, regression_loss, classification_loss = losses
        else:
            total_loss, confidence_loss, regression_loss = losses
        loss_str = f'[iteration_count : {iteration_count:6d}] total_loss => {total_loss:>8.4f}'
        loss_str += f', confidence_loss : {confidence_loss:>8.4f}'
        loss_str += f', regression_loss : {regression_loss:>8.4f}'
        if classification_loss != -1.0:
            loss_str += f', classification_loss : {classification_loss:>8.4f}'
        return loss_str

    def fit(self):
        gflops = get_flops(self.model, batch_size=1) * 1e-9
        self.model.summary()
        print(f'\nGFLOPs : {gflops:.4f}')
        print(f'train on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')
        iteration_count = 0
        max_val_pck = 0.0
        os.makedirs('checkpoint', exist_ok=True)
        loss_function = AbsoluteLogarithmicError(alpha=0.75, gamma=2.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        constant_limb_size = tf.constant(self.limb_size)
        if self.output_tensor_dimension == 1:
            compute_gradient = tf.function(self.compute_gradient_1d)
        elif self.output_tensor_dimension == 2:
            compute_gradient = tf.function(self.compute_gradient_2d)
        for x, y_true in self.train_data_generator.flow():
            self.lr_scheduler.update(optimizer, iteration_count)
            losses = compute_gradient(self.model, optimizer, loss_function, x, y_true, constant_limb_size)
            iteration_count += 1
            if self.training_view_flag:
                self.training_view_function()
            print(self.build_loss_str(iteration_count, losses))

            if self.output_tensor_dimension == 2:
                total_loss, confidence_loss, regression_loss, classification_loss = losses
                self.loss_history['total_loss'].append(float(total_loss))
                self.loss_history['confidence_loss'].append(float(confidence_loss))
                self.loss_history['regression_loss'].append(float(regression_loss))
                self.loss_history['classification_loss'].append(float(classification_loss))
            else:
                total_loss, confidence_loss, regression_loss = losses
                self.loss_history['total_loss'].append(float(total_loss))
                self.loss_history['confidence_loss'].append(float(confidence_loss))
                self.loss_history['regression_loss'].append(float(regression_loss))

            warm_up_end = iteration_count >= int(self.iterations * self.warm_up)
            if warm_up_end and iteration_count % self.checkpoint_interval == 0:
                print()
                valid_func="PCKb"
                if valid_func == "PCKh":
                    val_pck = self.calculate_pck(dataset='validation')
                elif valid_func == "PCKb": 
                    val_pck = self.calculate_pck(dataset='validation', distance_threshold=0.01, mode=self.mode)
                else:
                    val_oks = self.calculate_oks()

                if "PCK" in valid_func:
                    if val_pck > max_val_pck:
                        max_val_pck = val_pck
                        self.model.save(f'checkpoint/best_model_{iteration_count}_iter_{val_pck:.4f}_val_PCK.h5', include_optimizer=False)
                        print('best val PCK model saved')
                    else:
                        self.model.save(f'checkpoint/model_{iteration_count}_iter_{val_pck:.4f}_val_PCK.h5', include_optimizer=False)
                else:
                    if val_oks > max_val_pck:
                        max_val_pck = val_oks
                        self.model.save(f'checkpoint/best_model_{iteration_count}_iter_{val_oks:.4f}_val_OKS.h5', include_optimizer=False)
                        print('best val OKS model saved')
                    else:
                        self.model.save(f'checkpoint/model_{iteration_count}_iter_{val_oks:.4f}_val_OKS.h5', include_optimizer=False)
                print()

            if iteration_count == self.iterations:
                print('\ntrain end successfully')

                self.plot_loss()
                return

    def predict_images(self, dataset='validation'):
        assert dataset in ['train', 'validation']
        if dataset == 'train':
            image_paths = self.train_image_paths
        elif dataset == 'validation':
            image_paths = self.validation_image_paths
        for img_path in image_paths:
            img = self.predict(DataGenerator.load_img(img_path, color=True)[0])
            cv2.imshow(f'{dataset} images', img)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def predict_video(self, video_path, fps=15, size=(256, 512)):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cnt = 0
        while True:
            is_frame_exist, img = cap.read()
            if not is_frame_exist:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
            img = DataGenerator.resize(img, size)
            img = self.predict(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(int(1 / float(fps) * 1000))
            if key == 27:
                cap.release()
                return
            cnt += 1
            print(f'predict {cnt} frames...')
        cap.release()

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def circle_if_valid(self, img, v):
        if v[0] > self.confidence_threshold:
            x = int(v[1] * img.shape[1])
            y = int(v[2] * img.shape[0])
            img = cv2.circle(img, (x, y), 6, (128, 255, 128), thickness=-1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (x, y), 3, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
        return img

    def line_if_valid(self, img, p1, p2):
        if p1[0] > self.confidence_threshold and p2[0] > self.confidence_threshold:
            x1 = int(p1[1] * img.shape[1])
            y1 = int(p1[2] * img.shape[0])
            x2 = int(p2[1] * img.shape[1])
            y2 = int(p2[2] * img.shape[0])
            img = cv2.line(img, (x1, y1), (x2, y2), (64, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return img

    def draw_skeleton(self, img, y):
        if self.mode == "Human":
            img = self.line_if_valid(img, y[self.limb.HEAD.value], y[self.limb.NECK.value])

            img = self.line_if_valid(img, y[self.limb.NECK.value], y[self.limb.RIGHT_SHOULDER.value])
            img = self.line_if_valid(img, y[self.limb.RIGHT_SHOULDER.value], y[self.limb.RIGHT_ELBOW.value])
            img = self.line_if_valid(img, y[self.limb.RIGHT_ELBOW.value], y[self.limb.RIGHT_WRIST.value])

            img = self.line_if_valid(img, y[self.limb.NECK.value], y[self.limb.LEFT_SHOULDER.value])
            img = self.line_if_valid(img, y[self.limb.LEFT_SHOULDER.value], y[self.limb.LEFT_ELBOW.value])
            img = self.line_if_valid(img, y[self.limb.LEFT_ELBOW.value], y[self.limb.LEFT_WRIST.value])

            img = self.line_if_valid(img, y[self.limb.RIGHT_HIP.value], y[self.limb.LEFT_HIP.value])

            img = self.line_if_valid(img, y[self.limb.RIGHT_SHOULDER.value], y[self.limb.RIGHT_HIP.value])
            img = self.line_if_valid(img, y[self.limb.RIGHT_HIP.value], y[self.limb.RIGHT_KNEE.value])
            img = self.line_if_valid(img, y[self.limb.RIGHT_KNEE.value], y[self.limb.RIGHT_ANKLE.value])

            img = self.line_if_valid(img, y[self.limb.LEFT_SHOULDER.value], y[self.limb.LEFT_HIP.value])
            img = self.line_if_valid(img, y[self.limb.LEFT_HIP.value], y[self.limb.LEFT_KNEE.value])
            img = self.line_if_valid(img, y[self.limb.LEFT_KNEE.value], y[self.limb.LEFT_ANKLE.value])
        
        elif self.mode == "Pod":
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.FLB.value])
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.RLT.value])
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.FRT.value])
            img = self.line_if_valid(img, y[self.limb.FLB.value], y[self.limb.FRB.value])
            img = self.line_if_valid(img, y[self.limb.FLB.value], y[self.limb.RLB.value])
            img = self.line_if_valid(img, y[self.limb.FRB.value], y[self.limb.FRT.value])
            img = self.line_if_valid(img, y[self.limb.FRB.value], y[self.limb.RRB.value])
            img = self.line_if_valid(img, y[self.limb.FRT.value], y[self.limb.RRT.value])
            img = self.line_if_valid(img, y[self.limb.RLT.value], y[self.limb.RLB.value])
            img = self.line_if_valid(img, y[self.limb.RLT.value], y[self.limb.RRT.value])
            img = self.line_if_valid(img, y[self.limb.RLB.value], y[self.limb.RRB.value])
            img = self.line_if_valid(img, y[self.limb.RRB.value], y[self.limb.RRT.value]) 
        
        elif self.mode == "Pallet":
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.FLB.value])
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.RLT.value])
            img = self.line_if_valid(img, y[self.limb.FLT.value], y[self.limb.FRT.value])
            
            img = self.line_if_valid(img, y[self.limb.FLB.value], y[self.limb.FRB.value])
            img = self.line_if_valid(img, y[self.limb.FLB.value], y[self.limb.RLB.value])
            
            img = self.line_if_valid(img, y[self.limb.FRB.value], y[self.limb.FRT.value])
            img = self.line_if_valid(img, y[self.limb.FRB.value], y[self.limb.RRB.value])
            
            img = self.line_if_valid(img, y[self.limb.FRT.value], y[self.limb.RRT.value])
            
            img = self.line_if_valid(img, y[self.limb.RLT.value], y[self.limb.RLB.value])
            img = self.line_if_valid(img, y[self.limb.RLT.value], y[self.limb.RRT.value])
            
            img = self.line_if_valid(img, y[self.limb.RLB.value], y[self.limb.RRB.value])
            
            img = self.line_if_valid(img, y[self.limb.RRB.value], y[self.limb.RRT.value])

            img = self.line_if_valid(img, y[self.limb.HLLT.value], y[self.limb.HLLB.value])
            img = self.line_if_valid(img, y[self.limb.HLLT.value], y[self.limb.HLRT.value])
            img = self.line_if_valid(img, y[self.limb.HLRB.value], y[self.limb.HLLB.value])
            img = self.line_if_valid(img, y[self.limb.HLRB.value], y[self.limb.HLRT.value])

            img = self.line_if_valid(img, y[self.limb.HRLT.value], y[self.limb.HRLB.value])
            img = self.line_if_valid(img, y[self.limb.HRLT.value], y[self.limb.HRRT.value])
            img = self.line_if_valid(img, y[self.limb.HRRB.value], y[self.limb.HRLB.value])
            img = self.line_if_valid(img, y[self.limb.HRRB.value], y[self.limb.HRRT.value])
        
        for v in y:
            img = self.circle_if_valid(img, v)
        return img

    def post_process(self, y):
        target_shape = (self.limb_size, 3)
        if self.output_tensor_dimension == 1:
            return y.reshape(target_shape)
        else:
            rows, cols = self.output_shape[:2]
            res = np.zeros(shape=(self.limb_size, 3), dtype=np.float32)
            for row in range(rows):
                for col in range(cols):
                    for i in range(self.limb_size):
                        confidence = y[row][col][0]
                        if confidence < self.confidence_threshold:
                            continue
                        class_index = 0
                        max_class_score = 0.0
                        for j in range(self.limb_size):
                            class_score = y[row][col][j+3]
                            if class_score > max_class_score:
                                max_class_score = class_score
                                class_index = j
                        confidence *= max_class_score
                        if confidence < self.confidence_threshold:
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

    def predict(self, color_img, view_size=(256, 512)):
        raw = color_img
        if self.input_shape[-1] == 1:
            img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        else:
            img = color_img
        x = np.asarray(DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))).reshape((1,) + self.input_shape).astype('float32') / 255.0
        y = np.asarray(self.graph_forward(self.model, x)[0]).reshape(self.output_shape)
        y = self.post_process(y)
        img = self.draw_skeleton(DataGenerator.resize(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), view_size), y)
        return img

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_time < 0.5:
            return
        self.live_view_time = cur_time
        train_image = self.predict(DataGenerator.load_img(np.random.choice(self.train_image_paths), color=True)[0])
        validation_image = self.predict(DataGenerator.load_img(np.random.choice(self.validation_image_paths), color=True)[0])
        cv2.imshow('train', train_image)
        cv2.imshow('validation', validation_image)
        cv2.waitKey(1)

    def plot_loss(self):
        # x축: iteration
        x_axis = range(1, len(self.loss_history['total_loss']) + 1)

        plt.figure(figsize=(10, 6))

        # (1) total_loss
        plt.plot(x_axis, self.loss_history['total_loss'], label='total_loss', color='r', linewidth=1.5)

        # (2) confidence_loss
        plt.plot(x_axis, self.loss_history['confidence_loss'], label='confidence_loss', color='g', linewidth=1.5)

        # (3) regression_loss
        plt.plot(x_axis, self.loss_history['regression_loss'], label='regression_loss', color='b', linewidth=1.5)

        # (4) classification_loss (output_tensor_dimension == 2 인 경우에만)
        if self.output_tensor_dimension == 2 and 'classification_loss' in self.loss_history:
            plt.plot(x_axis, self.loss_history['classification_loss'], label='classification_loss', color='m', linewidth=1.5)

        # 그래프 설정
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.ylim((0.,1.))
        plt.legend()
        plt.grid(True)

        # 그래프 저장 혹은 표시
        plt.savefig('loss_curve.png')  # 'loss_curve.png' 파일로 저장
        # plt.show()                     # 화면에 표시

