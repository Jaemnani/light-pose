"""
Authors : inzapp

Github url : https://github.com/inzapp/light-pose

Copyright 2022 inzapp Authors. All Rights Reserved.

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
from light_pose import LightPose
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightPose 모드 선택")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['Human', 'Pod', 'Pallet'], 
        # required=True, 
        default="Pod",
        # default="Pallet",
        help="실행할 모드를 선택합니다: Human, Pod, Pallet 중 하나"
    )
    args = parser.parse_args()

    LightPose(
        mode= args.mode,
        train_image_path=r'./train_data/pod/train',
        validation_image_path=r'./train_data/pod/validation',
        input_shape=(96, 96, 1),
        lr=0.001,
        decay=5e-4,
        warm_up=0.5,
        momentum=0.9,
        batch_size=32,
        iterations=20000,
        checkpoint_interval=2000).fit()
