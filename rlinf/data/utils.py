# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def batch_pad_to_fixed_len(
    batch: list[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad



import numpy as np
from rlinf.envs.libero.utils import quat2axisangle
class Libero_Robot_State:
    def __init__(self, robot_state=None, img=None, wrist_img=None):
        """
        Convert raw robot state into the format expected by downstream models.

        Recent TFRecords store Libero states as 8D vectors (6D pose + 2D gripper).
        Older records contained a 9D vector with a quaternion that needed conversion
        to axis-angle. Handle both layouts gracefully.
        """
        robot_state = np.asarray(robot_state)
        if robot_state.shape[-1] == 9:
            robot_state = np.concatenate(
                [robot_state[2:5], quat2axisangle(robot_state[5:9]), robot_state[0:2]],
                axis=0,
            )
        elif robot_state.shape[-1] == 8:
            # Already (position, orientation in 3D, gripper) so keep as-is.
            robot_state = robot_state
        else:
            raise ValueError(f"Unexpected robot_state dimension {robot_state.shape}")

        self.data = {
            "robot_state": robot_state,
            "img": img,
            "wrist_img": wrist_img,
        }
    
    def to_dict(self):
        return self.data
