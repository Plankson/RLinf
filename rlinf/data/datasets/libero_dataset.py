import torch
import numpy as np
import os
from torch.nn import functional as F

from torch.utils.data import DataLoader
import os
import glob
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import io
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf.config.set_visible_devices([], 'GPU')
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
        if robot_state.shape[-1] == 8:
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

class RLDSPyTorchDataset(Dataset):
    def __init__(self, tf_dataset, instruction_to_id, is_chunk, target_task=None, chunk_size=1):
        self.data = []
        self.instruction_to_id = instruction_to_id
        for ep in tqdm(tf_dataset.as_numpy_iterator()):
            instructions = ep['steps/language_instruction']
            instruction = instructions[0].decode('utf-8')
            task_id = instruction_to_id[instruction]
            if target_task is not None and instruction != target_task:
                continue 
    
            actions = ep['steps/action']
            rewards = ep['steps/reward']
            states = ep['steps/observation/state']
            images = ep['steps/observation/image']
            wrist_images = ep['steps/observation/wrist_image']
            is_terminals = ep['steps/is_terminal']
            if not (len(actions) == len(rewards) == len(states) == len(images) == len(wrist_images)):
                continue
            totol_step = len(actions)
            for i in range(totol_step):
                is_terminal = (i==totol_step-1)
                state = Libero_Robot_State(states[i], images[i], wrist_images[i])
                next_step_id = min(i + chunk_size, totol_step - 1)
                next_state = Libero_Robot_State(states[next_step_id], images[next_step_id], wrist_images[next_step_id])
                goal_state = Libero_Robot_State(states[-1], images[-1], wrist_images[-1])
                remaining_steps = max(totol_step - 1 - i, 0)
                steps_to_goal = remaining_steps / float(chunk_size)
                if is_chunk:
                    action_chunk = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] * chunk_size).reshape(chunk_size, -1)                    
                    available = min(chunk_size, len(actions) - i)
                    action_chunk[:available] = actions[i:i+available]
                    action = action_chunk
                    reward = 0.0
                    for chunk_step in range(available):
                        reward += rewards[i + chunk_step]
                else:  
                    action = actions[i]
                    reward = rewards[i]
                
                data_dict = {
                    "state": state,
                    "next_state": next_state,
                    "goal_state": goal_state,
                    "action": action,
                    "reward": reward,
                    "instruction": instruction,
                    'task_id': task_id,
                    "is_terminal": is_terminal,
                    "success": True,    # all success traj
                    "steps_to_goal": steps_to_goal,
                }
                self.data.append(data_dict)
        
        print(f"Total transitions loaded: {len(self.data)}")
        print(f"Action chunking {'enabled' if is_chunk else 'disabled'} with size {chunk_size if is_chunk else 1}")


    def decode_image(self, image_bytes):
        """
            input: image stored in bytes format
            output: torch.Tensor [C, H, W], float32
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)  # [H, W, C]
        # image_np = np.array(image)[::-1, ::-1].copy()   # [H, W, C]
        image = np.transpose(image_np, (2, 0, 1))  # convert to [C, H, W]
        return torch.from_numpy(image) 

    def save_img(self, image_bytes, save_path):
        """
            input: image stored in bytes format
            output: save the image to the specified path
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(save_path)

    def __len__(self):
        return len(self.data)
    
    def strip_index(self, instr: str) -> str:
        """
        remove the index suffix from instruction
        eg: 
            'close the top drawer of the cabinet_1' -> 'close the top drawer of the cabinet'
            'turn on the stove_23' -> 'turn on the stove'
        """
        if "_" not in instr:
            return instr
        base, last = instr.rsplit("_", 1)
        if last.isdigit():
            return base
        return instr
    

    def __getitem__(self, idx):
            data = self.data[idx]
            obs = data['state'].to_dict()
            next_obs = data['next_state'].to_dict()
            goal_obs = data['goal_state'].to_dict()
            act = data['action']
            rew = data['reward']
            instruction = data['instruction']
            task_id = data['task_id']
            is_terminal = data['is_terminal']
            is_success = data['success']
            steps_to_goal = data['steps_to_goal']
            action = torch.tensor(act, dtype=torch.float32)
            # Map gripper channel from [-1, 1] to [0, 1] to match runtime expectations
            action[..., -1] = 0.5 * (action[..., -1] + 1.0)

            return {
                'state': {
                    'robot_state': torch.tensor(obs['robot_state'], dtype=torch.float32),
                    'img': self.decode_image(obs['img']),
                    'wrist_img': self.decode_image(obs['wrist_img'])
                },
                'next_state': {
                    'robot_state': torch.tensor(next_obs['robot_state'], dtype=torch.float32),
                    'img': self.decode_image(next_obs['img']),
                    'wrist_img': self.decode_image(next_obs['wrist_img'])
                },
                'goal_state': {
                    'robot_state': torch.tensor(goal_obs['robot_state'], dtype=torch.float32),
                    'img': self.decode_image(goal_obs['img']),
                    'wrist_img': self.decode_image(goal_obs['wrist_img'])
                },
                'action': action,
                'reward': torch.tensor(rew-1, dtype=torch.float32), # -1 if not finished, 0 if finished
                'instruction': instruction,
                'task_id': torch.tensor(task_id, dtype=torch.long),
                'is_terminal': torch.tensor(is_terminal, dtype=torch.bool),
                'is_success': torch.tensor(is_success, dtype=torch.bool),
                'steps_to_goal': torch.tensor(steps_to_goal, dtype=torch.float32),
            }

class Config:
    def __init__(self, path, batch_size=32, is_chunk=True, chunk_size=4, suffle=True):
        self.path = path
        self.batch_size = batch_size
        self.is_chunk = is_chunk
        self.chunk_size = chunk_size
        self.suffle = suffle
    
def load_rlds_dataloader(config, split_name="train"):
    batch_size = config.batch_size
    is_chunk = config.is_chunk
    chunk_size = config.chunk_size
    suffle = config.suffle
    data_dirs = config.path
    feature_description = {
        "steps/action": tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
        "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "steps/is_terminal": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "steps/language_instruction": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "steps/reward": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "steps/observation/state": tf.io.FixedLenSequenceFeature([8], tf.float32, allow_missing=True),
        "steps/observation/image": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "steps/observation/wrist_image": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    }

    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # get string2id table
    def get_all_instructions(tfrecord_files):
        instructions = set()
        for file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(file)
            parsed_dataset = raw_dataset.map(parse_example)
            for ep in parsed_dataset.as_numpy_iterator():
                instr = ep['steps/language_instruction'][0].decode('utf-8')
                instructions.add(instr)
        return sorted(instructions)
    all_tfrecord_files = []
    for data_dir in data_dirs:
        all_tfrecord_files.extend(sorted(glob.glob(os.path.join(data_dir, f"liber*-{split_name}.tfrecord-*-of-*"))))
        
    unique_instructions = get_all_instructions(all_tfrecord_files)
    instruction_to_id = {instr: idx for idx, instr in enumerate(unique_instructions)}
    num_task = instruction_to_id.__len__()
    print(instruction_to_id)

    datasets = []
    for data_dir in data_dirs:
        tfrecord_files = sorted(glob.glob(os.path.join(data_dir, f"liber*-{split_name}.tfrecord-*-of-*")))
        if not tfrecord_files:
            print(f"Warning: No matching TFRecord files found in {data_dir}")
            continue
            
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        parsed_dataset = raw_dataset.map(parse_example)
        rlds_dataset = RLDSPyTorchDataset(parsed_dataset, instruction_to_id,
                                          is_chunk=is_chunk, chunk_size=chunk_size
                                          )
        datasets.append(rlds_dataset)
        print(f"Loaded dataset from: {data_dir}, number of samples: {len(rlds_dataset)}")

    if not datasets:
        raise ValueError("There is no dataset!")
        
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Combined datset size: {len(combined_dataset)}")

    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=suffle,
        num_workers=4,  
        pin_memory=True  
    )
    dataloader.instruction_to_id = instruction_to_id
    dataloader.id_to_instruction = {v: k for k, v in instruction_to_id.items()}

    return dataloader, num_task 

if __name__ == "__main__":
    data_dirs = ["/data/users/qingyunpeng/dataset/modified_libero_rlds/libero_10_no_noops/1.0.0"]
    config = Config(path=data_dirs, batch_size=16, is_chunk=True, chunk_size=8, suffle=True)
    train_dataloader, _ = load_rlds_dataloader(config,)
    for batch in train_dataloader:
        print(batch['state']['robot_state'].shape)
        print(batch['state']['img'].shape)
        print(batch['state']['wrist_img'].shape)
        print(batch['action'].shape)
        print(batch['reward'].shape)
        print(batch['instruction'])
        break
