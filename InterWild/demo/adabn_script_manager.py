import subprocess
import os
import itertools
from datetime import datetime
import random
import json
from multiprocessing import Pool

# Define base paths
BASE_DIR = "C:/Users/vladi/RP/BN_hps_results"
MODEL_PATH = r"C:\Users\vladi\RP\InterWild\output\model_dump_old\snapshot_8_pck535handcoral.pth"
UPDATE_FOLDER = "C:/Users/vladi/RP/Research/IR_videos/every60thframe"
TEST_FOLDER = "C:/Users/vladi/RP/our_hands_dataset_labeled_previews/IR"

# Create results directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

# Parameters to test
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
MOMENTUM_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
TRAIN_BN_VALUES = [True, False]
POST_UPDATE_VALUES = [True, False]

# Add new parameters
UPDATE_FOLDER_ENABLED_VALUES = [True, False]
TEST_FOLDER_ENABLED_VALUES = [True, False]

def run_adabn(params):
    """Run the AdaBN script with given parameters"""
    # Construct filename based on parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_deepcoral_trainBN{params['train_bn']}_postUpdate{params['post_update']}"
    filename += f"_updateEnabled{params['update_folder_enabled']}_testEnabled{params['test_folder_enabled']}"
    
    
    if params['train_bn']:
        filename += f"_batch{params['batch_size']}"
        filename += f"_momentum{params['momentum']}"
    
    filename += f"_{timestamp}.json"
    output_path = os.path.join(BASE_DIR, filename)
    
    # Construct command with new parameters
    cmd = [
        "python", "adabn_script.py",
        "--model_path", MODEL_PATH,
        "--update_folder", UPDATE_FOLDER,
        "--test_folder", TEST_FOLDER,
        "--output_json", output_path,
        "--train_bn", str(params['train_bn']),
        "--post_update", str(params['post_update']),
        "--batch_size", str(params['batch_size']),
        "--momentum", str(params['momentum']),
        "--update_folder_enabled", str(params['update_folder_enabled']),
        "--test_folder_enabled", str(params['test_folder_enabled'])
    ]
    
    # Run the command
    print(f"\nRunning configuration: {filename}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully completed: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error running configuration: {filename}")
        print(f"Error: {e}")

def main():
    configs = []

    # 1) Define "priority" configurations in the specified order
    #    - update_folder_enabled=True, test_folder_enabled=True, batch_size=16, train_bn=True, post_update=False, momentum=0.1
    #    - same but with update_folder_enabled=False
    #    - then vary momentum with update_folder_enabled in [True, False], keeping other parameters the same
    priority_configs = [
    ]

    for update_folder_state in [True, False]:
        for momentum_value in [0.01, 0.05, 0.2, 0.4, 0.8]:
            priority_configs.append({
                'train_bn': True,
                'post_update': False,
                'batch_size': 16,
                'momentum': momentum_value,
                'update_folder_enabled': update_folder_state,
                'test_folder_enabled': True
            })
    
    # 2) New priority configurations with varying batch sizes and momentums
    new_priority_configs = [
        {
            'train_bn': True,
            'post_update': False,
            'batch_size': 16,
            'momentum': 0.05,
            'update_folder_enabled': False,
            'test_folder_enabled': True
        }
    ]
    for batch_size in BATCH_SIZES:
        for momentum in MOMENTUM_VALUES:
            new_priority_configs.append({
                'train_bn': True,
                'post_update': False,
                'batch_size': batch_size,
                'momentum': momentum,
                'update_folder_enabled': False,
                'test_folder_enabled': True
            })

    # 3) Gather "all" possible configurations in a list (from the original nested loops)
    all_configs = []
    for train_bn in TRAIN_BN_VALUES:
        for post_update in POST_UPDATE_VALUES:
            for update_enabled in UPDATE_FOLDER_ENABLED_VALUES:
                for test_enabled in TEST_FOLDER_ENABLED_VALUES:
                    # Skip if both folders are disabled
                    if not update_enabled and not test_enabled:
                        continue

                    if train_bn:
                        # When train_bn is True, test all combinations of batch sizes and momentum
                        for batch_size in BATCH_SIZES:
                            for momentum in MOMENTUM_VALUES:
                                config_dict = {
                                    'train_bn': train_bn,
                                    'post_update': post_update,
                                    'batch_size': batch_size,
                                    'momentum': momentum,
                                    'update_folder_enabled': update_enabled,
                                    'test_folder_enabled': test_enabled
                                }
                                all_configs.append(config_dict)
                    else:
                        # When train_bn is False, use default batch_size and momentum
                        config_dict = {
                            'train_bn': train_bn,
                            'post_update': post_update,
                            'batch_size': 8,  # Default value
                            'momentum': 0.1,  # Default value
                            'update_folder_enabled': update_enabled,
                            'test_folder_enabled': test_enabled
                        }
                        all_configs.append(config_dict)

    # 4) Remove duplicates (if any) among the "all_configs"
    unique_configs = []
    seen_set = set()
    for conf in all_configs:
        serialized = json.dumps(conf, sort_keys=True)
        if serialized not in seen_set:
            seen_set.add(serialized)
            unique_configs.append(conf)

    # 5) Shuffle random configurations, then prepend priority configs so they run first
    random.shuffle(unique_configs)
    final_configs = new_priority_configs + priority_configs + unique_configs

    total_configs = len(final_configs)
    print(f"Total configurations to run: {total_configs}")

    # 6) Single loop that goes over final_configs
    completed_configs = 0
    


    for f in final_configs:
        run_adabn(f)

    
    """
    for conf in final_configs:
        run_adabn(conf)
        completed_configs += 1
        print(f"Progress: {completed_configs}/{total_configs}")
    """
if __name__ == "__main__":
    main()