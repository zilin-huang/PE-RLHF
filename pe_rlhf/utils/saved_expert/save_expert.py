import pickle
import os
import numpy as np


# This function extracts and saves model weights from a model checkpoint file into a compressed .npz file,
# with an option to remove weights related to the value network
def compress_model(ckpt_path, path="safe_expert.npz", remove_value_network=False):
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        # Usually, "_optimizer_variables" are used to store the state of the optimizer,
        # which is typically used during training but not needed when loading model weights.
        worker["state"]["default_policy"].pop("_optimizer_variables")
    pickled_worker = pickle.dumps(worker)
    weights = worker["state"]["default_policy"]
    for i in weights.keys():
        print(i)
        print(weights[i].shape)
    if remove_value_network:
        weights = {k: v for k, v in weights.items() if "value" not in k}

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return

    try:
        np.savez_compressed(path, **weights)
        print("Numpy agent weight is saved at: {}!".format(path))
    except Exception as e:
        print(f"Error saving the numpy weights to {path}: {e}")


if __name__ == "__main__":
    ckpt = "/home/sky-lab/codes/PE-RLHF/pe_rlhf/run_main_exp/PE_RLHF/PE_RLHFTrainer_HumanInTheLoopEnv_a019e_00000_0_seed=0_2024-06-17_17-12-03/checkpoint_100/checkpoint-100"
    compress_model(ckpt, path="test_pe_rlhf.npz")
