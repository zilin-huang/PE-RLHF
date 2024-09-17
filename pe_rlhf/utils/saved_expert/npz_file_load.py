import numpy as np

def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model
    # except FileNotFoundError:
    # print("Can not find {}, didn't load anything".format(path))
    # return None


# Specify the path to the npz file
file_path = "/home/sky-lab/codes/PE-RLHF/pe_rlhf/utils/saved_expert/data/test_pe_rlhf.npz"

# Load the npz file using np.load
data = load_weights(file_path)

# View the keys (array names) contained in the file
array_keys = data.files
print("Arrays in the file:", array_keys)

# Access and manipulate these arrays
# for key in array_keys:
#     array_data = data[key]
#     print(f"Array '{key}':")
#     print(array_data)

# Close the file (recommended to close the file after using the data)
data.close()
