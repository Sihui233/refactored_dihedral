# controllers/paths.py
import os

def base_dir(p: int, mlp_class_lower: str, num_neurons: int, features: int, k: int) -> str:
    d = f"/home/mila/w/weis/scratch/DL/MLP_dihedral/qualitative_{p}_{mlp_class_lower}_{num_neurons}_features_{features}_k_{k}"
    os.makedirs(d, exist_ok=True)
    return d

def model_dir(p: int, batch_size: int, num_neurons: int, weight_decay: float, epochs: int, training_set_size: int,
              base: str, features: int) -> str:
    d = os.path.join(base, f"{p}_models_embed_{features}",
                     f"p={p}_bs={batch_size}_nn={num_neurons}_wd={weight_decay}_epochs={epochs}_training_set_size={training_set_size}")
    os.makedirs(d, exist_ok=True)
    return d

def seed_graph_dir(root: str, seed: int) -> str:
    g = os.path.join(root, f"graphs_seed_{seed}")
    os.makedirs(g, exist_ok=True)
    return g
