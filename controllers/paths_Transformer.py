# controllers/paths_Transformer.py
import os

def base_dir(p: int, model_tag: str, d_model: int, k: int) -> str:
    d = f"/home/mila/w/weis/scratch/DL/Transformer_dihedral/hypertuning1/qualitative_{p}_{model_tag}_dmodel_{d_model}_k_{k}"
    os.makedirs(d, exist_ok=True)
    return d

def model_dir(p: int, batch_size: int, weight_decay: float, epochs: int, training_set_size: int,
              base: str, d_model: int, nn_multiplier: int, num_mlp_layers: int, attn_coeff: float) -> str:
    d = os.path.join(
        base, f"{p}_models_embed_{d_model}",
        f"p={p}_bs={batch_size}_nn={nn_multiplier}_wd={weight_decay}_epochs={epochs}_"
        f"training_set_size={training_set_size}_mlpL={num_mlp_layers}_attn={attn_coeff}"
    )
    os.makedirs(d, exist_ok=True)
    return d

def seed_graph_dir(root: str, seed: int) -> str:
    g = os.path.join(root, f"graphs_seed_{seed}")
    os.makedirs(g, exist_ok=True)
    return g
