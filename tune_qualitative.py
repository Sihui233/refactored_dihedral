import subprocess

# ——— fixed from your example ———
learning_rate   = 0.0005
weight_decay    = 0.0001
p               = 59
batch_size      = 59
optimizer       = "adam"
epochs          = 3500
k               = 58
batch_experiment= "random_random"

# ——— easy to change later ———
features    = 128
num_neurons = 1024

# ——— your six options ———
model_class_map = {
    "no_embed":              "MLPOneHot",
    "one_embed":             "MLPOneEmbed",
    "two_embed":             "MLPTwoEmbed",
    "no_embed_cheating":     "MLPOneHot_cheating",
    "one_embed_cheating":    "MLPOneEmbed_cheating",
    "two_embed_cheating":    "MLPTwoEmbed_cheating",
}

def main():
    seeds = list(range(1, 101))

    for mlp_name in model_class_map:
        # for num_layers in (1, 2):
        num_layers = 3
        for seed in seeds:
            cmd = [
                "sbatch", "slurm_make_pca_homologies.sh",
                str(learning_rate),
                str(weight_decay),
                str(p),
                str(batch_size),
                optimizer,
                str(epochs),
                str(k),
                batch_experiment,
                str(num_neurons),
                mlp_name,
                str(features),
                str(num_layers),
                str(seed),
            ]

            print("Submitting:", " ".join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
