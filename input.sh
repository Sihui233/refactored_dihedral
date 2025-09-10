# "Usage: script.py <learning_rate> <weight_decay> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> <num_neurons> <MLP_class> <features> <num_layers> <random_seed_int_1> [<random_seed_int_2> ...]")
# python3 train_mlp_multilayer_dihedral.py 0.001 0.0001 18 18 adam 2500 58 random_random 128 two_embed 128 1 5
# python3 train_mlp_multilayer_dihedral.py 0.001 0.0001 19 19 adam 2500 61 random_random 128 two_embed 128 1 0 1 2 3 4 5

python3 /home/mila/w/weis/DL/refactored_dihedral/run/run_training_MLP.py 0.001 0.0001 24 24 adam 2500 77 random_random 512 two_embed 128 1 0

# python3 train_mlp_multilayer_dihedral.py 0.001 0.0001 25 25 adam 2500 80 random_random 512 two_embed 128 1 0 1 2 3

# python3 train_mlp_multilayer_dihedral.py 0.001 0.0001 34 34 adam 2500 109 random_random 512 two_embed 128 1 0 1 2 3

# python3 train_mlp_multilayer_dirichlet_bs_metrics_added.py 0.001 0.0001 18 18 adam 2500 18 random_random 128 one_embed 128 1 0 1 2 3 4

