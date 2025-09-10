# controllers/config.py
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    learning_rate: float
    weight_decay: float
    p: int
    batch_size: int
    optimizer: str
    epochs: int
    k: int
    batch_experiment: str
    num_neurons: int
    MLP_class: str
    features: int
    num_layers: int
    random_seeds: List[int]


    @classmethod
    def from_argv(cls, argv: list[str]) -> "Config":
        if len(argv) < 14:
            raise SystemExit("Usage: script.py <learning_rate> <weight_decay> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> <num_neurons> <MLP_class> <features> <num_layers> <random_seed_int_1> [<random_seed_int_2> ...]")
        return cls(
            learning_rate=float(argv[1]),
            weight_decay=float(argv[2]),
            p=int(argv[3]),
            batch_size=int(argv[4]),
            optimizer=argv[5],
            epochs=int(argv[6]),
            k=int(argv[7]),
            batch_experiment=argv[8],
            num_neurons=int(argv[9]),
            MLP_class=argv[10],
            features=int(argv[11]),
            num_layers=int(argv[12]),
            random_seeds=[int(a) for a in argv[13:]],
        )
