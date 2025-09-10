# controllers/config_Transformer.py
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # 共用超参
    learning_rate: float
    weight_decay: float
    p: int
    batch_size: int
    optimizer: str
    epochs: int
    k: int
    batch_experiment: str
    random_seeds: List[int]

    # Transformer 专属
    d_model: int
    d_head: int
    num_heads: int
    n_ctx: int
    act_type: str
    attn_coeff: float
    nn_multiplier: int
    num_mlp_layers: int
    eval_every: int = 1  # 默认每个 epoch 评估一次

    @classmethod
    def from_argv(cls, argv: list[str]) -> "Config":
        """
        Usage:
          script.py <lr> <wd> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment>
                    <d_model> <d_head> <num_heads> <n_ctx> <act_type> <attn_coeff>
                    <nn_multiplier> <num_mlp_layers> <eval_every> <seed1> [<seed2> ...]
        """
        if len(argv) < 19:
            raise SystemExit(
                "Usage: script.py <lr> <wd> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> "
                "<d_model> <d_head> <num_heads> <n_ctx> <act_type> <attn_coeff> "
                "<nn_multiplier> <num_mlp_layers> <eval_every> <seed1> [<seed2> ...]"
            )
        return cls(
            learning_rate=float(argv[1]),
            weight_decay=float(argv[2]),
            p=int(argv[3]),
            batch_size=int(argv[4]),
            optimizer=argv[5],
            epochs=int(argv[6]),
            k=int(argv[7]),
            batch_experiment=argv[8],
            d_model=int(argv[9]),
            d_head=int(argv[10]),
            num_heads=int(argv[11]),
            n_ctx=int(argv[12]),
            act_type=str(argv[13]),
            attn_coeff=float(argv[14]),
            nn_multiplier=int(argv[15]),
            num_mlp_layers=int(argv[16]),
            eval_every=int(argv[17]),
            random_seeds=[int(x) for x in argv[18:]],
        )
