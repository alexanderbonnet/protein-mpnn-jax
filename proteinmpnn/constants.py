"""Useful constants."""

ATOM_INDICES = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}

ATOM_PAIR_RBFS = [
    # ("CA", "CA"),
    ("N", "N"),
    ("C", "C"),
    ("O", "O"),
    ("CB", "CB"),
    ("CA", "N"),
    ("CA", "C"),
    ("CA", "O"),
    ("CA", "CB"),
    ("N", "C"),
    ("N", "O"),
    ("N", "CB"),
    ("CB", "C"),
    ("CB", "O"),
    ("O", "C"),
    ("N", "CA"),
    ("C", "CA"),
    ("O", "CA"),
    ("CB", "CA"),
    ("C", "N"),
    ("O", "N"),
    ("CB", "N"),
    ("C", "CB"),
    ("O", "CB"),
    ("C", "O"),
]

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# default model hyperparameters
DEFAULT_HYPERPARAMS = {
    "dim": 128,
    "k": 48,
    "num_encoder_blocks": 3,
    "num_decoder_blocks": 3,
    # trained with 0.1 dropout, will be set to zero for inference
    "dropout_rate": 0.1,
    "vocab": len(ALPHABET),
}
