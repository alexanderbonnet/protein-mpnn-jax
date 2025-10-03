"""Useful constants."""

EPS = 1e-6

ATOM_INDICES = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "O": 3,
    "CB": 4,
}

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
