# protein-mpnn-jax

A JAX implementation of Protein MPNN.

All credit goes to the [paper](https://www.science.org/doi/10.1126/science.add2187) authors. This repository is based off of the paper's [repository](https://github.com/dauparas/ProteinMPNN).

:warning: The repository is subject to change significantly in the near future.

Logit outputs for the model match the original reposiotry.

## Getting started

### Dependencies

Dependencies are managed using [uv](https://docs.astral.sh/uv/).

### Examples

Weights are available on [google drive](https://drive.google.com/drive/folders/1Nv2NJvWq3rCzHtWN8Qap-aAYrS9DR2AP?usp=sharing). They can be downloaded using the `scripts/download_weights.py` script. They were converted from PyTorch to JAX using the `scripts/convert_weights` script.

Sample sequences with

```bash
uv run python -m proteinmpnn.run --config_path <path to config> --eqx_path <path to model weights>.
```

Examples for configuration files are found in `examples/configs`.



```bash
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
