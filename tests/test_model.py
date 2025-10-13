# import json
# import tomllib
# from pathlib import Path

# import equinox as eqx
# import jax.numpy as jnp
# import jax.random as jr
# import pytest

# from proteinmpnn import model as model_
# from proteinmpnn import parse

# filepath = Path(__file__)


# @pytest.mark.parametrize(
#     "model_config_path, model_weights_path, structure_path, result_path",
#     [
#         (
#             filepath.parent.parent / "configs/models/v_48_002.toml",
#             filepath.parent.parent / "weights/v_48_002.eqx",
#             filepath.parent.parent / "examples/inputs/monomers/5L33.cif.gz",
#             filepath.parent / "assets/test-model.json",
#         )
#     ],
# )
# def test_model_v_48_002_sample(
#     model_config_path: Path,
#     model_weights_path: Path,
#     structure_path: Path,
#     result_path: Path,
# ) -> None:
#     model_config = tomllib.loads(model_config_path.read_text())

#     model = model_.ProteinMPNN(
#         **model_config["model"],
#         # the key does not matter here
#         key=jr.PRNGKey(0),
#     )

#     model = eqx.tree_deserialise_leaves(model_weights_path, model)

#     structure = parse.read_structure(structure_path, use_assembly=True)
#     residues = parse.parse_backbone(structure)
#     tensors = parse.prepare_tensors(residues)

#     sampled = model.sample(
#         sequence=jnp.zeros_like(tensors.residue_index),
#         pos=tensors.pos,
#         residue_index=tensors.residue_index,
#         chain_labels=tensors.chain_labels,
#         mask_nodes=tensors.mask,
#         decoding_order=jr.permutation(
#             x=jnp.arange(tensors.residue_index.shape[0]), key=jr.PRNGKey(42)
#         ),
#         key=jr.PRNGKey(53),
#         top_k=1,
#         temperature=1.0,
#     )

#     result = json.loads(result_path.read_text())

#     assert sampled.tolist() == result["result"]
