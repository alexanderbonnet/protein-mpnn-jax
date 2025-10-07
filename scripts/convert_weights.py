import equinox as eqx
import jax
import jax.numpy as jnp
import torch

conversion_map = {
    "positional_encoding.linear.weight": "features.embeddings.linear.weight",
    "positional_encoding.linear.bias": "features.embeddings.linear.bias",
    "edge_embedding.weight": "features.edge_embedding.weight",
    "edge_norm.weight": "features.norm_edges.weight",
    "edge_norm.bias": "features.norm_edges.bias",
}


def update_eqx_with_state_dict(
    module: eqx.Module,
    state_dict: dict[str, torch.Tensor],
    conversion_map: dict[str, str],
) -> eqx.Module:
    path_vals, treedef = jax.tree.flatten_with_path(module)
    updated_path_vals = []
    for names, array in path_vals:
        key = ".".join(x.name for x in names)
        if key in conversion_map:
            weights = state_dict["model_state_dict"][conversion_map[key]]
            updated_path_vals.append((names, jnp.asarray(weights)))
        else:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)
    return updated_module
