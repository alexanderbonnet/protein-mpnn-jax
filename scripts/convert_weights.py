import equinox as eqx
import jax
import jax.numpy as jnp
import torch

NUM_ENCODER_BLOCKS = 3
NUM_DECODER_BLOCKS = 3

conversion_map = {
    "features.positional_encoding.linear.weight": "features.embeddings.linear.weight",
    "features.positional_encoding.linear.bias": "features.embeddings.linear.bias",
    "features.edge_embedding.weight": "features.edge_embedding.weight",
    "features.edge_norm.weight": "features.norm_edges.weight",
    "features.edge_norm.bias": "features.norm_edges.bias",
    "edge_linear.weight": "W_e.weight",
    "edge_linear.bias": "W_e.bias",
    "linear_out.weight": "W_out.weight",
    "linear_out.bias": "W_out.bias",
    "sequence_embedding.weight": "W_s.weight",
}

for k in range(NUM_ENCODER_BLOCKS):
    conversion_map.update(
        {
            f"encoder_blocks.[{k}].in_block.linear_in.weight": f"encoder_layers.{k}.W1.weight",
            f"encoder_blocks.[{k}].in_block.linear_in.bias": f"encoder_layers.{k}.W1.bias",
            f"encoder_blocks.[{k}].in_block.hidden.weight": f"encoder_layers.{k}.W2.weight",
            f"encoder_blocks.[{k}].in_block.hidden.bias": f"encoder_layers.{k}.W2.bias",
            f"encoder_blocks.[{k}].in_block.linear_out.weight": f"encoder_layers.{k}.W3.weight",
            f"encoder_blocks.[{k}].in_block.linear_out.bias": f"encoder_layers.{k}.W3.bias",
            f"encoder_blocks.[{k}].out_block.linear_in.weight": f"encoder_layers.{k}.W11.weight",
            f"encoder_blocks.[{k}].out_block.linear_in.bias": f"encoder_layers.{k}.W11.bias",
            f"encoder_blocks.[{k}].out_block.hidden.weight": f"encoder_layers.{k}.W12.weight",
            f"encoder_blocks.[{k}].out_block.hidden.bias": f"encoder_layers.{k}.W12.bias",
            f"encoder_blocks.[{k}].out_block.linear_out.weight": f"encoder_layers.{k}.W13.weight",
            f"encoder_blocks.[{k}].out_block.linear_out.bias": f"encoder_layers.{k}.W13.bias",
            f"encoder_blocks.[{k}].norm1.weight": f"encoder_layers.{k}.norm1.weight",
            f"encoder_blocks.[{k}].norm1.bias": f"encoder_layers.{k}.norm1.bias",
            f"encoder_blocks.[{k}].norm2.weight": f"encoder_layers.{k}.norm2.weight",
            f"encoder_blocks.[{k}].norm2.bias": f"encoder_layers.{k}.norm2.bias",
            f"encoder_blocks.[{k}].norm3.weight": f"encoder_layers.{k}.norm3.weight",
            f"encoder_blocks.[{k}].norm3.bias": f"encoder_layers.{k}.norm3.bias",
            f"encoder_blocks.[{k}].feedforward.linear_in.weight": f"encoder_layers.{k}.dense.W_in.weight",
            f"encoder_blocks.[{k}].feedforward.linear_in.bias": f"encoder_layers.{k}.dense.W_in.bias",
            f"encoder_blocks.[{k}].feedforward.linear_out.weight": f"encoder_layers.{k}.dense.W_out.weight",
            f"encoder_blocks.[{k}].feedforward.linear_out.bias": f"encoder_layers.{k}.dense.W_out.bias",
        }
    )

for k in range(NUM_DECODER_BLOCKS):
    conversion_map.update(
        {
            f"decoder_blocks.[{k}].block.linear_in.weight": f"decoder_layers.{k}.W1.weight",
            f"decoder_blocks.[{k}].block.linear_in.bias": f"decoder_layers.{k}.W1.bias",
            f"decoder_blocks.[{k}].block.hidden.weight": f"decoder_layers.{k}.W2.weight",
            f"decoder_blocks.[{k}].block.hidden.bias": f"decoder_layers.{k}.W2.bias",
            f"decoder_blocks.[{k}].block.linear_out.weight": f"decoder_layers.{k}.W3.weight",
            f"decoder_blocks.[{k}].block.linear_out.bias": f"decoder_layers.{k}.W3.bias",
            f"decoder_blocks.[{k}].norm1.weight": f"decoder_layers.{k}.norm1.weight",
            f"decoder_blocks.[{k}].norm1.bias": f"decoder_layers.{k}.norm1.bias",
            f"decoder_blocks.[{k}].norm2.weight": f"decoder_layers.{k}.norm2.weight",
            f"decoder_blocks.[{k}].norm2.bias": f"decoder_layers.{k}.norm2.bias",
            f"decoder_blocks.[{k}].feedforward.linear_in.weight": f"decoder_layers.{k}.dense.W_in.weight",
            f"decoder_blocks.[{k}].feedforward.linear_in.bias": f"decoder_layers.{k}.dense.W_in.bias",
            f"decoder_blocks.[{k}].feedforward.linear_out.weight": f"decoder_layers.{k}.dense.W_out.weight",
            f"decoder_blocks.[{k}].feedforward.linear_out.bias": f"decoder_layers.{k}.dense.W_out.bias",
        }
    )


def update_eqx_with_state_dict(
    module: eqx.Module,
    state_dict: dict[str, torch.Tensor],
    conversion_map: dict[str, str],
) -> tuple[eqx.Module, int]:
    path_vals, treedef = jax.tree.flatten_with_path(module)
    updated_path_vals = []
    count = 0
    for names, array in path_vals:
        key = ".".join(str(x).strip(".") for x in names)
        if key in conversion_map:
            weights = state_dict[conversion_map[key]]
            assert array.shape == weights.shape, (
                f"{array.shape} != {weights.shape} for {key=}"
            )
            updated_path_vals.append((names, jnp.asarray(weights)))
            count += 1
        else:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)
    return updated_module, count
