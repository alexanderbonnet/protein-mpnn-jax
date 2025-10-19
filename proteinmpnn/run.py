import collections
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import loguru

from proteinmpnn import constants, mpnn, parse, utils


def write_fasta(sequence: str, path: str | Path, header: str) -> None:
    """Write a sequence to a FASTA file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, sequence]))


def run(
    weights_path: str,
    backbone_path: str,
    output_path: str,
    top_k: int = 1,
    temperature: float = 1.0,
    fixed_positions: list[int] | None = None,
    fixed_chains: list[str] | None = None,
    seed: int = 42,
) -> None:
    backbone = parse.read_structure(backbone_path, use_assembly=True)
    residues = parse.parse_backbone(backbone)
    inputs = parse.prepare_tensors(residues)

    key1, key2, key3 = jr.split(jr.PRNGKey(seed), num=3)

    decoding_order, decoding_start_index = utils.build_decoding_order(
        backbone_chains=[r.chain for r in residues],
        fixed_positions=fixed_positions,
        fixed_chains=fixed_chains,
        key=key1,
    )

    model = mpnn.ProteinMPNN(**constants.DEFAULT_HYPERPARAMS, key=key2)
    model = eqx.tree_deserialise_leaves(path_or_file=weights_path, like=model)
    nn.inference_mode(pytree=model, value=True)
    loguru.logger.info("Loaded model weights.")

    loguru.logger.info("Sampling sequence...")

    sampled = mpnn.sample(
        model=model,
        sequence=inputs.restypes,
        pos=inputs.pos,
        residue_index=inputs.residue_index,
        chain_labels=inputs.chain_labels,
        mask_nodes=inputs.mask,
        decoding_order=decoding_order,
        decoding_start_index=decoding_start_index,
        key=key3,
        top_k=top_k,
        temperature=temperature,
    )
    loguru.logger.info("Sampling sequence completed.")

    chain_lengths = collections.Counter([r.chain for r in residues])
    chains = list(dict.fromkeys([r.chain for r in residues]))

    fasta_text = []
    k = 0
    for chain in chains:
        header = f">{chain=}"
        sequence = "".join(
            [constants.ALPHABET[s] for s in sampled[k : k + chain_lengths[chain]]]
        )
        fasta_text.extend([header, sequence])
        k += chain_lengths[chain]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(fasta_text))

    loguru.logger.info(f"Wrote output to {output_path}.")


if __name__ == "__main__":
    import fire

    fire.Fire(run)
