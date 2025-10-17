import tomllib
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import loguru

from proteinmpnn import constants, mpnn, parse, utils


def write_fasta(sequence: str, output_path: str | Path, header: str) -> None:
    """Write a sequence to a FASTA file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join([header, sequence]))


def main(config_path: str):
    config = tomllib.loads(Path(config_path).read_text())

    backbone = parse.read_structure(
        config["inference"]["backbone_path"], use_assembly=True
    )
    residues = parse.parse_backbone(backbone)
    inputs = parse.prepare_tensors(residues)

    key1, key2, key3 = jr.split(jr.PRNGKey(config["seed"]), num=3)

    decoding_order, decoding_start_index = utils.build_decoding_order(
        backbone_chains=[r.chain for r in residues],
        fixed_positions=config["inference"]["fixed_positions"],
        fixed_chains=config["inference"]["fixed_chains"],
        key=key1,
    )

    model = mpnn.ProteinMPNN(**config["model"], key=key2)
    model = eqx.tree_deserialise_leaves(
        path_or_file=Path(config["model_weights"]), like=model
    )
    nn.inference_mode(pytree=model, value=True)
    loguru.logger.info("Loaded model weights.")

    loguru.logger.info("Sampling sequence...")

    sampled_sequence = mpnn.sample(
        model=model,
        sequence=inputs.restypes,
        pos=inputs.pos,
        residue_index=inputs.residue_index,
        chain_labels=inputs.chain_labels,
        mask_nodes=inputs.mask,
        decoding_order=decoding_order,
        decoding_start_index=decoding_start_index,
        key=key3,
        top_k=config["inference"]["top_k"],
        temperature=config["inference"]["temperature"],
    )
    loguru.logger.info("Sampling sequence completed.")

    write_fasta(
        sequence="".join([constants.ALPHABET[s] for s in sampled_sequence]),
        output_path=config["inference"]["output_path"],
        header=">sequence",
    )
    loguru.logger.info(f"Wrote output to {config['inference']['output_path']}.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
