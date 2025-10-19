import collections
import dataclasses
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import loguru

from proteinmpnn import constants, mpnn, parse, utils


@dataclasses.dataclass
class SampledSequence:
    sequence: str
    chain: str


def sample(
    weights_path: str | Path,
    backbone_path: str | Path,
    top_k: int = 1,
    temperature: float = 1.0,
    fixed_positions: list[int] | None = None,
    fixed_chains: list[str] | None = None,
    seed: int = 42,
) -> list[SampledSequence]:
    backbone = parse.read_structure(str(backbone_path), use_assembly=True)
    residues = parse.parse_backbone(backbone)
    inputs = parse.prepare_tensors(residues)

    key1, key2, key3 = jr.split(jr.PRNGKey(seed), num=3)

    decoding_order, decoding_start_index = utils.build_decoding_order(
        backbone_chains=[r.chain for r in residues],
        fixed_positions=fixed_positions,
        fixed_chains=fixed_chains,
        key=key1,
    )

    model = mpnn.ProteinMPNN(**constants.DEFAULT_HYPERPARAMS, key=key2)  # type: ignore[arg-type]
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

    results = []
    k = 0
    for chain in chains:
        sequence = "".join(
            [constants.ALPHABET[s] for s in sampled[k : k + chain_lengths[chain]]]
        )
        results.append(SampledSequence(sequence=sequence, chain=chain))
        k += chain_lengths[chain]

    return results
