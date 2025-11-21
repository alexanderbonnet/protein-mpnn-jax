import collections
from pathlib import Path
from typing import TypedDict

import equinox.nn as nn
import jax.random as jr
import loguru

from proteinmpnn import constants, mpnn, parse, utils


class SampleResult(TypedDict):
    sequence: str
    chain: str
    probabilities: list[float]


def sample(
    model_name: str,
    backbone_path: str | Path,
    top_k: int = 1,
    temperature: float = 1.0,
    fixed_positions: list[int] | None = None,
    fixed_chains: list[str] | None = None,
    seed: int = 42,
    progress_bar: bool = False,
) -> list[SampleResult]:
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

    model: mpnn.ProteinMPNN
    model = mpnn.ProteinMPNN.from_pretrained(model_name, key=key2)
    model = nn.inference_mode(pytree=model, value=True)

    loguru.logger.info("Loaded model weights. Sampling...")

    sampled, probabilities = model.sample(
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
        progress_bar=progress_bar,
    )
    loguru.logger.info("Sampling sequence completed.")

    chain_lengths = collections.Counter([r.chain for r in residues])
    chains = list(dict.fromkeys([r.chain for r in residues]))

    results = []
    k = 0
    for chain in chains:
        sequence = "".join([constants.VOCABULARY[s] for s in sampled[k : k + chain_lengths[chain]]])
        probs = probabilities[k : k + chain_lengths[chain]].tolist()
        results.append(SampleResult(sequence=sequence, chain=chain, probabilities=probs))
        k += chain_lengths[chain]

    return results
