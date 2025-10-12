"""Parsing utilities for protein structures."""

import dataclasses
import os

import gemmi
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int
from loguru import logger

from proteinmpnn import constants


@dataclasses.dataclass(frozen=True, slots=True)
class BackboneResidue:
    """Container for a single residue's structural information."""

    index: int
    carbon_alpha_pos: Float[Array, " 3"]
    carbon_pos: Float[Array, " 3"]
    nitrogen_pos: Float[Array, " 3"]
    oxygen_pos: Float[Array, " 3"]
    chain_index: int


def read_structure(filepath: os.PathLike, use_assembly: bool = True) -> gemmi.Structure:
    """Read and clean a structure from a PDB or mmCIF file. Optionally transform to biological assembly 1."""
    structure = gemmi.read_structure(filepath)
    if use_assembly:
        how = gemmi.HowToNameCopiedChain.Dup
        try:
            structure.transform_to_assembly(assembly_name="1", how=how)
        except RuntimeError:
            logger.info("No assembly '1' found in structure; using asymmetric unit.")

    # see https://gemmi.readthedocs.io/en/stable/mol.html#structure
    structure.setup_entities()
    structure.assign_label_seq_id()

    # clean up structure
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_waters()
    structure.remove_empty_chains()

    return structure


# NOTE: adapt parsing for training i.e handling missing residues, etc...
def parse_backbone(structure: gemmi.Structure) -> list[BackboneResidue]:
    """Parse a gemmi Structure into an AtomStructure object."""
    # only parse the first model
    model = structure[0]

    residues = []
    for i, chain in enumerate(model):
        for residue in chain.get_polymer():
            # zero-index the residue index
            residue_index = residue.label_seq - 1
            atom_pos = {atom.name: atom.pos.tolist() for atom in residue}
            residue_structure = BackboneResidue(
                index=residue_index,
                carbon_alpha_pos=jnp.array(atom_pos["CA"], dtype=jnp.float32),
                carbon_pos=jnp.array(atom_pos["C"], dtype=jnp.float32),
                nitrogen_pos=jnp.array(atom_pos["N"], dtype=jnp.float32),
                oxygen_pos=jnp.array(atom_pos["O"], dtype=jnp.float32),
                chain_index=i,
            )
            residues.append(residue_structure)

    return residues


# NOTE: add sequence parsing for training
@dataclasses.dataclass(frozen=True, slots=True)
class BackBoneTensors:
    """Container for tensors representing a protein backbone."""

    pos: Float[Array, "n 4 3"]
    residue_index: Int[Array, " n"]
    chain_labels: Int[Array, " n"]
    mask: Bool[Array, " n"]


def prepare_tensors(residues: list[BackboneResidue]) -> BackBoneTensors:
    """Prepare tensors from a list of BackboneResidue objects."""
    residue_index = jnp.array([r.index for r in residues], dtype=jnp.int32)
    chain_labels = jnp.array([r.chain_index for r in residues], dtype=jnp.int32)
    mask = jnp.ones_like(residue_index, dtype=jnp.bool)
    pos = jnp.zeros(shape=(residue_index.shape[0], 4, 3))
    pos = pos.at[:, constants.ATOM_INDICES["CA"], :].set(
        [r.carbon_alpha_pos for r in residues]
    )
    pos = pos.at[:, constants.ATOM_INDICES["C"], :].set(
        [r.carbon_pos for r in residues]
    )
    pos = pos.at[:, constants.ATOM_INDICES["O"], :].set(
        [r.oxygen_pos for r in residues]
    )
    pos = pos.at[:, constants.ATOM_INDICES["N"], :].set(
        [r.nitrogen_pos for r in residues]
    )
    return BackBoneTensors(
        pos=pos, chain_labels=chain_labels, residue_index=residue_index, mask=mask
    )
