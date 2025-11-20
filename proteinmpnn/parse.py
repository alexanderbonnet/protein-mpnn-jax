"""Parsing utilities for protein structures."""

import dataclasses
from pathlib import Path

import gemmi
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int
from loguru import logger

from proteinmpnn import constants


@dataclasses.dataclass(frozen=True, slots=True)
class BackboneResidue:
    """Container for a single residue's structural information.

    A residue is considered missing if at least one of its backbone atoms is missing.
    """

    index: int
    restype: int
    resname: str
    carbon_alpha_pos: Float[Array, " 3"]
    carbon_pos: Float[Array, " 3"]
    nitrogen_pos: Float[Array, " 3"]
    oxygen_pos: Float[Array, " 3"]
    chain: str
    chain_index: int
    missing: bool


def read_structure(filepath: str | Path, use_assembly: bool = True) -> gemmi.Structure:
    """Read and clean a structure from a PDB or mmCIF file. Optionally transform to biological assembly 1."""
    structure = gemmi.read_structure(str(filepath))
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


def get_single_letter_code(residue: gemmi.Residue) -> str:
    """Get a single-letter amino acid code for a gemmi Residue."""
    residue_info = gemmi.find_tabulated_residue(residue.name)
    if residue_info.is_amino_acid() and residue_info.is_standard():
        single_letter = residue_info.one_letter_code
        single_letter = single_letter.upper()
        # non-standard residues derived from a parent std residue are lowercase
        single_letter = single_letter if single_letter in constants.VOCABULARY else "X"
        return single_letter
    return "X"


def parse_backbone(structure: gemmi.Structure) -> list[BackboneResidue]:
    """Parse a gemmi Structure into an AtomStructure object."""
    # only parse the first model
    model = structure[0]

    residues = []
    for idx, chain in enumerate(model):
        for residue in chain.get_polymer():
            # zero-index the residue index
            residue_index = residue.label_seq - 1  # type: ignore[operator]
            atom_pos = {atom.name: atom.pos.tolist() for atom in residue}

            single_letter_code = get_single_letter_code(residue)
            restype = constants.VOCABULARY.index(single_letter_code)

            # if any of the atoms are missing, consider that the entire residue is missing
            missing = any(atom not in atom_pos for atom in ["CA", "C", "N", "O"])
            backbone_residue = BackboneResidue(
                index=residue_index,
                restype=restype,
                resname=single_letter_code,
                chain=chain.name,
                carbon_alpha_pos=jnp.array(atom_pos.get("CA", jnp.zeros(3))),
                carbon_pos=jnp.array(atom_pos.get("C", jnp.zeros(3))),
                nitrogen_pos=jnp.array(atom_pos.get("N", jnp.zeros(3))),
                oxygen_pos=jnp.array(atom_pos.get("O", jnp.zeros(3))),
                chain_index=idx,
                missing=missing,
            )
            residues.append(backbone_residue)

    return residues


# NOTE: add sequence parsing for training
@dataclasses.dataclass(frozen=True, slots=True)
class BackBoneTensors:
    """Container for tensors representing a protein backbone."""

    pos: Float[Array, "n 4 3"]
    residue_index: Int[Array, " n"]
    chain_labels: Int[Array, " n"]
    mask: Bool[Array, " n"]
    restypes: Int[Array, " n"]


def prepare_tensors(residues: list[BackboneResidue]) -> BackBoneTensors:
    """Prepare tensors from a list of BackboneResidue objects."""
    pos = jnp.zeros(shape=(len(residues), 4, 3))
    pos = pos.at[:, constants.ATOM_INDICES["CA"], :].set([r.carbon_alpha_pos for r in residues])
    pos = pos.at[:, constants.ATOM_INDICES["C"], :].set([r.carbon_pos for r in residues])
    pos = pos.at[:, constants.ATOM_INDICES["O"], :].set([r.oxygen_pos for r in residues])
    pos = pos.at[:, constants.ATOM_INDICES["N"], :].set([r.nitrogen_pos for r in residues])
    return BackBoneTensors(
        pos=pos,
        chain_labels=jnp.array([r.chain_index for r in residues], dtype=jnp.int32),
        residue_index=jnp.array([r.index for r in residues], dtype=jnp.int32),
        mask=~jnp.array([r.missing for r in residues], dtype=jnp.bool),
        restypes=jnp.array([r.restype for r in residues], dtype=jnp.int32),
    )
