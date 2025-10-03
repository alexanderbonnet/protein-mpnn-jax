import dataclasses
import os

import gemmi
import jax.numpy as jnp
from jaxtyping import Float, Int
from loguru import logger


@dataclasses.dataclass(frozen=True, slots=True)
class BackboneStructure:
    # residue data
    residue_index: Int[jnp.ndarray, " n"]

    # atom data
    atom_positions: Float[jnp.ndarray, "n 3"]

    # chain data
    chain_index: Int[jnp.ndarray, " n"]


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
    structure.remove_hydrogens()
    structure.remove_waters()
    structure.remove_empty_chains()

    return structure


def parse_structure(structure: gemmi.Structure) -> BackboneStructure:
    """Parse a gemmi Structure into an AtomStructure object."""
    # only parse the first model
    model = structure[0]

    chain_indices = []
    residue_indices = []
    atom_positions = []
    for i, chain in enumerate(model):
        for residue in chain.get_polymer():
            for atom in residue:
                if atom.name == "CA":
                    atom_positions.append(atom.pos.tolist())
                    residue_indices.append(residue.label_seq - 1)  # zero-indexed
                    chain_indices.append(i)

    return BackboneStructure(
        residue_index=jnp.array(residue_indices, dtype=jnp.int32),
        atom_positions=jnp.array(atom_positions, dtype=jnp.float32),
        chain_index=jnp.array(chain_indices, dtype=jnp.int32),
    )


def main():
    print("Hello from protein-mpnn-jax!")


if __name__ == "__main__":
    main()
