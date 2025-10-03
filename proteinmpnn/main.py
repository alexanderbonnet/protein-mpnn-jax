import dataclasses
import os

import gemmi
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from loguru import logger


@dataclasses.dataclass(frozen=True, slots=True)
class ResidueStructure:
    index: int
    carbon_alpha_pos: Float[Array, " 3"]
    carbon_pos: Float[Array, " 3"]
    nitrogen_pos: Float[Array, " 3"]
    oxygen_pos: Float[Array, " 3"]
    chain_index: int

    @property
    def carbon_beta(self):
        b = self.carbon_alpha_pos - self.nitrogen_pos
        c = self.carbon_pos - self.carbon_alpha_pos
        a = jnp.cross(b, c)
        return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + self.carbon_alpha_pos


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


def parse_structure(structure: gemmi.Structure) -> list[ResidueStructure]:
    """Parse a gemmi Structure into an AtomStructure object."""
    # only parse the first model
    model = structure[0]

    residues = []
    for i, chain in enumerate(model):
        for residue in enumerate(chain.get_polymer()):
            residue_index = residue.label_seq - 1  # zero-indexed
            atom_pos = {atom.name: atom.pos.tolist() for atom in residue}
            residue_structure = ResidueStructure(
                index=residue_index,
                carbon_alpha_pos=jnp.array(atom_pos["CA"], dtype=jnp.float32),
                carbon_pos=jnp.array(atom_pos["C"], dtype=jnp.float32),
                nitrogen_pos=jnp.array(atom_pos["N"], dtype=jnp.float32),
                oxygen_pos=jnp.array(atom_pos["O"], dtype=jnp.float32),
                chain_index=i,
            )
            residues.append(residue_structure)

    return residues


def main():
    print("Hello from protein-mpnn-jax!")


if __name__ == "__main__":
    main()
