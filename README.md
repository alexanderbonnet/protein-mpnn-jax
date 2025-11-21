# protein-mpnn-jax

A JAX implementation of Protein MPNN, a message passing graph neural network for inverse folding.

## Dependencies

Dependencies are managed using [uv](https://docs.astral.sh/uv/).

## Example

Example backbones are stored in `examples/`.

```python
import proteinmpnn

sampled = proteinmpnn.sample(
    model_name="soluble/v_48_002",
    backbone_path="examples/inputs/multimers/5WPA.cif.gz",
    top_k=1,
    temperature=0.1,
    fixed_positions=[],
    fixed_chains=["A"],
    seed=0,
    progress_bar=False,
)

# sampled[1] -> SampledSequence
# {'sequence': 'KTYTQRCRLFVGNLPADITEDEFK...', 'chain': 'A', probabilities: [0.79, 0.48, 1.0, 0.97, 0.71, ...]}
```

```bash
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
