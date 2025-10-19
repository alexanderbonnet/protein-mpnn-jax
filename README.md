# protein-mpnn-jax

A JAX implementation of Protein MPNN, a message passing graph neural network for inverse folding.

All credit goes to the [paper](https://www.science.org/doi/10.1126/science.add2187) authors. This repository is based off of the paper's [repository](https://github.com/dauparas/ProteinMPNN). Logit outputs for the model match the original repository.

## Getting started

### Dependencies

Dependencies are managed using [uv](https://docs.astral.sh/uv/).

### Weights

Weights are available on [google drive](https://drive.google.com/drive/folders/1Nv2NJvWq3rCzHtWN8Qap-aAYrS9DR2AP?usp=sharing). They can be downloaded using the `scripts/download_weights.py` script. They were converted from PyTorch to JAX using the `scripts/convert_weights` script.

### Example

Example backbones are stored in `examples/`.

```python
import proteinmpnn

sampled = proteinmpnn.sample(
    weights_path="weights/v_48_002.eqx",
    backbone_path="examples/inputs/multimers/5WPA.cif.gz",
    top_k=1,
    temperature=0.1,
    fixed_positions=[],
    fixed_chains=["A"],
    seed=0,
    progress_bar=False,
)

# sampled[0] = SampledSequence(sequence='MVKSYLEPGEKEYTNRCELFVGNLPKDMTMEKFKELFKKYGEPKNVFLNKEKGYGYISLRSRNRANIAKSELNGKEVGNKPLVIRFKKLEAALTVGNLDPEVTDELLREAFGQFGPVERAVVLVDKEGRATGRGEVLFETKEPAEKALKECSEKSFLLTSNPRPVIVEPKEELDDEIGRPEEEMEETEEYKKERAKGPRFAKPGTKEYKLASAWRKLEKEEKQQREQVEEMYKEQRESLEKYFEEEREKREAEKE', chain='B')
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
