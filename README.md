# Minty-DRS-examples
[[`arXiv`](https://arxiv.org/abs/2305.03605v2)]
This repository contains the code for the numerical experiments of the paper:
> [**Convergence of the Preconditioned Proximal Point Method and Douglas–Rachford Splitting in the Absence of Monotonicity**](https://arxiv.org/pdf/2305.03605v2.pdf)
>
> [Brecht Evens](https://www.kuleuven.be/wieiswie/nl/person/00123309), [Pieter Pas](https://www.kuleuven.be/wieiswie/nl/person/00131132), [Puya Latafat](https://www.kuleuven.be/wieiswie/nl/person/00113202), and [Panos Patrinos](https://www.kuleuven.be/wieiswie/nl/person/00102375).

## Dependencies
The required Python packages can be installed through the following command.
```bash
pip install -r requirements.txt
```

## Examples
To run an example script, simply perform the following command.
```bash
python example_script.py
```

The following scripts are provided in this repository:
- [toy.py](examples/toy.py): Example 2.6 from the paper (Figure 2)
- [von-neumann.py](examples/von-neumann.py): Example 2.7 from the paper (Figures 4 and 9)
- [saddle.py](examples/saddle.py): Example 6.1 from the paper (Figure 6)
- [saddle-semi.py](examples/saddle-semi.py): Example 6.1 from the paper (Figure 10)
- [stationary.py](examples/stationary.py): Example 6.4 from the paper (Figure 8)

