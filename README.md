# Minty-DRS-examples
[[`arXiv`](https://arxiv.org/abs/2305.03605)]
This repository contains the code for the numerical experiments of the paper:
> [**Convergence of the Preconditioned Proximal Point Method and Douglas–Rachford Splitting in the Absence of Monotonicity**](https://arxiv.org/pdf/2305.03605.pdf)
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
- [toy.py](examples/toy.py): Example 2.4 from the paper (Figure 2)
- [saddle.py](examples/saddle.py): Example 6.1 from the paper (Figure 5)
- [saddle-semi.py](examples/saddle-semi.py): Example 6.1 from the paper (Figure 8)
- [stationary.py](examples/stationary.py): Example 6.3 from the paper (Figure 7)

