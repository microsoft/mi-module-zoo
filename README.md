## Machine Intelligence PyTorch Module Zoo  ![PyPI](https://img.shields.io/pypi/v/mi-module-zoo?style=flat-square)

This package contains implementations standalone, commonly reusable PyTorch `nn.Module`s. To
install it run `pip install mi-module-zoo`.

This library is maintained by the [Machine Intelligence group](https://www.microsoft.com/en-us/research/theme/machine-intelligence/) in Microsoft Research.

### Modules

A list of the modules follows, for detailed documentation, please check the docstring
of each module.

* `mi_model_zoo.mlp.construct_mlp()` A function that generates an `nn.Sequential` for a
    multilinear perceptron.
* `mi_model_zoo.settransformer.SetTransformer` The [Set Transformer models](https://arxiv.org/abs/1810.00825).
* `mi_model_zoo.settransformer.ISAB` An Inducing-point Self-Attention Block from the [Set Transformer paper](https://arxiv.org/abs/1810.00825).
* `mi_model_zoo.RelationalMultiheadAttention` The relational multi-head attention variants,
   supporting both sparse and dense relationships,
   including [Shaw et. al. (2019)](https://www.aclweb.org/anthology/N18-2074/), [RAT-SQL](https://arxiv.org/pdf/1911.04942.pdf),
   and [GREAT](https://openreview.net/pdf?id=B1lnbRNtwr) variants.
* `mi_model_zoo.relationaltransformerlayers.RelationalTransformerEncoderLayer` A relational
   transformer encoder layer that supports both dense and sparse relations among elements. Supports
   ReZero and a variety of normalization modes.
* `mi_model_zoo.relationaltransformerlayers.RelationalTransformerDecoderLayer` A relational
   transformer decoder layer that supports both dense and sparse relations among encoded-decoded
   and decoded-decoded elements. Supports ReZero and a variety of normalization modes.



### Utilities
* `mi_model_zoo.utils.randomutils.set_seed()` Set the seed across Python, NumPy, and PyTorch (CPU+CUDA).
* `mi_model_zoo.utils.activationutils.get_activation_fn()` Get an activation function by name.


### Developing
To develop in this repository, clone the repository, install [pre-commit](https://pre-commit.com/), and run
```bash
pre-commit install
```

##### Releasing to pip
To deploy a package to PyPI, create a release on GitHub with a git tag of the form `vX.Y.Z`.
A GitHub Action will automatically build and push the package to PyPI.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
