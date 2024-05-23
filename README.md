# triton-translation-example
Exploring how to utilize NVIDIA's Triton Inference Server for hosting machine translation workflow.

## Creating Your Own Conda Environment
Taking directly from the [NVIDIA documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html?highlight=conda#creating-custom-execution-environments).
This highlights the importance of setting `export PYTHONNOUSERSITE=True` before calling
conda-pack

## Fasttext-Language-Identification
The first step in the process be a language identification model. For this workflow,
I will use the [fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification). 

### Input
Input will be Python strings. FastText requires that newlines be stripped, but this
will be handled internally.

### Output
Returns the language id, taken from [Wikipedia's list of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias) which appears to be the convention that [FastText adopted](https://github.com/facebookresearch/fastText/issues/1305#issuecomment-1586349534).

### Conda Environment
Within the model_repository/fasttext-language-identification directory do the following
```
$ conda env create -f environment.yml
$ conda-pack -n fasttext-language-identification -o fasttext-language-identification.tar.gz
```