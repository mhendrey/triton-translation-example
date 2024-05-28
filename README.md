# triton-translation-example
Exploring how to utilize NVIDIA's Triton Inference Server for hosting machine translation workflow.

### Pulling Triton Inference Server Container
Let's use the latest version of the docker container that has the Python & PyTorch backends
`$ docker pull nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3`

## Creating Your Own Conda Environment
Taking directly from the [NVIDIA documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html?highlight=conda#creating-custom-execution-environments).
This highlights the importance of setting `export PYTHONNOUSERSITE=True` before calling
conda-pack. In addition, the docs mention that the python version in the conda-pack
must match the python version in the container.  You can check this by first pulling
down the container of interest and then going into the container to check python
```
$ docker pull nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3
$ docker run -it nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3 /bin/bash
container:/opt/tritonserver# /usr/bin/python3 -V
container:/opt/tritonserver# exit
```

To create the conda pack needed for an environment:
```
$ conda env create -f environment.yml
$ conda pack -n <environment_name>
```

## Fasttext-Language-Identification
The first step in the process be a language identification model. For this workflow,
I will use the [fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification). 

### Input
In the config.pbtxt file, we specify that the input text, INPUT_TEXT, will be of
datatype `TYPE_STRING`. Unfortunately, this doesn't mean what you think it does. When
requests are given, these will be bytes. This means that the
TritonPythonClass.execute() must decode the input into strings before proceeding.

FastText requires that newlines be stripped, but this will be handled internally in the
`execute()` method.

### Output
Returns the language id, taken from [Wikipedia's list of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias) which appears to be the convention that [FastText adopted](https://github.com/facebookresearch/fastText/issues/1305#issuecomment-1586349534).

### Conda Environment
Within the model_repository/fasttext-language-identification directory do the following
```
$ conda env create -f environment.yml
$ conda-pack -n fasttext-language-identification
```

### Example Request
```
import json, requests
inference_request_fasttext = {
    "id": "abc",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        },
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/fasttext-language-identification/infer",
    json=inference_request_fasttext,
)
print(result.json())
```

## Seamless-M4T-v2-Large
Once we know the source language from fasttext, we will tranlsate the input text from
the source language into English. This leverages the SeamlessM4Tv2ForTextToText (large)
model in Tranformers.

### Input
Similar to the fasttext-language-identification model, the input text has a datatype of
TYPE_STRING in the config.pbtxt file.  Again, despite the name, incoming requests must
treat these like bytes and decode them back into strings.

In addition to the INPUT_TEXT, the Seamless-M4T model also takes a second input called
LANG_ID. This is the source language ID that must be provided when performing the
translation.

### Output
Python string of the translated input text is returned.

### Conda Environment
Within the model_repository/seamless-m4t-v2-large directory do the following
```
$ conda env create -f environment.yml
$ conda-pack -n seamless-m4t-v2-large
```

### Example Request
```
import json, requests
inference_request_seamless = {
    "id": "abc",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        },
        {
            "name": "SRC_LANG",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["spa"],
        }
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/seamless-m4t-v2-large/infer",
    json=inference_request_seamless,
)
print(result.json())
```
