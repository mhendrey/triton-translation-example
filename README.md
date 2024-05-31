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

## Launch Triton Inference Server
The Triton container is launched by mapping two different volumes. The first
is the model_repository of this github repo to /models in the container. This
matches the --model-repository option so that tritonserver knows where to find
your model information. The second maps the local disk cache of huggingface's hub to
/hub in the container. The different TritonPythonClass.initialize are use the
cache_dir=/hub to load the appropriate model.
```
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models -v /home/matthew/.cache/huggingface/hub:/hub nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3 tritonserver --model-repository=/models
```

## Fasttext-Language-Identification
The first step in the process will be a language identification model. For this
workflow, I will use the
[fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification). 

### Input
In the config.pbtxt file, we specify that the input text, INPUT_TEXT, will be of
datatype `TYPE_STRING`. Unfortunately, this doesn't mean what you think it does. When
requests are given, these will be bytes. This means that the
TritonPythonClass.execute() must decode the input into strings before proceeding.

FastText requires that newlines be stripped, but this will be handled internally in the
`execute()` method.

### Output
Returns the source language id, taken from
[Wikipedia's list of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias)
which appears to be the convention that [FastText adopted](https://github.com/facebookresearch/fastText/issues/1305#issuecomment-1586349534).

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
For translation, we use the SeamlessM4Tv2ForTextToText (large) model in Transformers.
By specifying that we just want the ForTextToText, we save some GPU RAM by not also
loading in the audio model. To further save space, the model is loaded with torch_dtype
of float16 instead of using the default float32. This saves about 1/2 the GPU RAM as
well. This model will take in the text to be translated, what language the source text
is, and finally what language do we want the text translated into.

NOTE: Transformers only takes in src_lang:str and tgt_lang:str. This seems to imply
that if you want to do batch processing that a batch must be all the same src_lang
and all must have the same tgt_lang too. This seems highly restrictive and requires
further investigation.

### Input
Similar to the fasttext-language-identification model, the input text has a datatype of
TYPE_STRING in the config.pbtxt file.  Again, despite the name, incoming requests must
treat these like bytes and decode them back into strings in TritonPythonClass.execute()

In addition to the INPUT_TEXT, the Seamless-M4T model also takes in two additional
inputs, SRC_LANG and TGT_LANG. These specify the original language of the text and what
language to translate into. Both are treated as type TYPE_STRING as well.

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
        },
        {
            "name": "TGT_LANG",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["eng"],
        }
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/seamless-m4t-v2-large/infer",
    json=inference_request_seamless,
)
print(result.json())
```

## Translate
This is the service level Triton deployment package that combines the language
identification and translation steps. This leverages the BLS (business logic scripting)
of Triton Inference Server. This allows us to skip the language identification step
if the requestor sends in an optional parameter, src_lang, in the request.

To start off, this is nearly a copy of the [BLS synchronized example](https://github.com/triton-inference-server/python_backend/blob/main/examples/bls/sync_model.py).

### Input
The input text has a datatype of TYPE_STRING which is a natural way for a client
to want to send the data to the translation service. To make this work, the input
name to this deployment package needed to match the name of the inputs to both
the FastText-Language-Identification and the Seamless-M4T-v2-Large.

### Output
The output text is also of datatype TYPE_STRING.

### Optional Request Parameters
The Translate service accepts two optional parameters to be passed in a request.

* src_lang - If provided, the language identification step is skipped
* tgt_lang - Specifies the language to translate into. If not provided 'eng' is used

See the examples below for how to use these.

### Conda Environment
Since this is a higher level of abstraction, there is no need for a conda package just
yet. It's possible when moving to the async that a conda environment will be needed.

### Example Request
The first example does not provide either `src_lang` or `tgt_lang`. In this case, the
language identification step will set `src_lang` and `tgt_lang` is set to 'eng' for
English.

```
import json, requests
inference_request = {
    "id": "abc",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        }
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/translate/infer",
    json=inference_request,
)
print(result.json())
```

In this second example, we pass in the `src_lang` as Spanish since it is known ahead
of time.

```
inference_request_src_lang = {
    "id": "abc",
    "parameters": {"src_lang": "spa"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        }
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/translate/infer",
    json=inference_request_src_lang,
)
print(result.json())
```
In this example, both `src_lang` is set as 'spa' and `tgt_lang` is set as 'fra' to
translate our Spanish sentence into French.

```
inference_request_src_tgt_lang = {
    "id": "abc",
    "parameters": {"src_lang": "spa", "tgt_lang": "fra"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        }
    ],
}
result = requests.post(
    url="http://localhost:8000/v2/models/translate/infer",
    json=inference_request_src_tgt_lang,
)
print(result.json())
```
