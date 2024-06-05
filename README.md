# Version 1
In this initial version, we follow the basic patterns that you find the in NVIDIA
documentation for creating a Triton Inference deployment package. To keep things
simple, we leverage the Python backend. This requires having a `.py` file
that defines a `TritonPythonClass` where you must specify the `initialize()` and
`execute()` of the class. The `execute()` takes in a list of `InferenceRequests` that
must be processed and the returns a list of `InferenceResponses`. In addition, each
deployment requires a config.pbtxt file which specifies the inputs and outputs of the
deployment.

In most of the
[examples](https://github.com/triton-inference-server/python_backend/blob/main/examples/add_sub/model.py),
the InferenceRequests are interated through in the `execute()` and an InferenceResponse
is created and appended to a list that gets returned.

To make this all work, we will create three Triton Inference deployments. The first two
are straightforward model deployments:

  * fasttext-language-identification
  * SeamlessM4Tv2ForTextToText

A service level deployment, Translate, will be created leveraging NVIDIA's Business
Logic Scripting (BLS) to allow for logic branching. Specifically, we will use the
language identification model if the source language is not provided as an optional
parameter in the request. If the source language is provided, then this step can be
skipped and the text sent directly to the SeamlessM4T model for translation.

## FastText-Language-Identification
The first step in the process will be a language identification model. For this
workflow, we will use the [fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification). Because this model is very small and fast, we
specify that it is to be run on the CPU (see config.pbtxt file)

### Input
In the config.pbtxt file, we specify that the input text, INPUT_TEXT, will be of
datatype `TYPE_STRING`. Unfortunately, this doesn't mean what you think it does. When
requests are given, these will be bytes. This means that the `execute()` must decode
the input into strings before proceeding.

FastText requires that newlines be stripped, but this will be handled internally in the
`execute()` method.

### Output
Returns the source language id, taken from
[Wikipedia's list of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias)
which appears to be the convention that [FastText adopted](https://github.com/facebookresearch/fastText/issues/1305#issuecomment-1586349534).

Fortunately, since both this model and the SeamlessM4T are by Meta, the output of this
model seems compatible with SeamlessM4T after a small amount of cleaning, which is what
is returned by the model.

### Example Request
```
import json, requests
inference_request_fasttext = {
    "id": "id_0",
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
print(result.json()["outputs"][0]["data"][0])
# spa - for Spanish
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
treat these like bytes and decode them back into strings in `execute()`

In addition to the INPUT_TEXT, the Seamless-M4T model also takes in two additional
inputs, SRC_LANG and TGT_LANG. These specify the original language of the text and what
language to translate into. Both are treated as type TYPE_STRING as well.

### Output
Python string of the translated input text is returned.

### Example Request
```
import json, requests
inference_request_seamless = {
    "id": "id_1",
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
print(result.json()["outputs"][0]["data"][0])
# Today is my birthday.
```

## Translate
This is the service level Triton Inference deployment package that combines the language
identification and translation steps. This leverages the BLS (business logic scripting)
of Triton Inference Server. This allows us to skip the language identification step
if the requestor sends in an optional parameter, src_lang, in the request.

Again, to keep things simple in the beginning, we follow the 
[BLS synchronized example](https://github.com/triton-inference-server/python_backend/blob/main/examples/bls/sync_model.py). In this example, we are blocking on each request and
thus processing each request one at a time. We will work in later steps to do this
asynchronously in order to speed things up.

The goal for these service level Triton Inference deployment packages is to create
a client friendly interface where we can specify some good default behavior and handle
any specific quirkiness that needs to be done to before sending to the model. For
example, most clients probably just want to send an entire file to be translated. So,
we should make that be possible and handle getting the text portion out of the
document, splitting the document up into appropriate pieces to be translated, and then
combine the translated text pieces before sending back to the client.

However, for this simple tutorial we will stick to a straightfoward service to keep the
code cleaner and easier to see the important pieces.

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

### Example Requests
The first example does not provide either `src_lang` or `tgt_lang`. In this case, the
language identification step will set `src_lang` and `tgt_lang` is set to 'eng' for
English.

```
import json, requests
inference_request = {
    "id": "id_2",
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
print(result.json()["outputs"][0]["data"][0])
# Today is my birthday.
```

In this second example, we pass in the `src_lang` as Spanish since it is known ahead
of time.

```
inference_request_src_lang = {
    "id": "id_3",
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
print(result.json()["outputs"][0]["data"][0])
# Today is my birthday.
```

In this example, both `src_lang` is set as 'spa' and `tgt_lang` is set as 'fra' to
translate our Spanish sentence into French.

```
inference_request_src_tgt_lang = {
    "id": "id_4",
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
print(result.json()["outputs"][0]["data"][0])
# Aujourd'hui est mon anniversaire.
```

## Perf_Analyzer
We will leverage the performance analyzer that Triton Inference Server
provides.

### Starting SDK Container
Keep the Triton Inference Server container running and start up the SDK container.
This has the perf_analyzer command-line tool and all the needed dependencies. The
docs mention you can `pip install tritonclient` to get perf_analyzer, but warn you
that it won't install all the dependencies you may need. That was the case when we
tried to pip install on an Ubuntu machine.

```
$ docker pull nvcr.io/nvidia/tritonserver:24.04-py-sdk
$ docker run --rm -it --net host -v ./data:/workspace/data nvcr.io/nvidia/tritonserver:24.04-py3-sdk
```
This start up the container and mounts the `/data` directory which contains the
load test data that will be used.

### Test Data
Test data comes from sentences pulled from a Spanish News Classification dataset on
[Kaggle](https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification).
The data was split on '.' to get sentences and then threw out very short sentences
as a simple data cleaning step. These are stored in `/data/spanish-sentences.json`
in a form specified by the Perf_Analyzer documentation.

### Running the perf_analyzer
Inside the SDK container, run the following command:

```
perf_analyzer -m translate --input-data data/spanish-sentences.json --bls-composing-models fasttext-language-identification,seamless-m4t-v2-large --measurement-interval 20000
```

This launches the perf_analyzer
  * `-m` Analyze the provided deployed model
  * `--input-data` Use the specified JSON file for testing data
  * `--bls-composing-models` Comma separated list of underlying deployments
  * `--measurement-interval` Length of testing window in ms

### Results
On an RTX 4090, we get the following:

*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 20000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client: 
    Request count: 204
    Throughput: 2.83324 infer/sec
    Avg latency: 353702 usec (standard deviation 119449 usec)
    p50 latency: 329801 usec
    p90 latency: 516625 usec
    p95 latency: 596750 usec
    p99 latency: 653970 usec
    Avg HTTP time: 353691 usec (send/recv 39 usec + response wait 353652 usec)
  Server: 
    Inference count: 204
    Execution count: 204
    Successful request count: 204
    Avg request latency: 353265 usec (overhead 866 usec + queue 83 usec + compute 352316 usec)

  Composing models: 
  fasttext-language-identification, version: 1
      Inference count: 204
      Execution count: 204
      Successful request count: 204
      Avg request latency: 453 usec (overhead 2 usec + queue 41 usec + compute input 11 usec + compute infer 388 usec + compute output 10 usec)

  seamless-m4t-v2-large, version: 1
      Inference count: 204
      Execution count: 204
      Successful request count: 204
      Avg request latency: 351951 usec (overhead 3 usec + queue 42 usec + compute input 10 usec + compute infer 351869 usec + compute output 26 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 2.83324 infer/sec, latency 353702 usec
