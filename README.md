# Version 2: Dynamic Batching
With a simple, perhaps even naive, approach implemented in v1, its safe to say that the infers/sec metric doesn't really blow you away. So let's continue on and see if we can improve the performance using Triton Inference Server's dynamic batching capability.

## Enabling Dynamic Batching
Let's start by simple enabling dynamic batching. To keep in the spirit of this
tutorial, let's start with the simplest approach which simple turns on dynamic
batching. To do this, we need to edit the config.pbtxt files to add the following
line to each of the three config.pbtxt files in the group at the top of the file.
This specifies the maximum batch size to accumulate before passing it to the models.

```
max_batch_size: 16
```
At the bottom of each of the files, add this line to enable the default dynamic
batching.

```
dynamic_batching {}
```

Now restart the service by running the `docker-compose up` command and let's send the following response from a python interpretter like we did before.

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
But this is throwing an error. So let's see that error by `$ result.json()` and
see it say

```{'error': "[request id: id_2] unexpected shape for input 'INPUT_TEXT' for model 'translate'. Expected [-1,1], got [1]. NOTE: Setting a non-zero max_batch_size in the model config requires a batch dimension to be prepended to each input shape. If you want to specify the full shape including the batch dim in your input dims config, try setting max_batch_size to zero. See the model configuration docs for more info on max_batch_size."}```

It turns out that dynamic batching causes the inputs shapes to change by prepending a
batching dimension to the input. In this case, each request is a batch of 1 and instead
of getting a request like ["Today is my birthday"] the input is
[["Today is my birthday"]].  This means that we need to make two changes to the code.

### Modifications
It seems we have a little work to do to make dynamic batching functional. We will need
to make two changes. The first is modifications in the code inside each of the
`execute()` functions. The second is to how requests are sent to the endpoints.

### Code Modications
At issue is that the Triton Tensor inputs will be of shape [1, 1] because of the
dynamic batching (even if we don't change the config.pbtxt `dims` fields) with the
first 1 being the size of the batch. This can easily be resolved by calling numpy's
`.reshape(-1)` once we convert the input Triton Tensors to numpy arrays. Once that is
done, the rest of the code will function the same as before.

For the fasttext-language-identification.py file we make just these two changes
```
-  input_text = input_text_tt.as_numpy()[0].decode("utf-8")
+  input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")

   src_lang_tt = pb_utils.Tensor(
      "SRC_LANG",
-     np.array([src_lang], dtype=self.output_dtype),
+     np.array([src_lang], dtype=self.output_dtype).reshape(-1, 1),
   )
```

Similarly, in seamless-m4t-v2-large.py we change
```
-  input_text = input_text_tt.as_numpy()[0].decode("utf-8")
-  src_lang = src_lang_tt.as_numpy()[0].decode("utf-8")
-  tgt_lang = tgt_lang_tt.as_numpy()[0].decode("utf-8")
+  input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
+  src_lang = src_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")
+  gt_lang = tgt_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")
 
   translated_text_tt = pb_utils.Tensor(
       "TRANSLATED_TEXT",
-      np.array([translated_text], dtype=self.output_dtype),
+      np.array([translated_text], dtype=self.output_dtype).reshape(-1, 1),
   )
```

Lastly, in translate.py we change:
```
-  tgt_lang_tt = pb_utils.Tensor("TGT_LANG", np.array([tgt_lang], np.object_))
+  tgt_lang_tt = pb_utils.Tensor(
+      "TGT_LANG", np.array([tgt_lang], np.object_).reshape(-1, 1)
+  )
 
   src_lang_tt = pb_utils.Tensor(
-      "SRC_LANG", np.array([src_lang], np.object_)
+      "SRC_LANG", np.array([src_lang], np.object_).reshape(-1, 1)
   )
```

With the code changes made, let's stop the Triton Inference Server and restart it by
calling the `docker-compose up` command from the parent directory.

### Request Modifications
The second change needed is in the `shape` field of a request sent to any of the
inference endpoints that have dynamic batching enabled (all of them in this example).
Previously the request had `shape: [1]` in the JSON, but now it must also have the
batch size listed. With a batch size of just 1, it becomes `shape: [1, 1]`.

#### FastText Example
For the fasttext-language-id, we now submit a request like so:

```
import json, requests
inference_request_fasttext = {
    "id": "id_0",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],  # Previously we just had [1]
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
#### SeamlessM4T Example
The seamless-m4t-v2-large request now looks like:

```
import json, requests
inference_request_seamless = {
    "id": "id_1",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1], # Previously we just had [1]
            "datatype": "BYTES",
            "data": ["Hoy es mi cumpleaños."],
        },
        {
            "name": "SRC_LANG",
            "shape": [1, 1], # Previously we just had [1]
            "datatype": "BYTES",
            "data": ["spa"],
        },
        {
            "name": "TGT_LANG",
            "shape": [1, 1], # Previously we just had [1]
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
#### Translate Examples
And finally, using the service level deployment, translate:

```
import json, requests
inference_request = {
    "id": "id_2",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1], # Previously we had just [1]
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

inference_request_src_lang = {
    "id": "id_3",
    "parameters": {"src_lang": "spa"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1], # Previously we had just [1]
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

inference_request_src_tgt_lang = {
    "id": "id_4",
    "parameters": {"src_lang": "spa", "tgt_lang": "fra"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1], # Previously we had just [1]
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
With the changes made, let's see how well we do with the perf_analyzer. Switch over to
the window where the SDK container is running and run the following command again.

```
perf_analyzer -m translate --input-data data/spanish-sentences.json --bls-composing-models fasttext-language-identification,seamless-m4t-v2-large --measurement-interval 20000
```
While that's running, remember that previously we got about 2.83 infer/sec.

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
    Request count: 199
    Throughput: 2.76382 infer/sec
    Avg latency: 362294 usec (standard deviation 121656 usec)
    p50 latency: 339668 usec
    p90 latency: 523333 usec
    p95 latency: 600798 usec
    p99 latency: 667485 usec
    Avg HTTP time: 362283 usec (send/recv 40 usec + response wait 362243 usec)
  Server: 
    Inference count: 199
    Execution count: 199
    Successful request count: 199
    Avg request latency: 361844 usec (overhead 891 usec + queue 164 usec + compute 360789 usec)

  Composing models: 
  fasttext-language-identification, version: 1
      Inference count: 199
      Execution count: 199
      Successful request count: 199
      Avg request latency: 559 usec (overhead 3 usec + queue 87 usec + compute input 14 usec + compute infer 441 usec + compute output 12 usec)

  seamless-m4t-v2-large, version: 1
      Inference count: 199
      Execution count: 199
      Successful request count: 199
      Avg request latency: 360401 usec (overhead 4 usec + queue 77 usec + compute input 11 usec + compute infer 360280 usec + compute output 27 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 2.76382 infer/sec, latency 362294 usec

WHAT!? That's not any faster. Where do you think we could have gone wrong?

### Next Steps
It turns out we didn't really accomplish much with this version of the code. Though we
have dynamic batching enabled, it turns out that all we have really changed is that
instead of calling `execute([request_1]); execute([request_2])` that we
are now doing `execute([request_1, request_2])` as a result of dynamic batching.
However, inside the `execute()` we are still just for-looping through the requests
and calling the model each time on a batch of size 1.  To really take advantage of
dynamic batching we need to gather up the inputs of the requests before calling the
model on them. 

This [Github issue](https://github.com/triton-inference-server/server/issues/5926#issuecomment-1585161393) has a good explanation and some pseudo code that we will
follow. Note that link shows a slightly more complicated version that we will get to
which handles the case when incoming requests could have different batch sizes. But
again, to keep things simple, let's assume that all incoming requests have a batch
size 1.

When we are finished our code will look something like this:

```
def initialize(self, args):
    # ...
    self.model = MyModel()

def execute(self, requests):
    batch = []
    for request in requests:
        np_input = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
        batch.append(np_input)
    
    # Gather into a single batch
    batched_input = np.vstack(batch)
    # Compute the full batch in one call
    batched_output = self.model.infer(batch_input)
    
    # Python backend must return a response for each request received
    responses = []
    for output in batched_output:
        # Make your Triton Tensor from the output
        output_tt = pb_utils.Tensor(...)
        response = pb_utils.InferenceResponse(
                output_tensors=[output_tt],
        )
        responses.append(response)
    return responses
```