import json
import numpy as np

# from transformers import SeamlessM4TForTextToText
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Perform translation using SeamlessM4T-large-v2's Text2Text"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # self.model = SeamlessM4TForTextToText.from_pretrained("", auto_map=True)
        # DEGUB - Returns input
        self.model = lambda x: x
        """
        self.model = SeamlessM4Tv2ForTextToText.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir="/hub")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        """

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """_summary_

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            _description_

        Returns
        -------
        List[pb_utils.InferenceResponse]
            _description_
        """
        output_dtype = self.output_dtype

        # Create a batch to give to the model
        # Likely not needed here, but certainly want it when on a GPU
        batch = []
        for request in requests:
            # Get INPUT
            input = pb_utils.get_input_tensor_by_name(request, "INPUT")
            # Convert TritonTensor -> numpy -> python list for fasttext
            input_str = input.as_numpy().tolist()[0]
            batch.append(input_str)

        # Batch inference
        """
        text_inputs = self.processor(
            text=batch,
            src_lang="need_this_input", # Seems to need to be all the same language
            return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(
            **text_inputs,
            tgt_lang="en",
            num_beams=5,
            num_return_sequences=1,
            max_new_tokens=3000,
            no_repeat_ngram_size=3,
        )
        output_text = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
        """
        # outputs = self.model.generate(inputs)
        outputs = batch

        responses = []
        for output, request in zip(outputs, requests):
            output = pb_utils.Tensor(
                "OUTPUT",
                np.array([output], dtype=output_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output],
                # id=request.request_id(), # Not sure you should do this
                # Found in https://github.com/triton-inference-server/server/issues/6181
                # Maybe it is just RequestID() which is the function in infer_request.cc
            )
            responses.append(inference_response)

        return responses
