import json
import numpy as np

import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
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

        self.model = SeamlessM4Tv2ForTextToText.from_pretrained(
            "facebook/seamless-m4t-v2-large",
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir="/hub",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            "facebook/seamless-m4t-v2-large",
            cache_dir="/hub",
            local_files_only=True,
        )

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
        input_texts = []
        src_langs = []
        for request in requests:
            # Get INPUT
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # Convert TritonTensor -> numpy -> python list for fasttext
            input_text = input_text.as_numpy().tolist()[0]
            input_texts.append(input_text)
            src_lang = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
            src_lang = src_lang.as_numpy().tolist()[0]
            src_langs.append(input_text)

        # Batch inference
        inputs_ids = self.processor(
            text=input_texts,
            src_lang=src_langs[0],  # For now need to use all the same input language
            tgt_lang="en",
            return_tensors="pt",
        ).to(self.device)
        output_tokens = self.model.generate(
            **inputs_ids,
            tgt_lang="en",
            num_beams=5,
            num_return_sequences=1,
            max_new_tokens=3000,
            no_repeat_ngram_size=3,
        )
        outputs = self.processor.batch_decode(output_tokens, skip_special_tokens=True)

        responses = []
        for output in outputs:
            output = pb_utils.Tensor(
                "OUTPUT",
                np.array([output], dtype=output_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output],
            )
            responses.append(inference_response)

        return responses
