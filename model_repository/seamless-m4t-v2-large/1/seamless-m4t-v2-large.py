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

        # Get TRANSLATED_TEXT configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests: List) -> List:
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
        tgt_langs = []
        for request in requests:
            # Get INPUT
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # Convert TritonTensor -> numpy -> python list for input to SeamlessM4T
            input_text = input_text.as_numpy().tolist()[0].decode("utf-8")
            input_texts.append(input_text)
            # Convert TritonTensor -> numpy -> btyes -> str
            src_lang = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
            src_lang = src_lang.as_numpy().tolist()[0].decode("utf-8")
            src_langs.append(src_lang)

            tgt_lang = pb_utils.get_input_tensor_by_name(request, "TGT_LANG")
            tgt_lang = tgt_lang.as_numpy().tolist()[0].decode("utf-8")
            tgt_langs.append(tgt_lang)

        # Batch inference
        inputs_ids = self.processor(
            text=input_texts,
            src_lang=src_langs[0],  # For now need to use all the same src lang
            tgt_lang=tgt_langs[0],  # For now need to use all the same tgt lang
            return_tensors="pt",
        ).to(self.device)
        output_tokens = self.model.generate(
            **inputs_ids,
            tgt_lang=tgt_langs[0],
            num_beams=5,
            num_return_sequences=1,
            max_new_tokens=3000,
            no_repeat_ngram_size=3,
        )
        outputs = self.processor.batch_decode(output_tokens, skip_special_tokens=True)

        responses = []
        for output in outputs:
            output = pb_utils.Tensor(
                "TRANSLATED_TEXT",
                np.array([output], dtype=output_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output],
            )
            responses.append(inference_response)

        return responses
