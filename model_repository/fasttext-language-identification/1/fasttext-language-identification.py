import fasttext
from huggingface_hub import hf_hub_download
import json
import numpy as np
import re
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model uses fasttext to perform language identification"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get SRC_LANG configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "SRC_LANG")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        """Load the model into CPU RAM"""
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
            cache_dir="/hub",
            local_files_only=True,
        )
        self.model = fasttext.load_model(model_path)
        self.REMOVE_NEWLINE = re.compile(r"\n")

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
        responses = []
        for request in requests:
            # Get INPUT_TEXT, this is a Triton Tensor
            input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # Convert to Python str
            # Though config.pbtxt specifies datatype as TYPE_STRING when sending
            # through a request it is BYTES. Thus must be decoded.
            input_text = input_text_tt.as_numpy()[0].decode("utf-8")
            # Replace newlines with ' '. FastText breaks on \n
            input_text_cleaned = self.REMOVE_NEWLINE.sub(" ", input_text)

            # Run through the model
            output_labels, _ = self.model.predict(input_text_cleaned, k=1)
            # Take just the first one because we used k = 1 in predict()
            # Returns '__label__<lang_id>_<script>' but SeamlessM4T uses just lang_id
            src_lang = output_labels[0].replace("__label__", "").split("_")[0]

            # Make Triton Inference Response
            src_lang_tt = pb_utils.Tensor(
                "SRC_LANG",
                np.array([src_lang], dtype=self.output_dtype),
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[src_lang_tt],
            )
            responses.append(response)

        return responses
