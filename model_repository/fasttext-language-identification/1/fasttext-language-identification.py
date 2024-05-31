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
        output_dtype = self.output_dtype

        # Create a batch to give to the model
        # Likely not needed here, but certainly want it when on a GPU
        batch = []
        for request in requests:
            # Get INPUT
            input = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            # Convert TritonTensor -> numpy -> python list for fasttext
            # Eventhough config.pbtxt specifies datatype: TYPE_STRING
            # when sending through a request it is BYTES. So it must be decoded
            input_str = input.as_numpy().tolist()[0].decode("utf-8")
            # Replace newlines with ' '. FastText breaks on \n
            batch.append(self.REMOVE_NEWLINE.sub(" ", input_str))

        # Batch inference
        output_labels, _ = self.model.predict(batch, k=1)

        responses = []
        for output_label, request in zip(output_labels, requests):
            # Clean up the label. It has '__label__<lang_id>_<script>'
            # Take just the first one because we used k=1 in predict()
            output_label, _ = output_label[0].replace("__label__", "").split("_")
            output = pb_utils.Tensor(
                "SRC_LANG",
                np.array([output_label], dtype=output_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output],
            )
            responses.append(inference_response)

        return responses
