import json
import numpy as np
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Service Level Deployment Package
    This handles things nicely for clients. Taking in 'strings' [really means bytes]
    and then handles the logic for using the language id model (if not specified)
    before passing on to translation model."""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get TRANSLATED_TEXT configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

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
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT_TEXT
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")

            # Get any optional parameters passed in.
            request_params = json.loads(request.parameters())
            src_lang = request_params.get("src_lang", None)
            tgt_lang = request_params.get("tgt_lang", "eng")
            tgt_lang = pb_utils.Tensor("TGT_LANG", np.array([tgt_lang], np.object_))

            # If the lang_id isn't passed in, then run language id model to set it
            if not src_lang:
                # Create inference request object
                infer_lang_id_request = pb_utils.InferenceRequest(
                    model_name="fasttext-language-identification",
                    requested_output_names=["SRC_LANG"],
                    inputs=[input_text],
                )

                # Peform synchronous blocking inference request
                infer_lang_id_response = infer_lang_id_request.exec()
                if infer_lang_id_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_lang_id_response.error().message()
                    )

                # Get the lang_id
                src_lang = pb_utils.get_output_tensor_by_name(
                    infer_lang_id_response, "SRC_LANG"
                )
            else:
                src_lang = pb_utils.Tensor("SRC_LANG", np.array([src_lang], np.object_))

            # Create inference request object for translation
            infer_seamless_request = pb_utils.InferenceRequest(
                model_name="seamless-m4t-v2-large",
                requested_output_names=["TRANSLATED_TEXT"],
                inputs=[input_text, src_lang, tgt_lang],
            )

            # Perform synchronous blocking inference request
            infer_seamless_response = infer_seamless_request.exec()
            if infer_seamless_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_seamless_response.error().message()
                )
            # Create the response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=infer_seamless_response.output_tensors()
            )
            responses.append(inference_response)
        return responses
