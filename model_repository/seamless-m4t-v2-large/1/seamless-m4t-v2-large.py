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
        responses = [None] * len(requests)
        batch_text = []
        batch_src_lang = []
        batch_tgt_lang = []
        batch2req = {}  # Maps batch id to corresponding request id
        for r_id, request in enumerate(requests):
            # Get the input data as Triton Tensors
            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                src_lang_tt = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
                tgt_lang_tt = pb_utils.get_input_tensor_by_name(request, "TGT_LANG")
            except Exception as exc:
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[r_id] = error_response
                continue

            # Convert TritonTensor -> numpy -> python str
            # NOTE: Triton converts your input string to bytes so you need to decode
            input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            src_lang = src_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            tgt_lang = tgt_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")

            batch2req[len(batch_text)] = r_id
            batch_text.append(input_text)
            batch_src_lang.append(src_lang)
            batch_tgt_lang.append(tgt_lang)

        # NOTE: For tutorial, we are going to not worry about the fact that SeamlessM4T
        # can take a text:List[str], but it ONLY takes src_lang:str & tgt_lang:str
        # This means you can't have a batch with mixed languages on input or translate
        # to different languages. But it seems like this is a good pattern to follow
        # so we will just assume (which our data is) all the same languages.
        src_lang = batch_src_lang[0]
        tgt_lang = batch_tgt_lang[0]
        # Run through the model for translation all at once
        # Tokenize
        try:
            input_ids = self.processor(
                text=batch_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                return_tensors="pt",
            ).to(self.device)
        except Exception as exc:
            for b_id, input_text in enumerate(batch_text):
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"seamless.processor had an error with the batch. You sent "
                        + f"{input_text}. Check that it is not the problem."
                    )
                )
                responses[batch2req[b_id]] = error_response
            return responses

        # Generate output tokens
        try:
            output_tokens = self.model.generate(
                **input_ids,
                tgt_lang=tgt_lang,
                num_beams=5,
                num_return_sequences=1,
                max_new_tokens=3000,
                no_repeat_ngram_size=3,
            )
        except Exception as exc:
            for b_id, input_text in enumerate(batch_text):
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"seamless.generate had an error in the batch. You sent "
                        + f"{input_text}. Check that it is not the problem"
                    )
                )
                responses[batch2req[b_id]] = error_response
            return responses

        # Decode tokens to text
        try:
            translated_batch_text = self.processor.batch_decode(
                output_tokens, skip_special_tokens=True
            )
        except Exception as exc:
            for b_id, input_text in enumerate(batch_text):
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "seamless.batch_decode had an erro in the batch. You sent "
                        + f"{input_text}. Check that it is not the problem"
                    )
                )
                responses[batch2req[b_id]] = error_response
            return responses

        for b_id, translated_text in enumerate(translated_batch_text):
            # Convert to TritonTensor & make the TritonInferenceResponse
            translated_text_tt = pb_utils.Tensor(
                "TRANSLATED_TEXT",
                np.array([translated_text], dtype=self.output_dtype).reshape(-1, 1),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_text_tt],
            )
            responses[batch2req[b_id]] = inference_response

        return responses
