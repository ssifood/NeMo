# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

from argparse import ArgumentParser

import torch

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.common.metrics.classification_accuracy import TopKClassificationAccuracy, compute_topk_accuracy
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="MatchboxNet-3x1x64-vad", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=1)

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.ckpt'):
        logging.info(f"Using local checkpoint from {args.asr_model}")
        asr_model = EncDecClassificationModel.load_from_checkpoint(args.asr_model)

    elif args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        #  TODO
        asr_model = EncDecClassificationModel.restore_from(restore_path=args.asr_model)

    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        #  TODO
        asr_model = EncDecClassificationModel.from_pretrained(model_name=args.asr_model)

    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model._cfg.labels,
            'batch_size': args.batch_size,
        }
    )
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()

    top_k = [args.topk]
    acc = TopKClassificationAccuracy(top_k)
    correct_counts = torch.tensor([0.0])
    total_counts = torch.tensor([0.0])

    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs = asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            correct_counts_batch, total_counts_batch = acc(logits=log_probs, labels=test_batch[2])

            correct_counts = torch.cat([correct_counts, correct_counts_batch], dim=0)
            total_counts = torch.cat([total_counts, total_counts_batch], dim=0)

        del test_batch

    final_acc = compute_topk_accuracy(correct_counts.unsqueeze(dim=-1), total_counts.unsqueeze(dim=-1))
    logging.info(f'Got classification accuracy for top-{args.topk} of {final_acc}.')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
