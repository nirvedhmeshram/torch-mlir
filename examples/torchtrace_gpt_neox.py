# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from PIL import Image
import requests
import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
import sys
sys.path.append('/home/nirvedh/gpt-neox/') #path to gpt models
import megatron.model.gpt2_model as gpt2model
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron

from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion

from megatron import print_rank_0
from megatron import mpu
from deepspeed import PipelineEngine, DeepSpeedEngine
from collections import deque


from transformers import BertTokenizer, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import transformers

mb = ModuleBuilder()

def get_neox_args():
    use_cache=True
    overwrite_values=None
    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    print("Here 0.1")
    if overwrite_values:
        _overwrite_values.update(overwrite_values)
    print("Here 0.2")
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=_overwrite_values)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()
    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    print("Here 1")
    initialize_megatron(neox_args)
    print("here 2")
    return neox_args

class Gpt2Module(torch.nn.Module):
    print("Here 0")

    def __init__(self,neox_args):
        super().__init__()
        self.gpt = gpt2model.GPT2ModelPipe(neox_args,topology=mpu.get_topology())

    def forward(self, input):
        return self.gpt(input)


class TestModule(torch.nn.Module):
    def __init__(self,neox_args):
        super().__init__()
        self.s = Gpt2Module(neox_args)

    def forward(self, x):
        return self.s.forward(x)


neox_args = get_neox_args()
test_module = TestModule(neox_args)


gpt = GPT2Model.from_pretrained("gpt2").cuda().half()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
test_text = (
    "Hello. How are you? I am fine thank you and you? yes Good. "
    "hi hi hi hi hi hi hi"  # 24
)

tokens = tokenizer(
    [test_text] * 4,
    return_tensors="pt",
)

attention_mask = tokens["attention_mask"].cuda()
attention_mask = attention_mask.view(attention_mask.size(0), -1)
attention_mask = attention_mask[:, None, None, :]
attention_mask = (1.0 - attention_mask) * -10000.0
attention_mask = attention_mask.repeat(1, 1, attention_mask.size()[-1], 1)
input_ids=tokens["input_ids"].cuda()
position_ids=torch.tensor([])

class_annotator = ClassAnnotator()
input_token=torch.tensor([4])
recursivescriptmodule = torch.jit.trace_module(test_module,{"forward":[input_ids,position_ids,attention_mask]})
torch.jit.save(recursivescriptmodule, "/tmp/foo.pt")

class_annotator.exportNone(recursivescriptmodule._c._type())
class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
class_annotator.annotateArgs(
    recursivescriptmodule._c._type(),
    ["forward"],
    [
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ],
    )
    # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator)

backend = refbackend.RefBackendLinalgOnTensorsBackend()
with mb.module.context:
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
    pm.run(mb.module)

compiled = backend.compile(mb.module)
jit_module = backend.load(compiled)

