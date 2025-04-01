# Code in this file is modified from the following link:
# https://github.com/hitachinsk/SAMed/blob/main/sam_lora_image_encoder_mask_decoder.py

# MIT License

# Copyright (c) 2023 Kaidong Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        qkv = self.lora_dropout(qkv)
        return qkv


class _LoRA_qkv_proj(nn.Module):
    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module, lora_dropout: float = 0.0):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x):
        x = self.proj(x) + self.w_b(self.w_a(x))
        x = self.lora_dropout(x)
        return x


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    self.args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, config, lora_layer=None, zero_initial=False):
        super(LoRA_Sam, self).__init__()

        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        self.config = config
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        if self.config["freeze_image_encoder"]:
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False

        if self.config["freeze_prompt_encoder"]:
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False

        if self.config["freeze_mask_decoder_transformer"]:
            for param in sam_model.mask_decoder.transformer.parameters():
                param.requires_grad = False

        if self.config["freeze_output_hypernetworks_mlps"]:
            for param in sam_model.mask_decoder.output_hypernetworks_mlps.parameters():
                param.requires_grad = False

        if self.config["freeze_upscaling_cnn"]:
            for param in sam_model.mask_decoder.output_upscaling.parameters():
                param.requires_grad = False

        if self.config["freeze_mask_decoder_mask_tokens"]:
            for param in sam_model.mask_decoder.mask_tokens.parameters():
                param.requires_grad = False

        if self.config["freeze_mask_decoder_iou"]:
            for param in sam_model.mask_decoder.iou_token.parameters():
                param.requires_grad = False
            for param in sam_model.mask_decoder.iou_prediction_head.parameters():
                param.requires_grad = False

        er = self.config["image_encoder_lora_rank"]
        if er > 0:
            # Here, we do the surgery
            for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, er, bias=False)
                w_b_linear_q = nn.Linear(er, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, er, bias=False)
                w_b_linear_v = nn.Linear(er, self.dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    lora_dropout=self.config["lora_dropout"],
                )

        # Additional surgery for the mask decoder
        self.self_attn_As = []
        self.self_attn_Bs = []
        self.cross_attn_ti_As = []
        self.cross_attn_ti_Bs = []
        self.cross_attn_it_As = []
        self.cross_attn_it_Bs = []

        dr = self.config["mask_decoder_lora_rank"]

        if dr > 0:
            decoder_transformer = sam_model.mask_decoder.transformer
            for layer_idx, blk in enumerate(decoder_transformer.layers):
                self_attn_q_proj = blk.self_attn.q_proj
                self_attn_v_proj = blk.self_attn.v_proj
                input_dim = blk.self_attn.embedding_dim
                output_dim = blk.self_attn.internal_dim
                w_a_linear_q_self_attn = nn.Linear(input_dim, dr, bias=False)
                w_b_linear_q_self_attn = nn.Linear(dr, output_dim, bias=False)
                w_a_linear_v_self_attn = nn.Linear(input_dim, dr, bias=False)
                w_b_linear_v_self_attn = nn.Linear(dr, output_dim, bias=False)
                self.self_attn_As.append(w_a_linear_q_self_attn)
                self.self_attn_Bs.append(w_b_linear_q_self_attn)
                self.self_attn_As.append(w_a_linear_v_self_attn)
                self.self_attn_Bs.append(w_b_linear_v_self_attn)
                blk.self_attn.q_proj = _LoRA_qkv_proj(
                    self_attn_q_proj,
                    w_a_linear_q_self_attn,
                    w_b_linear_q_self_attn,
                    lora_dropout=self.config["lora_dropout"],
                )
                blk.self_attn.v_proj = _LoRA_qkv_proj(
                    self_attn_v_proj,
                    w_a_linear_v_self_attn,
                    w_b_linear_v_self_attn,
                    lora_dropout=self.config["lora_dropout"],
                )

                cross_attn_ti_q_proj = blk.cross_attn_token_to_image.q_proj
                cross_attn_ti_v_proj = blk.cross_attn_token_to_image.v_proj
                ti_input_dim = blk.cross_attn_token_to_image.embedding_dim
                ti_output_dim = blk.cross_attn_token_to_image.internal_dim
                w_a_linear_q_cross_attn_ti = nn.Linear(ti_input_dim, dr, bias=False)
                w_b_linear_q_cross_attn_ti = nn.Linear(dr, ti_output_dim, bias=False)
                w_a_linear_v_cross_attn_ti = nn.Linear(ti_input_dim, dr, bias=False)
                w_b_linear_v_cross_attn_ti = nn.Linear(dr, ti_output_dim, bias=False)
                self.cross_attn_ti_As.append(w_a_linear_q_cross_attn_ti)
                self.cross_attn_ti_Bs.append(w_b_linear_q_cross_attn_ti)
                self.cross_attn_ti_As.append(w_a_linear_v_cross_attn_ti)
                self.cross_attn_ti_Bs.append(w_b_linear_v_cross_attn_ti)
                blk.cross_attn_token_to_image.q_proj = _LoRA_qkv_proj(
                    cross_attn_ti_q_proj,
                    w_a_linear_q_cross_attn_ti,
                    w_b_linear_q_cross_attn_ti,
                    lora_dropout=self.config["lora_dropout"],
                )
                blk.cross_attn_token_to_image.v_proj = _LoRA_qkv_proj(
                    cross_attn_ti_v_proj,
                    w_a_linear_v_cross_attn_ti,
                    w_b_linear_v_cross_attn_ti,
                    lora_dropout=self.config["lora_dropout"],
                )

                cross_attn_it_q_proj = blk.cross_attn_image_to_token.q_proj
                cross_attn_it_v_proj = blk.cross_attn_image_to_token.v_proj
                it_input_dim = blk.cross_attn_image_to_token.embedding_dim
                it_output_dim = blk.cross_attn_image_to_token.internal_dim
                w_a_linear_q_cross_attn_it = nn.Linear(it_input_dim, dr, bias=False)
                w_b_linear_q_cross_attn_it = nn.Linear(dr, it_output_dim, bias=False)
                w_a_linear_v_cross_attn_it = nn.Linear(it_input_dim, dr, bias=False)
                w_b_linear_v_cross_attn_it = nn.Linear(dr, it_output_dim, bias=False)
                self.cross_attn_it_As.append(w_a_linear_q_cross_attn_it)
                self.cross_attn_it_Bs.append(w_b_linear_q_cross_attn_it)
                self.cross_attn_it_As.append(w_a_linear_v_cross_attn_it)
                self.cross_attn_it_Bs.append(w_b_linear_v_cross_attn_it)
                blk.cross_attn_image_to_token.q_proj = _LoRA_qkv_proj(
                    cross_attn_it_q_proj,
                    w_a_linear_q_cross_attn_it,
                    w_b_linear_q_cross_attn_it,
                    lora_dropout=self.config["lora_dropout"],
                )
                blk.cross_attn_image_to_token.v_proj = _LoRA_qkv_proj(
                    cross_attn_it_v_proj,
                    w_a_linear_v_cross_attn_it,
                    w_b_linear_v_cross_attn_it,
                    lora_dropout=self.config["lora_dropout"],
                )

            # final attention token to image
            block = decoder_transformer.final_attn_token_to_image
            fa_ti_q_proj = block.q_proj
            fa_ti_v_proj = block.v_proj
            in_dim, out_dim = block.embedding_dim, block.internal_dim
            self.fa_ti_q_proj_A = nn.Linear(in_dim, dr, bias=False)
            self.fa_ti_q_proj_B = nn.Linear(dr, out_dim, bias=False)
            self.fa_ti_v_proj_A = nn.Linear(in_dim, dr, bias=False)
            self.fa_ti_v_proj_B = nn.Linear(dr, out_dim, bias=False)

            block.q_proj = _LoRA_qkv_proj(
                fa_ti_q_proj,
                self.fa_ti_q_proj_A,
                self.fa_ti_q_proj_B,
                lora_dropout=self.config["lora_dropout"],
            )
            block.v_proj = _LoRA_qkv_proj(
                fa_ti_v_proj,
                self.fa_ti_v_proj_A,
                self.fa_ti_v_proj_B,
                lora_dropout=self.config["lora_dropout"],
            )

        self.reset_parameters(zero_initial=zero_initial)
        self.sam = sam_model

    def save_lora_parameters(self, filename) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """
        filename = str(filename)
        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        sa_a_tensors = {f"sa_a_{i:03d}": self.self_attn_As[i].weight for i in range(len(self.self_attn_As))}
        sa_b_tensors = {f"sa_b_{i:03d}": self.self_attn_Bs[i].weight for i in range(len(self.self_attn_Bs))}
        cti_a_tensors = {f"cti_a_{i:03d}": self.cross_attn_ti_As[i].weight for i in range(len(self.cross_attn_ti_As))}
        cti_b_tensors = {f"cti_b_{i:03d}": self.cross_attn_ti_Bs[i].weight for i in range(len(self.cross_attn_ti_Bs))}
        cit_a_tensors = {f"cit_a_{i:03d}": self.cross_attn_it_As[i].weight for i in range(len(self.cross_attn_it_As))}
        cit_b_tensors = {f"cit_b_{i:03d}": self.cross_attn_it_Bs[i].weight for i in range(len(self.cross_attn_it_Bs))}
        fa_ti_tensors = {
            "fati_qa": self.fa_ti_q_proj_A.weight,
            "fati_qb": self.fa_ti_q_proj_B.weight,
            "fati_va": self.fa_ti_v_proj_A.weight,
            "fati_vb": self.fa_ti_v_proj_B.weight,
        }
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if "prompt_encoder" in key:
                prompt_encoder_tensors[key] = value
            if "mask_decoder" in key and "transformer" not in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **a_tensors,
            **b_tensors,
            **sa_a_tensors,
            **sa_b_tensors,
            **cti_a_tensors,
            **cti_b_tensors,
            **cit_a_tensors,
            **cit_b_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
            **fa_ti_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """
        filename = str(filename)
        assert filename.endswith(".pt") or filename.endswith(".pth")

        # Load state dict to CPU first to prevent errors when CUDA is unavailable
        state_dict = torch.load(filename, map_location=torch.device('cpu'), weights_only=True)

        # Determine the target device from the main SAM model
        target_device = next(self.sam.parameters()).device

        # Load LoRA weights for image encoder blocks and move to target device
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor.to(target_device))

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor.to(target_device))

        # Load LoRA weights for mask decoder self-attention and move to target device
        for i, sa_A_linear in enumerate(self.self_attn_As):
            saved_key = f"sa_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            sa_A_linear.weight = Parameter(saved_tensor.to(target_device))

        for i, sa_B_linear in enumerate(self.self_attn_Bs):
            saved_key = f"sa_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            sa_B_linear.weight = Parameter(saved_tensor.to(target_device))

        # Load LoRA weights for mask decoder cross-attention (token to image) and move to target device
        for i, cti_a_linear in enumerate(self.cross_attn_ti_As):
            saved_key = f"cti_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            cti_a_linear.weight = Parameter(saved_tensor.to(target_device))

        for i, cti_b_linear in enumerate(self.cross_attn_ti_Bs):
            saved_key = f"cti_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            cti_b_linear.weight = Parameter(saved_tensor.to(target_device))

        # Load LoRA weights for mask decoder cross-attention (image to token) and move to target device
        for i, cit_a_linear in enumerate(self.cross_attn_it_As):
            saved_key = f"cit_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            cit_a_linear.weight = Parameter(saved_tensor.to(target_device))

        for i, cit_b_linear in enumerate(self.cross_attn_it_Bs):
            saved_key = f"cit_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            cit_b_linear.weight = Parameter(saved_tensor.to(target_device))

        # Load LoRA weights for final attention (token to image) and move to target device
        self.fa_ti_q_proj_A.weight = Parameter(state_dict["fati_qa"].to(target_device))
        self.fa_ti_q_proj_B.weight = Parameter(state_dict["fati_qb"].to(target_device))
        self.fa_ti_v_proj_A.weight = Parameter(state_dict["fati_va"].to(target_device))
        self.fa_ti_v_proj_B.weight = Parameter(state_dict["fati_vb"].to(target_device))

        # Prepare to load non-LoRA parts (prompt encoder, parts of mask decoder)
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # Load prompt encoder weights and move to target device
        prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
        prompt_encoder_new_state_dict = {
            k: state_dict[k].to(target_device) for k in prompt_encoder_keys if k in state_dict
        }
        sam_dict.update(prompt_encoder_new_state_dict)

        # Load mask decoder weights (excluding transformer parts modified by LoRA) and move to target device
        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k and "transformer" not in k]
        mask_decoder_new_state_dict = {
            k: state_dict[k].to(target_device) for k in mask_decoder_keys if k in state_dict
        }
        sam_dict.update(mask_decoder_new_state_dict)

        # Load the updated state dict into the main SAM model
        self.sam.load_state_dict(sam_dict)

    def kaiming_uniform_5(self, tensor) -> None:
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    def reset_parameters(self, zero_initial=False) -> None:
        if zero_initial:
            initial_func = nn.init.zeros_
        else:
            initial_func = self.kaiming_uniform_5
        if self.config["image_encoder_lora_rank"] > 0:
            for w_A in self.w_As:
                initial_func(w_A.weight)
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)
        if self.config["mask_decoder_lora_rank"] > 0:
            for w_A in self.self_attn_As:
                initial_func(w_A.weight)
            for w_B in self.self_attn_Bs:
                nn.init.zeros_(w_B.weight)
            for w_A in self.cross_attn_ti_As:
                initial_func(w_A.weight)
            for w_B in self.cross_attn_ti_Bs:
                nn.init.zeros_(w_B.weight)
            for w_A in self.cross_attn_it_As:
                initial_func(w_A.weight)
            for w_B in self.cross_attn_it_Bs:
                nn.init.zeros_(w_B.weight)
            initial_func(self.fa_ti_q_proj_A.weight)
            nn.init.zeros_(self.fa_ti_q_proj_B.weight)
            initial_func(self.fa_ti_v_proj_A.weight)
            nn.init.zeros_(self.fa_ti_v_proj_B.weight)

    def forward(self, batched_input, multimask_output, input_image_embeddings=None):
        return self.sam(batched_input, multimask_output, input_image_embeddings)

    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
