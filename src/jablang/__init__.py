from dataclasses import fields
from functools import singledispatch

import einops
import equinox as eqx
import jax
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from functools import partial
import numpy as np
import ablang

@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


# basic types
from_torch.register(torch.Tensor, lambda x: np.array(x.detach()))
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.GELU, lambda _: jax.nn.gelu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])


class AbstractFromTorch(eqx.Module):
    """
    Default implementation of `from_torch` for equinox modules.
    This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
    Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.

    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module):
        # assemble arguments to `cls` constructor from `model`

        field_to_type = {field.name: field.type for field in fields(cls)}
        kwargs = {
            child: from_torch(child_module)
            for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        }

        # add fields that are not child_modules or parameters
        for field_name, field_type in field_to_type.items():
            if not hasattr(model, field_name):
                if not isinstance(None, field_type):
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                    )
                else:
                    kwargs[field_name] = None
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        # check we're not passing any additional properties
        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
            )

        return cls(**kwargs)


def register_from_torch(torch_module_type):
    """Class decorator to register an equinox module for conversion from a torch module."""

    def decorator(cls):
        from_torch.register(torch_module_type, cls.from_torch)
        return cls

    return decorator


# this isn't very jax-y
def _vmap(f, tensor, *args):
    for _ in range(len(tensor.shape) - 1):
        f = jax.vmap(f)
    return f(tensor, *args)


def vmap_to_last_dimension(f):
    return partial(_vmap, f)


@register_from_torch(torch.nn.Linear)
class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics"""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(l: torch.nn.Linear):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))


@register_from_torch(torch.nn.LayerNorm)
class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics"""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        ln = eqx.nn.LayerNorm(
            shape=x.shape[-1],
            eps=self.eps,
            use_weight=self.weight is not None,
            use_bias=self.bias is not None,
        )
        ln = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            ln,
            (self.weight, self.bias),
            is_leaf=lambda x: x is None,
        )

        return vmap_to_last_dimension(ln)(x)

    @staticmethod
    def from_torch(l: torch.nn.LayerNorm):
        return LayerNorm(
            weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps
        )


@register_from_torch(torch.nn.Sequential)
class Sequential(eqx.Module):
    _modules: dict[
        str, AbstractFromTorch
    ]  # IMHO this is a fairly wild design choice, but this is really how pytorch works.

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))


@register_from_torch(torch.nn.modules.sparse.Embedding)
class SparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __call__(self, indices):
        ndims = len(indices.shape)

        def apply(index):
            return self.embedding(index)

        f = apply
        for _ in range(ndims):
            f = jax.vmap(f)

        return f(indices)

    @staticmethod
    def from_torch(m: torch.nn.modules.sparse.Embedding):
        return SparseEmbedding(embedding=eqx.nn.Embedding(weight=from_torch(m.weight)))


@register_from_torch(torch.nn.Embedding)
class Embedding(eqx.Module):
    weight: Float[Array, "Vocab Embedding"]
    padding_idx: int

    def __call__(self, indices):
        return jax.numpy.take(
            self.weight,
            indices,
            axis=0,
        )

    @staticmethod
    def from_torch(m: torch.nn.Embedding):
        return Embedding(
            weight=from_torch(m.weight),
            padding_idx=m.padding_idx,
        )


@register_from_torch(ablang.AbHead)
class AbHead(eqx.Module):
    dense: Linear
    layer_norm: LayerNorm
    decoder: Linear

    def __call__(self, features):
        x = jax.nn.gelu(self.dense(features))
        return self.decoder(self.layer_norm(x))

    @staticmethod
    def from_torch(m: ablang.AbHead):
        assert m.activation.__name__ == "gelu"
        return AbHead(
            dense=from_torch(m.dense),
            layer_norm=from_torch(m.layer_norm),
            decoder=from_torch(m.decoder),
        )


@register_from_torch(ablang.embedding.AbEmbeddings)
class AbEmbeddings(eqx.Module):
    AAEmbeddings: Embedding
    PositionEmbeddings: Embedding
    LayerNorm: LayerNorm
    pad_token_idx: int

    def __call__(self, tokens):
        embeddings = self.AAEmbeddings(tokens)
        mask = tokens != self.pad_token_idx
        position_ids = jnp.cumsum(mask, axis=-1).astype(jnp.int32) * mask
        position_embeddings = self.PositionEmbeddings(position_ids)
        return self.LayerNorm(embeddings + position_embeddings)

    @staticmethod
    def from_torch(m: ablang.embedding.AbEmbeddings):
        assert m.Dropout.training is False
        return AbEmbeddings(
            AAEmbeddings=from_torch(m.AAEmbeddings),
            PositionEmbeddings=from_torch(m.PositionEmbeddings),
            LayerNorm=from_torch(m.LayerNorm),
            pad_token_idx=m.pad_token_id,
        )


@register_from_torch(ablang.encoderblocks.ThirdMultiHeadAttention)
class MHA(eqx.Module):
    attention: eqx.nn.MultiheadAttention

    def __call__(self, hidden_states, mask=None):
        return jax.vmap(self.attention)(hidden_states, hidden_states, hidden_states, mask=mask)

    @staticmethod
    def from_torch(m: ablang.encoderblocks.ThirdMultiHeadAttention):
        m = m.Attention
        assert m.dropout_module.training is False
        eqx_attn = eqx.nn.MultiheadAttention(
            num_heads=m.num_heads,
            key_size=m.kdim,
            query_size=m.kdim,
            use_key_bias=True,
            use_value_bias=True,
            use_query_bias=True,
            use_output_bias=True,
            key=jax.random.key(0),
        )
        eqx_attn = eqx.tree_at(
            lambda m: (m.query_proj, m.key_proj, m.value_proj, m.output_proj),
            eqx_attn,
            from_torch((m.q_proj, m.k_proj, m.v_proj, m.out_proj)),
        )

        return MHA(attention=eqx_attn)


@register_from_torch(ablang.encoderblocks.IntermediateLayer)
class IntermediateLayer(eqx.Module):
    expand_dense: Linear
    dense_dense: Linear
    layer_norm: LayerNorm

    def __call__(self, input):
        x = self.expand_dense(input)
        x = jax.nn.gelu(x)
        x = self.dense_dense(x)
        return self.layer_norm(x + input)

    @staticmethod
    def from_torch(m: ablang.encoderblocks.IntermediateLayer):
        assert m.intermediate_act_fn.__name__ == "gelu"
        assert m.dropout.training is False
        return IntermediateLayer(
            expand_dense=from_torch(m.expand_dense),
            dense_dense=from_torch(m.dense_dense),
            layer_norm=from_torch(m.LayerNorm),
        )

@register_from_torch(ablang.encoderblocks.EncoderBlock)
class EncoderBlock(eqx.Module):
    attention: MHA
    intermediate: IntermediateLayer
    layer_norm: LayerNorm

    def __call__(self, input, mask=None):
        x = self.attention(input, mask=mask)
        x = self.layer_norm(x + input)
        x = self.intermediate(x)
        return x

    @staticmethod
    def from_torch(m: ablang.encoderblocks.EncoderBlock):
        assert m.MHADropout.training is False
        return EncoderBlock(
            attention=from_torch(m.MultiHeadAttention),
            intermediate=from_torch(m.IntermediateLayer),
            layer_norm=from_torch(m.MHALayerNorm),
        )
        

# @register_from_torch(esm.layers.transformer_stack.TransformerStack)
# class TransformerStack(AbstractFromTorch):
#     block_params: UnifiedTransformerBlock
#     block_static: UnifiedTransformerBlock
#     norm: LayerNorm

#     def __call__(self, x: Float[Array, "B N D"]):
#         def body(x, params):
#             layer = eqx.combine(self.block_static, params)
#             x = layer(x)
#             return x, x

#         final_state, all_states = jax.lax.scan(
#             body,
#             x,
#             self.block_params,
#         )
#         return self.norm(final_state), all_states

#     @staticmethod
#     def from_torch(m: esm.layers.transformer_stack.TransformerStack):
#         blocks = [from_torch(b) for b in m.blocks]
#         block_params = jax.tree.map(
#             lambda *v: jnp.stack(v),
#             *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
#         )
#         block_static = eqx.partition(blocks[0], eqx.is_inexact_array)[1]
#         return TransformerStack(
#             block_params=block_params,
#             block_static=block_static,
#             norm=from_torch(m.norm),
#         )


@register_from_torch(ablang.encoderblocks.EncoderBlocks)
class EncoderBlocks(eqx.Module):
    layer_params: EncoderBlock
    layer_static: EncoderBlock

    def __call__(self, x, attention_mask=None):
        def body(x, params):
            layer = eqx.combine(self.layer_static, params)
            x = layer(x, mask=attention_mask)
            return x, None

        final_state, _ = jax.lax.scan(
            body,
            x,
            self.layer_params,
        )
        return final_state
    
    @staticmethod
    def from_torch(m: ablang.encoderblocks.EncoderBlocks):
        blocks = [from_torch(b) for b in m.Layers]
        block_params = jax.tree_map(
            lambda *v: jnp.stack(v),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        block_static = eqx.partition(blocks[0], eqx.is_inexact_array)[1]
        return EncoderBlocks(
            layer_params=block_params,
            layer_static=block_static,
        )

@register_from_torch(ablang.AbRep)
class AbRep(eqx.Module):
    embedding: AbEmbeddings
    encoder: EncoderBlocks
    pad_idx: int

    def __call__(self, x: Int[Array, "B N"]):
        attention_mask = x == self.pad_idx
        x = self.embedding(x)
        return self.encoder(x, attention_mask=None)#attention_mask)

    @staticmethod
    def from_torch(m: ablang.AbRep):
        return AbRep(
            embedding=from_torch(m.AbEmbeddings),
            encoder=from_torch(m.EncoderBlocks),
            pad_idx=m.hparams.pad_token_id
        )
    
@register_from_torch(ablang.AbLang)
class AbLang(eqx.Module):
    rep: AbRep
    head: AbHead


    def __call__(self, x: Int[Array, "B N"]):
        return self.head(self.rep(x))

    @staticmethod
    def from_torch(m: ablang.AbLang):
        return AbLang(
            rep=from_torch(m.AbRep),
            head=from_torch(m.AbHead),
        )