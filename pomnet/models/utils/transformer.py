import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from pomnet.models.utils.builder import TRANSFORMER



@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #     ['self_attn', 'norm', 'cross_attn', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            # assert not self.pre_norm, f'Use prenorm in ' \
            #                           f'{self.__class__.__name__},' \
            #                           f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@TRANSFORMER.register_module()
class DynamicConv(BaseModule):
    """Implements Dynamic Convolution.
    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .
    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        with_proj (bool): Project two-dimentional feature to
            one-dimentional feature. Default to True.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DynamicConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.
        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).
        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        input_feature = input_feature.flatten(2).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features