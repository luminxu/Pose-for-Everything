import torch
import torch.nn as nn
import numpy as np

from pomnet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import (Conv2d, Linear, xavier_init, build_upsample_layer, ConvModule,
                      constant_init, normal_init, build_conv_layer, build_norm_layer)
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)

from mmpose.models import HEADS
from mmpose.models.heads import TopdownHeatmapBaseHead
from mmpose.models.builder import build_loss
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.models.utils.ops import resize
from pomnet.models.utils import build_transformer


@TRANSFORMER.register_module()
class Transformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(Transformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, mask_query):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].

        Notes:
            x: query image features with shape [bs, c, h, w]
            mask: mask for x with shape [bs, h, w]
            pos_embed: positional embedding for x with shape [bs, c, h, w]
            query_embed: sample keypoint features with shape [bs, num_query, c]
            mask_query: mask for query_embed with shape [bs, num_query]
        Outputs:
            out_dec: [num_layers, bs, num_query, c]
            memory: [bs, c, h, w]

        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)

        query_embed = query_embed.permute(1, 0, 2)  # [bs, num_query, c] -> [num_query, bs, c]
        # target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, c]
        out_dec = self.decoder(
            query=query_embed,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            # query_pos=query_embed,
            query_key_padding_mask=mask_query,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory


class RelationModel(nn.Module):
    """
    Generic Matching Network from Lu et al 2018
    Clas Agnostic Counting.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(RelationModel, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    inplace=True)

    def forward(self, pooled_feature_s, feature_q):
        """Forward function."""

        feature_sample = pooled_feature_s.expand_as(feature_q).clone()
        feature = torch.cat((feature_q, feature_sample), dim=1)

        feature = self.conv(feature)

        return feature


@HEADS.register_module()
class TransformerHead(TopdownHeatmapBaseHead):

    def __init__(self,
                 in_channels,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super().__init__()

        out_channels = 1
        self.in_channels = in_channels
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'

        self.relation_model = RelationModel(self.embed_dims * 2, self.embed_dims)

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            assert num_deconv_layers > 0
            conv_channels = num_deconv_filters[
                -1]

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

        self._init_layers()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        deconv_in_channels = self.embed_dims

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=deconv_in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            deconv_in_channels = planes

        return nn.Sequential(*layers)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.query_proj = Linear(
            self.in_channels, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

        for m in self.relation_model.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x, feature_s, target_s, mask_s):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        # process query image feature
        x = self.input_proj(x)
        masks = x.new_zeros((x.shape[0], x.shape[2], x.shape[3])).to(torch.bool)
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]

        # process keypoint token feature
        # query_embed: [bs, num_query, c]
        # masks_query: [bs, num_query]
        query_embed_list = []
        for feature, target in zip(feature_s, target_s):
            resized_feature = resize(
                input=feature,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False)
            target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
            query_embed = target.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
            query_embed_list.append(query_embed)
        query_embed = torch.mean(torch.stack(query_embed_list, dim=0), 0)
        query_embed = query_embed * mask_s
        query_embed = self.query_proj(query_embed)
        masks_query = (~mask_s.to(torch.bool)).squeeze(-1)

        # outs_dec: [nb_dec, bs, num_query, c]
        # memory: [bs, c, h, w]
        outs_dec, memory = self.transformer(x, masks, query_embed,
                                       pos_embed, masks_query)
        out_dec = outs_dec[-1] # [bs, num_query, c]

        heatmaps = []
        for kpt in range(out_dec.shape[1]):
            pooled_feature_s = out_dec[:, kpt].unsqueeze(-1).unsqueeze(-1)
            feature = self.relation_model(pooled_feature_s, memory)
            heatmap = self.deconv_layers(feature)
            heatmap = self.final_layer(heatmap)
            heatmaps.append(heatmap)
        heatmaps = torch.cat(heatmaps, dim=1)

        return heatmaps

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['mse_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatMap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                thr=0.2)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])

            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result
