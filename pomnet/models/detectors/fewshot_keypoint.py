import math

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS

@POSENETS.register_module()
class FewShotKeypoint(BasePose):
    """Few-shot keypoint detectors.

    Args:
        encoder_sample (dict): Backbone modules to extract feature.
        encoder_query (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 encoder_sample,
                 encoder_query,
                 keypoint_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.encoder_sample = builder.build_backbone(encoder_sample)
        self.encoder_query = builder.build_backbone(encoder_query)

        self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained=pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.encoder_sample.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head.init_weights()

    def forward(self,
                img_s,
                img_q,
                target_s=None,
                target_weight_s=None,
                target_q=None,
                target_weight_q=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes and image paths.
        """
        if return_loss:
            return self.forward_train(img_s, target_s, target_weight_s,
                                      img_q, target_q, target_weight_q,
                                      img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img_s, target_s, target_weight_s,
                      img_q, target_q, target_weight_q, img_metas, **kwargs)

    def forward_train(self, img_s, target_s, target_weight_s,
                      img_q, target_q, target_weight_q,
                      img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        feature_s = [self.encoder_sample(img) for img in img_s]
        feature_q = self.encoder_query(img_q)

        output = self.keypoint_head(feature_s, target_s, feature_q)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            mask_s = target_weight_s[0]
            for target_weight in target_weight_s:
                mask_s = mask_s * target_weight

            keypoint_losses = self.keypoint_head.get_loss(
                output, target_q, target_weight_q * mask_s)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target_q, target_weight_q * mask_s)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img_s, target_s, target_weight_s,
                      img_q, target_q, target_weight_q, img_metas=None, **kwargs):
        """Defines the computation performed at every call when testing."""
        batch_size, _, img_height, img_width = img_q.shape

        result = {}
        feature_s = [self.encoder_sample(img) for img in img_s]
        feature_q = self.encoder_query(img_q)

        output_all = []

        for kid in range(target_q.shape[1]):
            target_sample = [target[:, kid:kid+1] for target in target_s]
            output = self.keypoint_head(feature_s, target_sample, feature_q)
            output_heatmap = output.detach().cpu().numpy()
            output_all.append(output_heatmap)

        output_all = np.hstack(output_all)

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_all, img_size=[img_width, img_height])
            result.update(keypoint_result)

        return result

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
            imshow(enlarge, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img


