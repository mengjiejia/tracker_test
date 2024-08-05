# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import random
import argparse
import json
import os
import cv2
import torch
import numpy as np
from scipy import ndimage

from data_utils import tensor2img, GOT10k_dataset
from pysot.pysot.models.model_builder_apn import ModelBuilderAPN
from pysot.pysot.tracker.siamapn_tracker import SiamAPNTracker
from pysot.pysot.utils.bbox import get_axis_aligned_bbox
from pysot.pysot.utils.model_load import load_pretrain
from pysot.toolkit.datasets import DatasetFactory
from pysot.pysot.core.config_apn import cfg
from pysot.toolkit.utils.statistics import overlap_ratio
import sys
from data_utils import normalize

parser = argparse.ArgumentParser(description='siamapn tracking')
parser.add_argument('--dataset', default='V4RFlight112', type=str,
                    help='datasets')
parser.add_argument('--snapshot', default='/home/mengjie/PycharmProjects/Ad2Attack/snapshot/general_model.pth',
                    type=str,
                    help='snapshot of models to eval')  # './snapshot/general_model.pth'
parser.add_argument('--config', default='../experiments/config.yaml', type=str,
                    help='config file')
parser.add_argument('--trackername', default='SiamAPN', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()
from Setting import *
from Model_config_test import *
from Model_config_test_R import *

model_name = opt.model
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


def squeeze_color(img, i):
    p = np.power(2, i) - 1
    image = img * p
    image = np.rint(image)
    image = image / p

    return image


def median_filter(img, s):
    image = ndimage.median_filter(img, size=s)

    return image


# main() backup
# def main():
#     # load config
#     # print(args.config)
#     cfg.merge_from_file(args.config)
#     model = ModelBuilderAPN()
#
#     # print(args.snapshot)
#     model = load_pretrain(model, args.snapshot).cuda().eval()
#     tracker = SiamAPNTracker(model)
#
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)
#
#     # create dataset
#     dataset = DatasetFactory.create_dataset(name=args.dataset,
#                                             dataset_root=dataset_root,
#                                             load_img=False)
#
#     model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
#     # OPE tracking
#     # print('video name', len(dataset['air conditioning box1'][0]))
#     # stop
#     # video is a tuple (img, groundtruth bounding box)
#     for v_idx, video in enumerate(dataset):
#         # if v_idx == 20:
#         #     break
#         # if video.name=='uav1_1':
#         if args.video != '':
#             # test one special video
#             if video.name != args.video:
#                 continue
#         toc = 0
#         pred_bboxes_adv = []
#         pred_bboxes = []
#         scores_adv = []
#         scores = []
#         track_times = []
#
#         model_path = os.path.join('results', args.dataset, model_name)
#         if not os.path.isdir(model_path):
#             os.makedirs(model_path)
#
#         img_dir = os.path.join(model_path, video.name)
#
#         if not os.path.isdir(img_dir):
#             os.makedirs(img_dir)
#
#         for idx, (img, gt_bbox) in enumerate(video):
#             # squeeze color
#             # img = squeeze_color(img, 4)
#
#             # median_filter
#             # img = median_filter(img, 4)
#             # remember to calculate the running time separately for normal and attack model
#             tic = cv2.getTickCount()
#             if idx == 0:
#                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
#                 gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
#                 tracker.init(img, gt_bbox_)
#                 pred_bbox = gt_bbox_
#                 pred_img = img
#                 scores.append(None)
#                 if 'VOT2018-LT' == args.dataset:
#                     pred_bboxes.append([1])
#                 else:
#                     pred_bboxes.append(pred_bbox)
#                     pred_bboxes_adv.append(pred_bbox)
#             else:
#
#                 # normal image prediction
#                 # before
#                 # outputs = tracker.track_base(img)
#                 x_crop, scale_z = tracker.get_x_crop(img)
#                 outputs = tracker.track(img, x_crop)
#                 pred_bbox = outputs['bbox']
#                 pred_bboxes.append(pred_bbox)
#                 scores.append(outputs['best_score'])
#
#                 # attack prediction
#                 zhanbi = (pred_bbox[2] * pred_bbox[3]) / (pred_img.shape[0] * pred_img.shape[1])
#                 outputs_adv = tracker.track_adv(img, zhanbi, AdA)
#                 pred_bbox_adv = outputs_adv['bbox']
#                 pred_bboxes_adv.append(pred_bbox_adv)
#                 scores_adv.append(outputs_adv['best_score'])
#
#
#             toc += cv2.getTickCount() - tic
#             track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
#             if idx == 0:
#                 cv2.destroyAllWindows()
#             # if args.vis and idx > 0:
#             if idx > 0:
#                 # draw normal prediction in yellow color
#                 pred_bbox = list(map(int, pred_bbox))
#                 cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
#                               (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
#
#                 # draw attack prediction in red color
#                 pred_bbox_adv = list(map(int, pred_bbox_adv))
#                 cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
#                               (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255), 3)
#
#
#                 # draw ground truth in green color
#                 if not np.isnan(gt_bbox).any():
#                     gt_bbox = list(map(int, gt_bbox))
#                     cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
#                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
#
#                 cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                 # save image
#                 # img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
#                 # cv2.imwrite(img_name, img)
#                 #
#                 # cv2.imshow(video.name, img)
#                 if video.name == 'basketball player1' and idx == 225:
#                     cv2.waitKey(100)
#                 cv2.waitKey(1)
#         toc /= cv2.getTickFrequency()
#         # save results
#
#         if not os.path.isdir(model_path):
#             os.makedirs(model_path)
#
#         normal_img_dir = os.path.join(model_path, 'normal')
#         adv_img_dir = os.path.join(model_path, 'attack')
#
#         if not os.path.isdir(normal_img_dir):
#             os.makedirs(normal_img_dir)
#         if not os.path.isdir(adv_img_dir):
#             os.makedirs(adv_img_dir)
#         result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
#         result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))
#
#         # write normal prediction bounding box
#         with open(result_path_normal, 'w') as f:
#             for x in pred_bboxes:
#                 f.write(','.join([str(i) for i in x]) + '\n')
#
#         # write attack prediction bounding box
#         with open(result_path_adv, 'w') as f:
#             for x in pred_bboxes_adv:
#                 f.write(','.join([str(i) for i in x]) + '\n')
#
#         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
#             v_idx + 1, video.name, toc, idx / toc))

# def my_main_backup():
#     # load config
#     # print(args.config)
#     cfg.merge_from_file(args.config)
#     model_normal = ModelBuilderAPN()
#     model_attack = ModelBuilderAPN()
#
#     # print(args.snapshot)
#     model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
#     tracker_normal = SiamAPNTracker(model_normal)
#
#     model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
#     tracker_attack = SiamAPNTracker(model_attack)
#
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)
#
#     # create dataset
#     dataset = DatasetFactory.create_dataset(name=args.dataset,
#                                             dataset_root=dataset_root,
#                                             load_img=False)
#
#     model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
#     # OPE tracking
#     # print('video name', len(dataset['air conditioning box1'][0]))
#     # stop
#     # video is a tuple (img, groundtruth bounding box)
#     for v_idx, video in enumerate(dataset):
#         # if v_idx == 20:
#         #     break
#         # if video.name=='uav1_1':
#         if args.video != '':
#             # test one special video
#             if video.name != args.video:
#                 continue
#         toc = 0
#         pred_bboxes_adv = []
#         pred_bboxes = []
#         scores_adv = []
#         scores = []
#         track_times = []
#
#         model_path = os.path.join('results', args.dataset, model_name)
#         if not os.path.isdir(model_path):
#             os.makedirs(model_path)
#
#         img_dir = os.path.join(model_path, video.name)
#
#         if not os.path.isdir(img_dir):
#             os.makedirs(img_dir)
#
#         for idx, (img, gt_bbox) in enumerate(video):
#             # squeeze color
#             # img = squeeze_color(img, 4)
#
#             # median_filter
#             # img = median_filter(img, 4)
#             # remember to calculate the running time separately for normal and attack model
#             tic = cv2.getTickCount()
#             if idx == 0:
#                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
#                 gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
#                 pred_bbox = gt_bbox_
#                 pred_bbox_adv = gt_bbox_
#                 pred_img = img
#                 tracker_normal.init(img, gt_bbox_)
#                 zhanbi_template = (gt_bbox_[2] * gt_bbox_[3]) / (pred_img.shape[0] * pred_img.shape[1])
#                 tracker_attack.init_adv(img, gt_bbox_, zhanbi_template, AdA)
#                 # initialize AdA template
#                 AdA.template = tracker_attack.template
#
#                 scores.append(None)
#                 if 'VOT2018-LT' == args.dataset:
#                     pred_bboxes.append([1])
#                 else:
#                     pred_bboxes.append(pred_bbox)
#                     pred_bboxes_adv.append(pred_bbox)
#             else:
#
#                 # normal image prediction
#                 # before
#                 # outputs = tracker.track_base(img)
#                 # from here
#                 x_crop, scale_z = tracker_normal.get_x_crop(img)
#                 stop
#
#                 outputs = tracker_normal.track(img, x_crop)
#                 stop
#                 pred_bbox_normal = outputs['bbox']
#                 pred_bboxes.append(pred_bbox_normal)
#                 scores.append(outputs['best_score'])
#
#                 # attack prediction
#                 zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
#                 # loss = tracker.check_loss(img, zhanbi, AdA)
#                 # print('loss', loss)
#                 # cv2.imshow(video.name, img_clean1)
#
#                 outputs_adv = tracker_attack.track_adv(img, zhanbi, AdA)
#                 # pred_bbox = outputs_adv['bbox']
#                 pred_bbox_adv = outputs_adv['bbox']
#                 pred_bboxes_adv.append(pred_bbox_adv)
#                 scores_adv.append(outputs_adv['best_score'])
#
#
#             toc += cv2.getTickCount() - tic
#             track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
#             if idx == 0:
#                 cv2.destroyAllWindows()
#             # if args.vis and idx > 0:
#             if idx > 0:
#                 # draw normal prediction in yellow color
#                 pred_bbox_normal = list(map(int, pred_bbox_normal))
#                 cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
#                               (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]), (0, 255, 255), 3)
#
#                 # draw attack prediction in red color
#                 pred_bbox_adv = list(map(int, pred_bbox_adv))
#                 cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
#                               (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255), 3)
#
#
#                 # draw ground truth in green color
#                 if not np.isnan(gt_bbox).any():
#                     gt_bbox = list(map(int, gt_bbox))
#                     cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
#                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
#
#                 cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                 # save image
#                 img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
#                 cv2.imwrite(img_name, img)
#                 #
#                 # cv2.imshow(video.name, img)
#                 if video.name == 'basketball player1' and idx == 225:
#                     cv2.waitKey(100)
#                 cv2.waitKey(1)
#         toc /= cv2.getTickFrequency()
#         # save results
#
#         if not os.path.isdir(model_path):
#             os.makedirs(model_path)
#
#         normal_img_dir = os.path.join(model_path, 'normal')
#         adv_img_dir = os.path.join(model_path, 'attack')
#
#         if not os.path.isdir(normal_img_dir):
#             os.makedirs(normal_img_dir)
#         if not os.path.isdir(adv_img_dir):
#             os.makedirs(adv_img_dir)
#         result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
#         result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))
#
#         # write normal prediction bounding box
#         with open(result_path_normal, 'w') as f:
#             for x in pred_bboxes:
#                 f.write(','.join([str(i) for i in x]) + '\n')
#
#         # write attack prediction bounding box
#         with open(result_path_adv, 'w') as f:
#             for x in pred_bboxes_adv:
#                 f.write(','.join([str(i) for i in x]) + '\n')
#
#         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
#             v_idx + 1, video.name, toc, idx / toc))

# main for loss
def main():
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_normal = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()

    # print(args.snapshot)
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    # OPE tracking
    # print('video name', len(dataset['air conditioning box1'][0]))
    # stop
    # video is a tuple (img, groundtruth bounding box)
    for v_idx, video in enumerate(dataset):
        if v_idx == 4:
            break
        # if video.name=='uav1_1':
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_adv = []
        pred_bboxes = []
        scores_adv = []
        scores = []
        track_times = []
        iou_v = []

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):
            # squeeze color
            # img = squeeze_color(img, 4)

            # median_filter
            img_filter = median_filter(img, 3)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox = gt_bbox_
                pred_bbox_adv = gt_bbox_
                pred_img = img
                tracker_normal.init(img_filter, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                iou_v.append(None)

                # attack template
                # zhanbi_template = (gt_bbox_[2] * gt_bbox_[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # tracker_attack.init_adv(img, gt_bbox_, zhanbi_template, AdA)
                # # initialize AdA template
                # AdA.template = tracker_attack.template

                # scores.append(None)
                # scores_adv.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                    pred_bboxes_adv.append(pred_bbox)
            else:

                # normal image prediction
                # before
                # outputs = tracker.track_base(img)
                # from here
                x_crop, scale_z = tracker_normal.get_x_crop(img_filter)
                outputs = tracker_normal.track(img_filter, x_crop)

                pred_bbox_normal = outputs['bbox']
                pred_bboxes.append(pred_bbox_normal)
                scores.append(outputs['best_score'])

                # attack prediction
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # print(idx, 'zhanbi', zhanbi)

                # loss = tracker.check_loss(img, zhanbi, AdA)
                # print('loss', loss)
                # cv2.imshow(video.name, img_clean1)

                outputs_adv = tracker_attack.track_adv(img, zhanbi, AdA)
                # pred_bbox = outputs_adv['bbox']
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])

                bbox_normal = np.array(pred_bbox_normal)
                bbox_normal = np.expand_dims(bbox_normal, axis=0)
                bbox_adv = np.array(pred_bbox_adv)
                bbox_adv = np.expand_dims(bbox_adv, axis=0)

                iou_img = overlap_ratio(bbox_normal, bbox_adv)
                iou_v.append(iou_img[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            # if args.vis and idx > 0:
            if idx > 0:
                # draw normal prediction in yellow color
                pred_bbox_normal = list(map(int, pred_bbox_normal))
                cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
                              (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]),
                              (0, 255, 255), 3)

                # draw attack prediction in red color
                pred_bbox_adv = list(map(int, pred_bbox_adv))
                cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
                              (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255),
                              3)

                # draw ground truth in green color
                if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # save image
                img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
                cv2.imwrite(img_name, img)
                #
                cv2.imshow(video.name, img)
                if video.name == 'basketball player1' and idx == 225:
                    cv2.waitKey(100)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        normal_img_dir = os.path.join(model_path, 'normal')
        adv_img_dir = os.path.join(model_path, 'attack')

        if not os.path.isdir(normal_img_dir):
            os.makedirs(normal_img_dir)
        if not os.path.isdir(adv_img_dir):
            os.makedirs(adv_img_dir)
        result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        with open(result_path_normal, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write attack prediction bounding box
        with open(result_path_adv, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')

        iou_path = os.path.join(img_dir, 'iou.json')
        json.dump(iou_v, open(iou_path, 'w'))

        score_diff = np.array(scores) - np.array(scores_adv)
        score_diff_path = os.path.join(img_dir, 'score_diff.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))


# test loss
#
# def forward(self,target_sz=(255,255)):
#     if self.zhanbi < 0.002:
#         block_num = 1
#         search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(128, 128), mode='bilinear')
#     elif self.zhanbi < 0.02:
#         block_num = 2
#         search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(64, 64), mode='bilinear')
#     else:
#         block_num = 3
#         search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(32, 32), mode='bilinear')
#     search512_adv1 = self.netA(search512_clean2, block_num)
#
#     self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
#     self.search_adv255 = self.search_adv1 * 127.5 + 127.5
#
# def optimize_parameters(self):
#     with torch.no_grad():
#         self.siam.model.template(self.template)
#         self.score_maps_clean_o = self.siam.get_heat_map(self.search_clean255,softmax=True)#(5HWN,),with softmax
#     self.forward()
#     self.score_maps_adv,self.reg_res_adv = self.siam.get_cls_reg(self.search_adv255,softmax=False)#(5HWN,2)without softmax,(5HWN,4)
#     self.score_maps_clean, self.reg_res_clean = self.siam.get_cls_reg(self.search_clean255, softmax=False)
#     self.optimizer_A.zero_grad()
#     self.backward_A()
#     self.optimizer_A.step()

# save clean and adversarial images

def save_images():
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_normal = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()

    # print(args.snapshot)
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    for v_idx, video in enumerate(dataset):

        # if v_idx == 4:
        #    break
        # if video.name=='uav1_1':
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_adv = []
        pred_bboxes = []
        scores_adv = []
        scores = []
        track_times = []
        iou_v = []

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):
            # squeeze color
            # img = squeeze_color(img, 4)

            # median_filter
            # img_filter = median_filter(img, 3)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox = gt_bbox_
                pred_bbox_adv = gt_bbox_
                pred_img = img
                tracker_normal.init(img, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                iou_v.append(None)

                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                    pred_bboxes_adv.append(pred_bbox)
            else:

                # normal image prediction
                # before
                # outputs = tracker.track_base(img)
                # from here
                x_crop, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(
                    img)
                outputs = tracker_normal.track(img, x_crop)

                pred_bbox_normal = outputs['bbox']
                pred_bboxes.append(pred_bbox_normal)
                scores.append(outputs['best_score'])

                # attack prediction
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # print(idx, 'zhanbi', zhanbi)

                # loss = tracker.check_loss(img, zhanbi, AdA)
                # print('loss', loss)
                # cv2.imshow(video.name, img_clean1)

                # outputs_adv, x_crop_adv = tracker_attack.track_adv(img, zhanbi, AdA)
                outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(
                    img, zhanbi, AdA)
                # pred_bbox = outputs_adv['bbox']
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])

                bbox_normal = np.array(pred_bbox_normal)
                bbox_normal = np.expand_dims(bbox_normal, axis=0)
                bbox_adv = np.array(pred_bbox_adv)
                bbox_adv = np.expand_dims(bbox_adv, axis=0)

                iou_img = overlap_ratio(bbox_normal, bbox_adv)
                iou_v.append(iou_img[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            # if args.vis and idx > 0:
            if idx > 0:
                # save image
                tracker_normal.save_img(x_crop, x_crop_adv, img_dir, idx)

        toc /= cv2.getTickFrequency()
        # save results

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))


# input is the attack full image
def filter_test(filter_size):
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_filter = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()
    model_normal = ModelBuilderAPN()

    # model filter
    model_filter = load_pretrain(model_filter, args.snapshot).cuda().eval()
    tracker_filter = SiamAPNTracker(model_filter)

    # model attack
    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    # model normal
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)

    model_path = os.path.join('results', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    output_path = os.path.join(model_path, 'output.txt')
    bbox_path = os.path.join(model_path, 'bbox')
    if not os.path.isdir(bbox_path):
        os.makedirs(bbox_path)
    normal_img_dir = os.path.join(bbox_path, 'normal')
    adv_img_dir = os.path.join(bbox_path, 'attack')
    filter_img_dir = os.path.join(bbox_path, 'filter')

    if not os.path.isdir(normal_img_dir):
        os.makedirs(normal_img_dir)
    if not os.path.isdir(adv_img_dir):
        os.makedirs(adv_img_dir)
    if not os.path.isdir(filter_img_dir):
        os.makedirs(filter_img_dir)

    for v_idx, video in enumerate(dataset):
        if v_idx == 4:
            break

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_filter = []
        pred_bboxes_normal = []
        pred_bboxes_adv = []
        scores_filter = []
        scores_normal = []
        scores_adv = []
        track_times = []
        iou_v_f_normal = []
        iou_v_f_attack = []
        iou_v_normal_attack = []

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):

            print('Testing ', video.name, ' img ', idx)
            # squeeze color
            # img = squeeze_color(img, 4)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox_filter = gt_bbox_
                pred_bbox_normal = gt_bbox_
                pred_bbox_adv = gt_bbox_

                # median_filter
                img_filter = median_filter(img, filter_size)
                pred_img = img
                tracker_filter.init(img_filter, gt_bbox_)
                tracker_normal.init(img, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                # iou_v.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes_normal.append([1])
                else:
                    pred_bboxes_filter.append(gt_bbox_)
                    pred_bboxes_normal.append(gt_bbox_)
                    pred_bboxes_adv.append(gt_bbox_)
            else:

                # attack model
                attack_img = img.copy()
                # need to test hard copy or not
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv, x_crop = tracker_attack.track_adv(
                    attack_img, zhanbi, AdA)
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])
                a = torch.squeeze(x_crop_adv)
                # print(a.shape)
                a = torch.permute(a, (1, 2, 0))
                # print(a.shape)
                x_crop_adv = a.detach().cpu().numpy()
                # print('org_patch_size_adv',org_patch_size_adv)
                x_crop_adv = cv2.resize(x_crop_adv, (org_patch_size_adv[0], org_patch_size_adv[1]))

                # image under attack
                attack_img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3], :] = x_crop_adv[
                                                                                                      context_2_adv[0]:
                                                                                                      context_2_adv[1],
                                                                                                      context_2_adv[2]:
                                                                                                      context_2_adv[3],
                                                                                                      :]

                # save image
                img_save_path = os.path.join(img_dir, 'img')
                if not os.path.isdir(img_save_path):
                    os.makedirs(img_save_path)
                img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
                cv2.imwrite(img_name, attack_img)
                # median_filter
                attack_img_filter = median_filter(attack_img, filter_size)

                # input is normal image, feed into filter model and normal model
                # filter model
                # x_crop_filter, scale_z, x_crop_filter_index = tracker_filter.get_x_crop(img_filter)
                x_crop_filter, scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(
                    attack_img_filter)
                outputs = tracker_filter.track(attack_img_filter, x_crop_filter)

                pred_bbox_filter = outputs['bbox']
                pred_bboxes_filter.append(pred_bbox_filter)
                scores_filter.append(outputs['best_score'])

                # normal model
                # x_crop, scale_z, x_crop_normal_index = tracker_filter.get_x_crop(img)
                x_crop, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(
                    attack_img)
                outputs = tracker_normal.track(attack_img, x_crop)

                pred_bbox_normal = outputs['bbox']
                pred_bboxes_normal.append(pred_bbox_normal)
                scores_normal.append(outputs['best_score'])

                # attack model
                # zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # outputs_adv, x_crop_adv, x_crop = tracker_attack.track_adv(img, zhanbi, AdA)
                # pred_bbox_adv = outputs_adv['bbox']
                # pred_bboxes_adv.append(pred_bbox_adv)
                # scores_adv.append(outputs_adv['best_score'])

                # calculate iou
                bbox_filter = np.array(pred_bbox_filter)
                bbox_filter = np.expand_dims(bbox_filter, axis=0)

                bbox_normal = np.array(pred_bbox_normal)
                bbox_normal = np.expand_dims(bbox_normal, axis=0)

                bbox_adv = np.array(pred_bbox_adv)
                bbox_adv = np.expand_dims(bbox_adv, axis=0)

                # filter vs. normal
                iou_img_f_normal = overlap_ratio(bbox_filter, bbox_normal)
                iou_v_f_normal.append(iou_img_f_normal[0])
                # filter vs. attack
                iou_img_f_att = overlap_ratio(bbox_filter, bbox_adv)
                iou_v_f_attack.append(iou_img_f_att[0])
                # normal vs. attack
                iou_img_normal_att = overlap_ratio(bbox_normal, bbox_adv)
                iou_v_normal_attack.append(iou_img_normal_att[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

            if idx > 0:
                # draw normal prediction in yellow color
                pred_bbox_normal = list(map(int, pred_bbox_normal))
                cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
                              (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]),
                              (0, 255, 255), 3)

                # draw attack prediction in red color
                pred_bbox_adv = list(map(int, pred_bbox_adv))
                cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
                              (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255),
                              3)

                # draw filter prediction in blue color
                pred_bbox_filter = list(map(int, pred_bbox_filter))
                cv2.rectangle(img, (pred_bbox_filter[0], pred_bbox_filter[1]),
                              (pred_bbox_filter[0] + pred_bbox_filter[2], pred_bbox_filter[1] + pred_bbox_filter[3]),
                              (255, 0, 0), 3)

                # draw ground truth in green color
                if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # save image
                img_save_path = os.path.join(img_dir, 'img')
                if not os.path.isdir(img_save_path):
                    os.makedirs(img_save_path)
                img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
                # cv2.imwrite(img_name, img)
                #
                # cv2.imshow(video.name, img)
                # if video.name == 'basketball player1' and idx == 225:
                #     cv2.waitKey(100)
                # cv2.waitKey(1)

        toc /= cv2.getTickFrequency()

        tmp = sys.stdout
        sys.stdout = open(output_path, 'a')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

        sys.stdout.close()  # <-comment out to see the output
        sys.stdout = tmp

        # print('iou_v_f_normal', type(iou_v_f_normal))
        # print('iou_img_f_att', type(iou_img_f_att))
        # print('iou_v_normal_attack', type(iou_v_normal_attack))
        # stop

        # save results
        r_path_1 = os.path.join(img_dir, 'filter_normal')
        if not os.path.isdir(r_path_1):
            os.makedirs(r_path_1)
        iou_v_f_normal_path = os.path.join(r_path_1, 'iou_f_n.json')
        # iou_v_f_normal = iou_v_f_normal.tolist()
        json.dump(iou_v_f_normal, open(iou_v_f_normal_path, 'w'))

        score_diff = np.array(scores_filter) - np.array(scores_normal)
        score_diff_path = os.path.join(r_path_1, 'score_diff_f_n.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        r_path_2 = os.path.join(img_dir, 'filter_attack')
        if not os.path.isdir(r_path_2):
            os.makedirs(r_path_2)

        iou_img_f_att_path = os.path.join(r_path_2, 'iou_f_a.json')
        json.dump(iou_v_f_attack, open(iou_img_f_att_path, 'w'))

        score_diff = np.array(scores_filter) - np.array(scores_adv)
        score_diff_path = os.path.join(r_path_2, 'score_diff_f_a.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        r_path_3 = os.path.join(img_dir, 'normal_attack')
        if not os.path.isdir(r_path_3):
            os.makedirs(r_path_3)

        iou_v_normal_attack_path = os.path.join(r_path_3, 'iou_n_a.json')
        json.dump(iou_v_normal_attack, open(iou_v_normal_attack_path, 'w'))

        score_diff = np.array(scores_normal) - np.array(scores_adv)
        score_diff_path = os.path.join(r_path_3, 'score_diff_n_a.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        # save bbox

        result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))
        result_path_filter = os.path.join(filter_img_dir, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        with open(result_path_normal, 'w') as f:
            for x in pred_bboxes_normal:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write attack prediction bounding box
        with open(result_path_adv, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write filter prediction bounding box
        with open(result_path_filter, 'w') as f:
            for x in pred_bboxes_filter:
                f.write(','.join([str(i) for i in x]) + '\n')

    # else:
    #     for v_idx, video in enumerate(dataset):
    #         if v_idx == 4:
    #             break
    #
    #         if args.video != '':
    #             # test one special video
    #             if video.name != args.video:
    #                 continue
    #         toc = 0
    #         pred_bboxes_adv = []
    #         pred_bboxes = []
    #         scores_adv = []
    #         scores = []
    #         track_times = []
    #         iou_v = []
    #
    #         model_path = os.path.join('results', args.dataset, model_name)
    #         if not os.path.isdir(model_path):
    #             os.makedirs(model_path)
    #
    #         img_dir = os.path.join(model_path, video.name)
    #
    #         if not os.path.isdir(img_dir):
    #             os.makedirs(img_dir)
    #
    #         for idx, (img, gt_bbox) in enumerate(video):
    #             # squeeze color
    #             # img = squeeze_color(img, 4)
    #
    #             # remember to calculate the running time separately for normal and attack model
    #             tic = cv2.getTickCount()
    #             if idx == 0:
    #                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    #                 gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    #                 pred_bbox = gt_bbox_
    #                 pred_bbox_adv = gt_bbox_
    #                 pred_img = img
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #                 tracker_filter.init(img_filter, gt_bbox_)
    #                 tracker_attack.init(img, gt_bbox_)
    #                 # iou_v.append(None)
    #                 if 'VOT2018-LT' == args.dataset:
    #                     pred_bboxes.append([1])
    #                 else:
    #                     pred_bboxes.append(pred_bbox)
    #                     pred_bboxes_adv.append(pred_bbox)
    #             else:
    #
    #                 # input is an image, feed into filter model and attack model
    #
    #                 # attack model
    #                 zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
    #                 outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(img, zhanbi, AdA)
    #                 pred_bbox_adv = outputs_adv['bbox']
    #                 pred_bboxes_adv.append(pred_bbox_adv)
    #                 scores_adv.append(outputs_adv['best_score'])
    #                 #print(img.shape)
    #                 #print(x_crop_adv.shape)
    #                 a = torch.squeeze(x_crop_adv)
    #                 #print(a.shape)
    #                 a = torch.permute(a, (1,2,0))
    #                 #print(a.shape)
    #                 x_crop_adv = a.detach().cpu().numpy()
    #                 # print('org_patch_size_adv',org_patch_size_adv)
    #                 x_crop_adv = cv2.resize(x_crop_adv, (org_patch_size_adv[0], org_patch_size_adv[1]))
    #                 # image under attack
    #                 img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3], :] = x_crop_adv[context_2_adv[0]:context_2_adv[1], context_2_adv[2]:context_2_adv[3], :]
    #                 # cv2.imshow('im', img)
    #                 # cv2.waitKey(0)
    #                 #
    #                 # # closing all open windows
    #                 # cv2.destroyAllWindows()
    #                 # stop
    #
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #
    #
    #                 # normal model
    #                 x_crop_filter, scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(img_filter)
    #                 outputs = tracker_filter.track(img, x_crop_filter)
    #
    #                 pred_bbox_normal = outputs['bbox']
    #                 pred_bboxes.append(pred_bbox_normal)
    #                 scores.append(outputs['best_score'])
    #
    #
    #
    #                 bbox_normal = np.array(pred_bbox_normal)
    #                 bbox_normal = np.expand_dims(bbox_normal, axis=0)
    #                 bbox_adv = np.array(pred_bbox_adv)
    #                 bbox_adv = np.expand_dims(bbox_adv, axis=0)
    #
    #                 iou_img = overlap_ratio(bbox_normal, bbox_adv)
    #                 iou_v.append(iou_img[0])
    #
    #
    #             toc += cv2.getTickCount() - tic
    #             track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
    #             if idx == 0:
    #                 cv2.destroyAllWindows()
    #             # if args.vis and idx > 0:
    #             if idx > 0:
    #             # save image
    #                 tracker_normal.my_save_img(img, img_filter, img_dir, idx)
    #
    #         toc /= cv2.getTickFrequency()
    #
    #         # save results
    #         iou_path = os.path.join(img_dir, 'iou.json')
    #         json.dump(iou_v, open(iou_path, 'w'))
    #
    #         score_diff = np.array(scores) - np.array(scores_adv)
    #         score_diff_path = os.path.join(img_dir, 'score_diff.json')
    #         score_diff = score_diff.tolist()
    #         json.dump(score_diff, open(score_diff_path, 'w'))
    #
    #         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
    #             v_idx + 1, video.name, toc, idx / toc))


# filter crop image
def filter_crop_val(filter_size):
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_filter = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()
    model_normal = ModelBuilderAPN()

    # model filter
    model_filter = load_pretrain(model_filter, args.snapshot).cuda().eval()
    tracker_filter = SiamAPNTracker(model_filter)

    # model attack
    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    # model normal
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)
    dataset_root = '/media/mengjie/Data/Downloads/val'
    # create dataset
    # dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                         dataset_root=dataset_root,
    #                                         load_img=False)

    dataset = os.listdir(dataset_root)
    dataset.remove('list.txt')

    # model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    model_name = 'val_filter' + str(filter_size) + '_attack'

    model_path = os.path.join('results', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    output_path = os.path.join(model_path, 'output.txt')
    bbox_path = os.path.join(model_path, 'bbox')
    if not os.path.isdir(bbox_path):
        os.makedirs(bbox_path)
    normal_img_dir = os.path.join(bbox_path, 'normal')
    adv_img_dir = os.path.join(bbox_path, 'attack')
    filter_img_dir = os.path.join(bbox_path, 'filter')

    if not os.path.isdir(normal_img_dir):
        os.makedirs(normal_img_dir)
    if not os.path.isdir(adv_img_dir):
        os.makedirs(adv_img_dir)
    if not os.path.isdir(filter_img_dir):
        os.makedirs(filter_img_dir)

    for v_idx, video in enumerate(dataset):
        # if v_idx == 4:
        #     break

        # if args.video != '':
        #     # test one special video
        #     if video.name != args.video:
        #         continue
        toc = 0
        pred_bboxes_filter = []
        pred_bboxes_normal = []
        pred_bboxes_adv = []
        scores_filter = []
        scores_normal = []
        scores_adv = []
        track_times = []
        iou_v_f_normal = []
        iou_v_f_attack = []
        iou_v_normal_attack = []

        img_dir = os.path.join(model_path, video)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        img_paths = sorted(glob.glob(os.path.join(video, '*.jpg')))
        json_path = os.path.join(video, 'zhanbi.json')
        gt_path = os.path.join(video, 'groundtruth.txt')
        with open(gt_path) as f:
            gt_video = f.read().splitlines()

        for idx, gt_line in enumerate(gt_video):
            gt_bbox = gt_line.strip().split(',')
            img = cv2.imread(img_paths[idx])
            if idx % 20 == 0:
                print('Testing ', video, ' img ', idx)
            # squeeze color
            # img = squeeze_color(img, 4)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox_filter = gt_bbox_
                # pred_bbox_normal = gt_bbox_
                pred_bbox_adv = gt_bbox_

                # median_filter
                img_filter = median_filter(img, filter_size)
                pred_img = img
                tracker_filter.init(img_filter, gt_bbox_)
                tracker_normal.init(img, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                # iou_v.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes_normal.append([1])
                else:
                    pred_bboxes_filter.append(gt_bbox_)
                    # pred_bboxes_normal.append(gt_bbox_)
                    pred_bboxes_adv.append(gt_bbox_)
            else:

                # attack model
                # need to test hard copy or not
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(
                    img, zhanbi, AdA)
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])

                # median_filter
                x_crop_adv = x_crop_adv.cpu().detach().numpy()
                x_crop_adv_filter = median_filter(x_crop_adv, filter_size)
                x_crop_adv_filter = torch.from_numpy(x_crop_adv_filter)
                x_crop_adv_filter = x_crop_adv_filter.cuda()

                # input is normal image, feed into filter model and normal model
                # filter model
                # x_crop_filter, scale_z, x_crop_filter_index = tracker_filter.get_x_crop(img_filter)
                # x_crop_filter , scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(attack_img_filter)
                outputs = tracker_filter.track(img, x_crop_adv_filter)

                pred_bbox_filter = outputs['bbox']
                pred_bboxes_filter.append(pred_bbox_filter)
                scores_filter.append(outputs['best_score'])

                # normal model
                # x_crop, scale_z, x_crop_normal_index = tracker_filter.get_x_crop(img)
                # x_crop, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(attack_img)
                # outputs = tracker_normal.track(attack_img, x_crop)

                # pred_bbox_normal = outputs['bbox']
                # pred_bboxes_normal.append(pred_bbox_normal)
                # scores_normal.append(outputs['best_score'])

                # attack model
                # zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # outputs_adv, x_crop_adv, x_crop = tracker_attack.track_adv(img, zhanbi, AdA)
                # pred_bbox_adv = outputs_adv['bbox']
                # pred_bboxes_adv.append(pred_bbox_adv)
                # scores_adv.append(outputs_adv['best_score'])

                # calculate iou
                bbox_filter = np.array(pred_bbox_filter)
                bbox_filter = np.expand_dims(bbox_filter, axis=0)

                # bbox_normal = np.array(pred_bbox_normal)
                # bbox_normal = np.expand_dims(bbox_normal, axis=0)

                bbox_adv = np.array(pred_bbox_adv)
                bbox_adv = np.expand_dims(bbox_adv, axis=0)

                # filter vs. normal
                # iou_img_f_normal = overlap_ratio(bbox_filter, bbox_normal)
                # iou_v_f_normal.append(iou_img_f_normal[0])
                # filter vs. attack
                iou_img_f_att = overlap_ratio(bbox_filter, bbox_adv)
                iou_v_f_attack.append(iou_img_f_att[0])
                # normal vs. attack
                # iou_img_normal_att = overlap_ratio(bbox_normal, bbox_adv)
                # iou_v_normal_attack.append(iou_img_normal_att[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

            # if idx > 0:
            #     # draw normal prediction in yellow color
            #     # pred_bbox_normal = list(map(int, pred_bbox_normal))
            #     # cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
            #     #               (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]), (0, 255, 255), 3)
            #
            #     # draw attack prediction in red color
            #     pred_bbox_adv = list(map(int, pred_bbox_adv))
            #     cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
            #                   (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255), 3)
            #
            #     # draw filter prediction in blue color
            #     pred_bbox_filter = list(map(int, pred_bbox_filter))
            #     cv2.rectangle(img, (pred_bbox_filter[0], pred_bbox_filter[1]),
            #                   (pred_bbox_filter[0] + pred_bbox_filter[2], pred_bbox_filter[1] + pred_bbox_filter[3]), (255, 0, 0), 3)
            #
            #
            #     # draw ground truth in green color
            #     if not np.isnan(gt_bbox).any():
            #         gt_bbox = list(map(int, gt_bbox))
            #         cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
            #                       (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
            #
            #     cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #     # save image
            #     img_save_path = os.path.join(img_dir, 'img')
            #     if not os.path.isdir(img_save_path):
            #         os.makedirs(img_save_path)
            #     img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
            #     cv2.imwrite(img_name, img)
            #
            # cv2.imshow(video.name, img)
            # if video.name == 'basketball player1' and idx == 225:
            #     cv2.waitKey(100)
            # cv2.waitKey(1)

        toc /= cv2.getTickFrequency()

        tmp = sys.stdout
        sys.stdout = open(output_path, 'a')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

        sys.stdout.close()  # <-comment out to see the output
        sys.stdout = tmp

        # print('iou_v_f_normal', type(iou_v_f_normal))
        # print('iou_img_f_att', type(iou_img_f_att))
        # print('iou_v_normal_attack', type(iou_v_normal_attack))
        # stop

        # save results
        # r_path_1 = os.path.join(img_dir, 'filter_normal')
        # if not os.path.isdir(r_path_1):
        #     os.makedirs(r_path_1)
        # iou_v_f_normal_path = os.path.join(r_path_1, 'iou_f_n.json')
        # # iou_v_f_normal = iou_v_f_normal.tolist()
        # json.dump(iou_v_f_normal, open(iou_v_f_normal_path, 'w'))
        #
        # score_diff = np.array(scores_filter) - np.array(scores_normal)
        # score_diff_path = os.path.join(r_path_1, 'score_diff_f_n.json')
        # score_diff = score_diff.tolist()
        # json.dump(score_diff, open(score_diff_path, 'w'))

        r_path_2 = os.path.join(img_dir, 'filter_attack')
        if not os.path.isdir(r_path_2):
            os.makedirs(r_path_2)

        iou_img_f_att_path = os.path.join(r_path_2, 'iou_f_a.json')
        json.dump(iou_v_f_attack, open(iou_img_f_att_path, 'w'))

        score_diff = np.array(scores_filter) - np.array(scores_adv)
        score_diff_path = os.path.join(r_path_2, 'score_diff_f_a.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        # r_path_3 = os.path.join(img_dir, 'normal_attack')
        # if not os.path.isdir(r_path_3):
        #     os.makedirs(r_path_3)
        #
        # iou_v_normal_attack_path = os.path.join(r_path_3, 'iou_n_a.json')
        # json.dump(iou_v_normal_attack, open(iou_v_normal_attack_path, 'w'))
        #
        # score_diff = np.array(scores_normal) - np.array(scores_adv)
        # score_diff_path = os.path.join(r_path_3, 'score_diff_n_a.json')
        # score_diff = score_diff.tolist()
        # json.dump(score_diff, open(score_diff_path, 'w'))

        # save bbox

        # result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video))
        result_path_filter = os.path.join(filter_img_dir, '{}.txt'.format(video))

        # write normal prediction bounding box
        # with open(result_path_normal, 'w') as f:
        #     for x in pred_bboxes_normal:
        #         f.write(','.join([str(i) for i in x]) + '\n')

        # write attack prediction bounding box
        with open(result_path_adv, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write filter prediction bounding box

        with open(result_path_filter, 'w') as f:
            for x in pred_bboxes_filter:
                f.write(','.join([str(i) for i in x]) + '\n')

    # else:
    #     for v_idx, video in enumerate(dataset):
    #         if v_idx == 4:
    #             break
    #
    #         if args.video != '':
    #             # test one special video
    #             if video.name != args.video:
    #                 continue
    #         toc = 0
    #         pred_bboxes_adv = []
    #         pred_bboxes = []
    #         scores_adv = []
    #         scores = []
    #         track_times = []
    #         iou_v = []
    #
    #         model_path = os.path.join('results', args.dataset, model_name)
    #         if not os.path.isdir(model_path):
    #             os.makedirs(model_path)
    #
    #         img_dir = os.path.join(model_path, video.name)
    #
    #         if not os.path.isdir(img_dir):
    #             os.makedirs(img_dir)
    #
    #         for idx, (img, gt_bbox) in enumerate(video):
    #             # squeeze color
    #             # img = squeeze_color(img, 4)
    #
    #             # remember to calculate the running time separately for normal and attack model
    #             tic = cv2.getTickCount()
    #             if idx == 0:
    #                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    #                 gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    #                 pred_bbox = gt_bbox_
    #                 pred_bbox_adv = gt_bbox_
    #                 pred_img = img
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #                 tracker_filter.init(img_filter, gt_bbox_)
    #                 tracker_attack.init(img, gt_bbox_)
    #                 # iou_v.append(None)
    #                 if 'VOT2018-LT' == args.dataset:
    #                     pred_bboxes.append([1])
    #                 else:
    #                     pred_bboxes.append(pred_bbox)
    #                     pred_bboxes_adv.append(pred_bbox)
    #             else:
    #
    #                 # input is an image, feed into filter model and attack model
    #
    #                 # attack model
    #                 zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
    #                 outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(img, zhanbi, AdA)
    #                 pred_bbox_adv = outputs_adv['bbox']
    #                 pred_bboxes_adv.append(pred_bbox_adv)
    #                 scores_adv.append(outputs_adv['best_score'])
    #                 #print(img.shape)
    #                 #print(x_crop_adv.shape)
    #                 a = torch.squeeze(x_crop_adv)
    #                 #print(a.shape)
    #                 a = torch.permute(a, (1,2,0))
    #                 #print(a.shape)
    #                 x_crop_adv = a.detach().cpu().numpy()
    #                 # print('org_patch_size_adv',org_patch_size_adv)
    #                 x_crop_adv = cv2.resize(x_crop_adv, (org_patch_size_adv[0], org_patch_size_adv[1]))
    #                 # image under attack
    #                 img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3], :] = x_crop_adv[context_2_adv[0]:context_2_adv[1], context_2_adv[2]:context_2_adv[3], :]
    #                 # cv2.imshow('im', img)
    #                 # cv2.waitKey(0)
    #                 #
    #                 # # closing all open windows
    #                 # cv2.destroyAllWindows()
    #                 # stop
    #
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #
    #
    #                 # normal model
    #                 x_crop_filter, scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(img_filter)
    #                 outputs = tracker_filter.track(img, x_crop_filter)
    #
    #                 pred_bbox_normal = outputs['bbox']
    #                 pred_bboxes.append(pred_bbox_normal)
    #                 scores.append(outputs['best_score'])
    #
    #
    #
    #                 bbox_normal = np.array(pred_bbox_normal)
    #                 bbox_normal = np.expand_dims(bbox_normal, axis=0)
    #                 bbox_adv = np.array(pred_bbox_adv)
    #                 bbox_adv = np.expand_dims(bbox_adv, axis=0)
    #
    #                 iou_img = overlap_ratio(bbox_normal, bbox_adv)
    #                 iou_v.append(iou_img[0])
    #
    #
    #             toc += cv2.getTickCount() - tic
    #             track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
    #             if idx == 0:
    #                 cv2.destroyAllWindows()
    #             # if args.vis and idx > 0:
    #             if idx > 0:
    #             # save image
    #                 tracker_normal.my_save_img(img, img_filter, img_dir, idx)
    #
    #         toc /= cv2.getTickFrequency()
    #
    #         # save results
    #         iou_path = os.path.join(img_dir, 'iou.json')
    #         json.dump(iou_v, open(iou_path, 'w'))
    #
    #         score_diff = np.array(scores) - np.array(scores_adv)
    #         score_diff_path = os.path.join(img_dir, 'score_diff.json')
    #         score_diff = score_diff.tolist()
    #         json.dump(score_diff, open(score_diff_path, 'w'))
    #
    #         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
    #             v_idx + 1, video.name, toc, idx / toc))


# input is the full image
def normal_test(filter_size):
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_filter = ModelBuilderAPN()
    model_normal = ModelBuilderAPN()

    # model filter
    model_filter = load_pretrain(model_filter, args.snapshot).cuda().eval()
    tracker_filter = SiamAPNTracker(model_filter)

    # model normal
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    model_name = 'filter' + str(filter_size) + '_normal'

    model_path = os.path.join('results', args.dataset, model_name)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    output_path = os.path.join(model_path, 'output.txt')
    bbox_path = os.path.join(model_path, 'bbox')
    if not os.path.isdir(bbox_path):
        os.makedirs(bbox_path)
    normal_img_dir = os.path.join(bbox_path, 'normal')
    filter_img_dir = os.path.join(bbox_path, 'filter')

    if not os.path.isdir(normal_img_dir):
        os.makedirs(normal_img_dir)
    if not os.path.isdir(filter_img_dir):
        os.makedirs(filter_img_dir)

    for v_idx, video in enumerate(dataset):

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_filter = []
        pred_bboxes_normal = []
        scores_filter = []
        scores_normal = []

        track_times = []
        iou_v_f_normal = []

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):
            if idx % 200 == 0:
                print('Testing ', video.name, ' img ', idx)
            # squeeze color
            # img = squeeze_color(img, 4)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox_filter = gt_bbox_
                pred_bbox_normal = gt_bbox_

                # median_filter
                img_filter = median_filter(img, filter_size)

                pred_img = img
                tracker_filter.init(img_filter, gt_bbox_)
                # tracker_filter.init(img, gt_bbox_)
                tracker_normal.init(img, gt_bbox_)

                # iou_v.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes_normal.append([1])
                else:
                    pred_bboxes_filter.append(gt_bbox_)
                    pred_bboxes_normal.append(gt_bbox_)
            else:

                # median_filter
                # img_filter = median_filter(img, filter_size)

                # normal model
                # x_crop, scale_z, x_crop_normal_index = tracker_filter.get_x_crop(img)
                x_crop, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(
                    img)
                outputs = tracker_normal.track(img, x_crop)

                pred_bbox_normal = outputs['bbox']
                pred_bboxes_normal.append(pred_bbox_normal)
                scores_normal.append(outputs['best_score'])

                # input is normal image, feed into filter model and normal model
                # filter model

                # median_filter
                x_crop = x_crop.cpu().detach().numpy()
                x_crop_filter = median_filter(x_crop, filter_size)
                x_crop_filter = torch.from_numpy(x_crop_filter)
                x_crop_filter = x_crop_filter.cuda()

                # x_crop_filter, scale_z, x_crop_filter_index = tracker_filter.get_x_crop(img_filter)
                # x_crop_filter , scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(img_filter)
                # outputs = tracker_filter.track(img_filter, x_crop_filter)
                outputs = tracker_filter.track(img, x_crop_filter)

                pred_bbox_filter = outputs['bbox']
                pred_bboxes_filter.append(pred_bbox_filter)
                scores_filter.append(outputs['best_score'])

                # attack model
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                outputs_adv, x_crop_adv, x_crop = tracker_attack.track_adv(img, zhanbi, AdA)
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])

                # calculate iou
                bbox_filter = np.array(pred_bbox_filter)
                bbox_filter = np.expand_dims(bbox_filter, axis=0)

                bbox_normal = np.array(pred_bbox_normal)
                bbox_normal = np.expand_dims(bbox_normal, axis=0)

                # filter vs. normal
                iou_img_f_normal = overlap_ratio(bbox_filter, bbox_normal)
                iou_v_f_normal.append(iou_img_f_normal[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

            # if idx > 0:
            #     # draw normal prediction in yellow color
            #     pred_bbox_normal = list(map(int, pred_bbox_normal))
            #     cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
            #                   (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]), (0, 255, 255), 3)
            #
            #
            #     # draw filter prediction in blue color
            #     pred_bbox_filter = list(map(int, pred_bbox_filter))
            #     cv2.rectangle(img, (pred_bbox_filter[0], pred_bbox_filter[1]),
            #                  (pred_bbox_filter[0] + pred_bbox_filter[2], pred_bbox_filter[1] + pred_bbox_filter[3]), (255, 0, 0), 3)
            #
            #
            #     # draw ground truth in green color
            #     if not np.isnan(gt_bbox).any():
            #        gt_bbox = list(map(int, gt_bbox))
            #        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
            #                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
            #
            #     cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #     # save image
            #     # save image
            #     img_save_path = os.path.join(img_dir, 'img')
            #     if not os.path.isdir(img_save_path):
            #        os.makedirs(img_save_path)
            #     img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
            #     cv2.imwrite(img_name, img)

            #
            # cv2.imshow(video.name, img)
            # if video.name == 'basketball player1' and idx == 225:
            #     cv2.waitKey(100)
            # cv2.waitKey(1)

        toc /= cv2.getTickFrequency()

        tmp = sys.stdout
        sys.stdout = open(output_path, 'a')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

        sys.stdout.close()
        sys.stdout = tmp

        # save results
        r_path_1 = os.path.join(img_dir, 'filter_normal')
        if not os.path.isdir(r_path_1):
            os.makedirs(r_path_1)
        iou_v_f_normal_path = os.path.join(r_path_1, 'iou_f_n.json')
        json.dump(iou_v_f_normal, open(iou_v_f_normal_path, 'w'))

        score_diff = np.array(scores_filter) - np.array(scores_normal)
        score_diff_path = os.path.join(r_path_1, 'score_diff_f_n.json')
        score_diff = score_diff.tolist()
        json.dump(score_diff, open(score_diff_path, 'w'))

        # save bbox

        result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_filter = os.path.join(filter_img_dir, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        with open(result_path_normal, 'w') as f:
            for x in pred_bboxes_normal:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write filter prediction bounding box
        with open(result_path_filter, 'w') as f:
            for x in pred_bboxes_filter:
                f.write(','.join([str(i) for i in x]) + '\n')

    # else:
    #     for v_idx, video in enumerate(dataset):
    #         if v_idx == 4:
    #             break
    #
    #         if args.video != '':
    #             # test one special video
    #             if video.name != args.video:
    #                 continue
    #         toc = 0
    #         pred_bboxes_adv = []
    #         pred_bboxes = []
    #         scores_adv = []
    #         scores = []
    #         track_times = []
    #         iou_v = []
    #
    #         model_path = os.path.join('results', args.dataset, model_name)
    #         if not os.path.isdir(model_path):
    #             os.makedirs(model_path)
    #
    #         img_dir = os.path.join(model_path, video.name)
    #
    #         if not os.path.isdir(img_dir):
    #             os.makedirs(img_dir)
    #
    #         for idx, (img, gt_bbox) in enumerate(video):
    #             # squeeze color
    #             # img = squeeze_color(img, 4)
    #
    #             # remember to calculate the running time separately for normal and attack model
    #             tic = cv2.getTickCount()
    #             if idx == 0:
    #                 cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    #                 gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    #                 pred_bbox = gt_bbox_
    #                 pred_bbox_adv = gt_bbox_
    #                 pred_img = img
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #                 tracker_filter.init(img_filter, gt_bbox_)
    #                 tracker_attack.init(img, gt_bbox_)
    #                 # iou_v.append(None)
    #                 if 'VOT2018-LT' == args.dataset:
    #                     pred_bboxes.append([1])
    #                 else:
    #                     pred_bboxes.append(pred_bbox)
    #                     pred_bboxes_adv.append(pred_bbox)
    #             else:
    #
    #                 # input is an image, feed into filter model and attack model
    #
    #                 # attack model
    #                 zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
    #                 outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(img, zhanbi, AdA)
    #                 pred_bbox_adv = outputs_adv['bbox']
    #                 pred_bboxes_adv.append(pred_bbox_adv)
    #                 scores_adv.append(outputs_adv['best_score'])
    #                 #print(img.shape)
    #                 #print(x_crop_adv.shape)
    #                 a = torch.squeeze(x_crop_adv)
    #                 #print(a.shape)
    #                 a = torch.permute(a, (1,2,0))
    #                 #print(a.shape)
    #                 x_crop_adv = a.detach().cpu().numpy()
    #                 # print('org_patch_size_adv',org_patch_size_adv)
    #                 x_crop_adv = cv2.resize(x_crop_adv, (org_patch_size_adv[0], org_patch_size_adv[1]))
    #                 # image under attack
    #                 img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3], :] = x_crop_adv[context_2_adv[0]:context_2_adv[1], context_2_adv[2]:context_2_adv[3], :]
    #                 # cv2.imshow('im', img)
    #                 # cv2.waitKey(0)
    #                 #
    #                 # # closing all open windows
    #                 # cv2.destroyAllWindows()
    #                 # stop
    #
    #                 # median_filter
    #                 img_filter = median_filter(img, 3)
    #
    #
    #                 # normal model
    #                 x_crop_filter, scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(img_filter)
    #                 outputs = tracker_filter.track(img, x_crop_filter)
    #
    #                 pred_bbox_normal = outputs['bbox']
    #                 pred_bboxes.append(pred_bbox_normal)
    #                 scores.append(outputs['best_score'])
    #
    #
    #
    #                 bbox_normal = np.array(pred_bbox_normal)
    #                 bbox_normal = np.expand_dims(bbox_normal, axis=0)
    #                 bbox_adv = np.array(pred_bbox_adv)
    #                 bbox_adv = np.expand_dims(bbox_adv, axis=0)
    #
    #                 iou_img = overlap_ratio(bbox_normal, bbox_adv)
    #                 iou_v.append(iou_img[0])
    #
    #
    #             toc += cv2.getTickCount() - tic
    #             track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
    #             if idx == 0:
    #                 cv2.destroyAllWindows()
    #             # if args.vis and idx > 0:
    #             if idx > 0:
    #             # save image
    #                 tracker_normal.my_save_img(img, img_filter, img_dir, idx)
    #
    #         toc /= cv2.getTickFrequency()
    #
    #         # save results
    #         iou_path = os.path.join(img_dir, 'iou.json')
    #         json.dump(iou_v, open(iou_path, 'w'))
    #
    #         score_diff = np.array(scores) - np.array(scores_adv)
    #         score_diff_path = os.path.join(img_dir, 'score_diff.json')
    #         score_diff = score_diff.tolist()
    #         json.dump(score_diff, open(score_diff_path, 'w'))
    #
    #         print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
    #             v_idx + 1, video.name, toc, idx / toc))


def save_images():
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_normal = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()

    # print(args.snapshot)
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    # dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    train_rep = '/media/mengjie/Data/Downloads/crop287_adv'
    # create dataset
    # load got10k dataset
    dataset = GOT10k_dataset(train_rep)
    print('Start Testing')

    for v_idx, video in enumerate(dataset):
        # video = (init_tensor, search_tensor, zhanbi, cur_folder)

        save_path = os.path.join('/media/mengjie/Data/Downloads', 'crop287_adv(test)')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        img_dir = os.path.join(save_path, video[3])

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        AdA.set_input(video)
        AdA.forward()

        tensor_adv = AdA.search_adv255

        for idx in range(len(tensor_adv)):
            img_adv = tensor2img(tensor_adv[idx])
            frame_id = idx + 1
            cv2.imwrite(os.path.join(img_dir, '%08d.jpg' % frame_id), img_adv)


def AdA_R_test(epoch):
    # load config
    # print(args.config)

    threshold = 0.15
    filter_size = 3
    cfg.merge_from_file(args.config)
    model_attack = ModelBuilderAPN()
    model_normal = ModelBuilderAPN()
    model_filter = ModelBuilderAPN()

    # model filter
    model_filter = load_pretrain(model_filter, args.snapshot).cuda().eval()
    tracker_filter = SiamAPNTracker(model_filter)

    # model attack
    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    # model normal
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    model_name = 'AdA_R_' + str(epoch)

    model_path = os.path.join('results', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    output_path = os.path.join(model_path, 'output.txt')
    bbox_path = os.path.join(model_path, 'bbox')
    if not os.path.isdir(bbox_path):
        os.makedirs(bbox_path)
    normal_img_dir = os.path.join(bbox_path, 'normal')
    rec_img_dir = os.path.join(bbox_path, 'recovery')

    if not os.path.isdir(normal_img_dir):
        os.makedirs(normal_img_dir)
    if not os.path.isdir(rec_img_dir):
        os.makedirs(rec_img_dir)

    normal_idx_path = os.path.join(model_path, 'normal_idx')
    if not os.path.isdir(normal_idx_path):
        os.makedirs(normal_idx_path)

    for v_idx, video in enumerate(dataset):
        start = random.randint(5, 50)
        flag = True
        # if v_idx <= 4:
        #     continue

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_normal = []
        pred_bboxes_adv = []
        pred_bboxes_rec = []
        pred_bboxes_filter = []
        scores_filter = []
        scores_adv = []
        scores_normal = []
        scores_rec = []
        track_times = []
        iou_v_f_normal = []
        iou_v_f_rec = []
        iou_v_normal_attack = []

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        print('Attack to ', video.name, ' will start at frame ', start)
        normal_idx = []

        for idx, (img, gt_bbox) in enumerate(video):

            if idx % 200 == 0:
                print('Testing ', video.name, ' img ', idx)
            # squeeze color
            # img = squeeze_color(img, 4)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox_normal = gt_bbox_
                pred_bbox_adv = gt_bbox_
                pred_bbox_rec = gt_bbox_

                # median_filter
                pred_img = img
                # median_filter
                img_filter = median_filter(img, filter_size)
                tracker_filter.init(img_filter, gt_bbox_)

                tracker_normal.init(img, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                # iou_v.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes_normal.append([1])
                else:
                    pred_bboxes_rec.append(gt_bbox_)
                    pred_bboxes_normal.append(gt_bbox_)
                    pred_bboxes_adv.append(gt_bbox_)
                    pred_bboxes_filter.append(gt_bbox_)
            else:

                rec_full_img = img.copy()
                # attack model
                # need to test hard copy or not
                if idx >= start:
                    zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                    # the last item from output is x_crop before attack
                    outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv, x_crop = tracker_attack.track_adv(
                        img, zhanbi, AdA)
                else:
                    x_crop_adv, scale_z, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.get_x_crop(
                        img)
                    outputs_adv = tracker_attack.track(img, x_crop_adv)

                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                # save score for detection
                # scores_adv.append(outputs_adv['best_score'])

                # median_filter
                x_crop_adv_1 = x_crop_adv.clone()
                x_crop_adv_1 = x_crop_adv_1.cpu().detach().numpy()
                x_crop_adv_filter = median_filter(x_crop_adv_1, filter_size)
                x_crop_adv_filter = torch.from_numpy(x_crop_adv_filter)
                x_crop_adv_filter = x_crop_adv_filter.cuda()

                # input is normal image, feed into filter model and normal model
                # filter model
                # x_crop_filter, scale_z, x_crop_filter_index = tracker_filter.get_x_crop(img_filter)
                # x_crop_filter , scale_z, context_2_filter, contex_1_filter, org_patch_size_filter = tracker_filter.get_x_crop(attack_img_filter)
                outputs_filter = tracker_filter.track(img, x_crop_adv_filter)

                # pred_bbox_filter = outputs['bbox']
                # pred_bboxes_filter.append(pred_bbox_filter)

                # anomaly detection
                if abs(outputs_adv['best_score'] - outputs_filter['best_score']) > threshold:

                    # go to recovery networks
                    x_crop_adv_tensor = normalize(x_crop_adv)
                    with torch.no_grad():
                        AdA_R.search_attack1 = x_crop_adv_tensor
                        AdA_R.num_search = x_crop_adv_tensor.size(0)
                        AdA_R.zhanbi = zhanbi
                        AdA_R.forward_R((287, 287))

                    img_rec = AdA_R.search_rec255

                    # save crop image
                    img_rec_2 = tensor2img(img_rec)
                    frame_id = idx
                    cv2.imwrite(os.path.join(img_dir, '%08d_recovery.jpg' % frame_id), img_rec_2)

                    # save full recovery image
                    a = torch.squeeze(img_rec)
                    # print(a.shape)
                    a = torch.permute(a, (1, 2, 0))
                    # print(a.shape)
                    x_crop_rec = a.detach().cpu().numpy()
                    # print('org_patch_size_adv',org_patch_size_adv)
                    x_crop_rec = cv2.resize(x_crop_rec, (org_patch_size_adv[0], org_patch_size_adv[1]))

                    # full image recovery
                    rec_full_img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3], :] = x_crop_rec[
                                                                                                            context_2_adv[
                                                                                                                0]:
                                                                                                            context_2_adv[
                                                                                                                1],
                                                                                                            context_2_adv[
                                                                                                                2]:
                                                                                                            context_2_adv[
                                                                                                                3],
                                                                                                            :]

                    # save image
                    img_save_path = os.path.join(img_dir, 'img_recover')
                    if not os.path.isdir(img_save_path):
                        os.makedirs(img_save_path)
                    img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
                    cv2.imwrite(img_name, rec_full_img)

                    # feed recovery image to tracker
                    if flag:
                        tracker_normal.center_pos = tracker_attack.center_pos
                        tracker_normal.size = tracker_attack.size
                        flag = False
                    x_crop_rec_new, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(
                        rec_full_img)
                    outputs_rec = tracker_normal.track(rec_full_img, x_crop_rec_new)
                    pred_bbox_rec = outputs_rec['bbox']
                    pred_bboxes_rec.append(pred_bbox_rec)

                    img_rec_2 = tensor2img(x_crop_rec_new)
                    frame_id = idx
                    cv2.imwrite(os.path.join(img_dir, '%08d_recovery_new.jpg' % frame_id), img_rec_2)

                    x_crop_adv_2 = tensor2img(x_crop_adv)
                    cv2.imwrite(os.path.join(img_dir, '%08d_adv.jpg' % frame_id), x_crop_adv_2)


                else:
                    pred_bbox_rec = pred_bbox_adv
                    pred_bboxes_rec.append(pred_bbox_adv)
                    normal_idx.append(idx)

                # normal model
                # x_crop, scale_z, x_crop_normal_index = tracker_filter.get_x_crop(img)
                # x_crop_normal, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(img)
                # outputs_normal = tracker_normal.track(img, x_crop_normal)
                #
                # pred_bbox_normal = outputs_normal['bbox']
                # pred_bboxes_normal.append(pred_bbox_normal)
                # scores_normal.append(outputs['best_score'])

                # calculate iou
                # bbox_rec = np.array(pred_bbox_rec)
                # bbox_rec = np.expand_dims(bbox_rec, axis=0)
                #
                # bbox_normal = np.array(pred_bbox_normal)
                # bbox_normal = np.expand_dims(bbox_normal, axis=0)

                # bbox_adv = np.array(pred_bbox_adv)
                # bbox_adv = np.expand_dims(bbox_adv, axis=0)

                # filter vs. normal
                # iou_img_f_normal = overlap_ratio(bbox_filter, bbox_normal)
                # iou_v_f_normal.append(iou_img_f_normal[0])
                # filter vs. attack
                # iou_img_f_rec = overlap_ratio(bbox_rec, bbox_normal)
                # iou_v_f_rec.append(iou_img_f_rec[0])
                # normal vs. attack
                # iou_img_normal_att = overlap_ratio(bbox_normal, bbox_adv)
                # iou_v_normal_attack.append(iou_img_normal_att[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

            # if idx > 0:
            #     # draw normal prediction in yellow color
            #     # pred_bbox_normal = list(map(int, pred_bbox_normal))
            #     # cv2.rectangle(img, (pred_bbox_normal[0], pred_bbox_normal[1]),
            #     #               (pred_bbox_normal[0] + pred_bbox_normal[2], pred_bbox_normal[1] + pred_bbox_normal[3]), (0, 255, 255), 3)
            #
            #     # draw attack prediction in red color
            #     pred_bbox_adv = list(map(int, pred_bbox_adv))
            #     cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
            #                   (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255),
            #                   3)
            #
            #     # draw recovery prediction in blue color
            #     pred_bbox_rec = list(map(int, pred_bbox_rec))
            #     cv2.rectangle(img, (pred_bbox_rec[0], pred_bbox_rec[1]),
            #                   (pred_bbox_rec[0] + pred_bbox_rec[2], pred_bbox_rec[1] + pred_bbox_rec[3]), (255, 0, 0),
            #                   3)
            #
            #     # draw ground truth in green color
            #     if not np.isnan(gt_bbox).any():
            #         gt_bbox = list(map(int, gt_bbox))
            #         cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
            #                       (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
            #
            #     cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #     # save image
            #     img_save_path = os.path.join(img_dir, 'img')
            #     if not os.path.isdir(img_save_path):
            #         os.makedirs(img_save_path)
            #     img_name = os.path.join(img_save_path, '{}.jpg'.format(idx))
            #     cv2.imwrite(img_name, img)

            # cv2.imshow(video.name, img)
            # if video.name == 'basketball player1' and idx == 225:
            #     cv2.waitKey(100)
            # cv2.waitKey(1)

        toc /= cv2.getTickFrequency()

        tmp = sys.stdout
        sys.stdout = open(output_path, 'a')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

        sys.stdout.close()  # <-comment out to see the output
        sys.stdout = tmp

        # print('iou_v_f_normal', type(iou_v_f_normal))
        # print('iou_img_f_att', type(iou_img_f_att))
        # print('iou_v_normal_attack', type(iou_v_normal_attack))
        # stop

        # save results
        # r_path_1 = os.path.join(img_dir, 'filter_normal')
        # if not os.path.isdir(r_path_1):
        #     os.makedirs(r_path_1)
        # iou_v_f_normal_path = os.path.join(r_path_1, 'iou_f_n.json')
        # # iou_v_f_normal = iou_v_f_normal.tolist()
        # json.dump(iou_v_f_normal, open(iou_v_f_normal_path, 'w'))
        #
        # score_diff = np.array(scores_filter) - np.array(scores_normal)
        # score_diff_path = os.path.join(r_path_1, 'score_diff_f_n.json')
        # score_diff = score_diff.tolist()
        # json.dump(score_diff, open(score_diff_path, 'w'))

        ## need to comment
        # r_path_2 = os.path.join(img_dir, 'rec_normal')
        # if not os.path.isdir(r_path_2):
        #     os.makedirs(r_path_2)
        #
        # iou_img_f_rec_path = os.path.join(r_path_2, 'iou_r_n.json')
        # json.dump(iou_v_f_rec, open(iou_img_f_rec_path, 'w'))

        # score_diff = np.array(scores_filter) - np.array(scores_adv)
        # score_diff_path = os.path.join(r_path_2, 'score_diff_f_a.json')
        # score_diff = score_diff.tolist()
        # json.dump(score_diff, open(score_diff_path, 'w'))

        # r_path_3 = os.path.join(img_dir, 'normal_attack')
        # if not os.path.isdir(r_path_3):
        #     os.makedirs(r_path_3)
        #
        # iou_v_normal_attack_path = os.path.join(r_path_3, 'iou_n_a.json')
        # json.dump(iou_v_normal_attack, open(iou_v_normal_attack_path, 'w'))
        #
        # score_diff = np.array(scores_normal) - np.array(scores_adv)
        # score_diff_path = os.path.join(r_path_3, 'score_diff_n_a.json')
        # score_diff = score_diff.tolist()
        # json.dump(score_diff, open(score_diff_path, 'w'))

        # save bbox

        # result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_rec = os.path.join(rec_img_dir, '{}.txt'.format(video.name))
        # result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        # with open(result_path_normal, 'w') as f:
        #     for x in pred_bboxes_normal:
        #         f.write(','.join([str(i) for i in x]) + '\n')

        # write attack prediction bounding box
        # with open(result_path_adv, 'w') as f:
        #     for x in pred_bboxes_adv:
        #         f.write(','.join([str(i) for i in x]) + '\n')

        # write filter prediction bounding box
        # with open(result_path_normal, 'w') as f:
        #     for x in pred_bboxes_normal:
        #         f.write(','.join([str(i) for i in x]) + '\n')

        with open(result_path_rec, 'w') as f:
            for x in pred_bboxes_rec:
                f.write(','.join([str(i) for i in x]) + '\n')

        result_path_normal_idx = os.path.join(normal_idx_path, '{}.json'.format(video.name))
        json.dump(normal_idx, open(result_path_normal_idx, 'w'))
        start_idx_path = os.path.join(normal_idx_path, '{}_start.json'.format(video.name))
        json.dump(start, open(start_idx_path, 'w'))


if __name__ == '__main__':
    # save_images()
    #filter_test(3)
    filter_crop_val(3)
    stop
    epoch = 50

    AdA_R_test(epoch)

    stop
    filter_size_range = [2, 3, 4, 5, 6]

    for filter_size in filter_size_range:
        # attack test
        if filter_size > 4:
            # attack test
            filter_crop_test(filter_size)
            # normal test
            normal_test(filter_size)
        elif filter_size == 4:
            # normal test
            normal_test(filter_size)
        else:
            continue
