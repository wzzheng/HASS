import os
import pickle
import torch
import copy
from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from pathlib import Path


class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self.data_path = '../data/kitti'
        self.root_path = Path(self.data_path)
        self.dbinfo_path = self.root_path / ('10%_2_6conf-based_trained.pkl')
        with open(self.dbinfo_path, 'rb') as f:
            self.all_db_infos = pickle.load(f)

    def forward(self, batch_dict):
        if self.training:
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]
            un_batch_indices = batch_dict_ema['points'][:, 0].long()
            source_points = batch_dict_ema['points'][:, 1:5]
            un_bs_mask = (un_batch_indices == 1)
            un_pot = source_points[un_bs_mask].unsqueeze(dim=0)

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema,
                                                                            no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)

                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    new_box = pred_dicts[ind]['pred_boxes'].clone()
                    pseudo_label = pred_dicts[ind]['pred_labels']
                    new_label = pred_dicts[ind]['pred_labels'].clone()
                    pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
                    new_sem_score = pred_dicts[ind]['pred_sem_scores'].clone()
                    sample_idx = batch_dict_ema['frame_id'][ind]
                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        continue

                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))
                        
                    newthresh = [0.5,0.25,0.25]
                    new_thresh = torch.tensor(newthresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))
                        
                    valid_inds = pseudo_score > conf_thresh.squeeze()
                    new_valid = pseudo_score > new_thresh.squeeze()
                    
                    valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])
                    new_valid = new_valid * (pseudo_sem_score > 0.6)

                    pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]
                    new_box = new_box[new_valid]
                    new_label = new_label[new_valid]
                    new_sem_score = new_sem_score[new_valid]
                    
                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                        

                new_label = new_label.cpu().numpy()
                cls_names = ['Car', 'Pedestrian', 'Cyclist']
                pred = {}
                pse_box = new_box
                pse_box = pse_box.cpu().numpy()
                unpt = un_pot.cpu().numpy()[0]
                pointslist = []
                pointpath = []
                
                split_dir = self.root_path / 'ImageSets' / ('train_0.10_2' + '.txt')
                sample_id_list = [x.strip().split(' ')[0] for x in open(split_dir).readlines()] if split_dir.exists() else None
                
                if pse_box.shape[0] > 0 and sample_idx not in sample_id_list:
                    for cls in cls_names:
                        num_infos = len(self.all_db_infos[cls])
                        count_infos = 0
                        for x in range(num_infos):
                            if self.all_db_infos[cls][count_infos]['image_idx'] == sample_idx:
                                self.all_db_infos[cls].pop(count_infos)
                                count_infos -= 1
                            count_infos += 1

                    pred['name'] = np.array(cls_names)[new_label - 1]
                    names = pred['name']
                    pred['frame_id'] = sample_idx
                    pred['gt_boxes'] = pse_box

                    pred['score'] = new_sem_score.cpu().numpy()
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(unpt[:, 0:3]), torch.from_numpy(pse_box)
                    ).numpy()

                    database_save_path = self.root_path / ('10%_2_6conf-based_trained')
                    for i in range(pse_box.shape[0]):
                        gt_points = unpt[point_indices[i] > 0]
                        gt_points[:, :3] -= pse_box[i, :3]
                        filename = '%s_%s_%d.bin' % (sample_idx, pred['name'][i], i)
                        filepath = database_save_path / filename
                        pointslist.append(gt_points)
                        pointpath.append(filepath)
                    
                        iou_labels = iou3d_nms_utils.boxes_iou3d_gpu(batch_dict_ema['gt_boxes'][1, ...][:, 0:7],
                                                                     torch.from_numpy(
                                                                         pred['gt_boxes'][i]).cuda().unsqueeze(
                                                                         dim=0)).max()
                        iou_labels = iou_labels.cpu().numpy()
                        db_path = str(filepath.relative_to(self.data_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                   'box3d_lidar': pred['gt_boxes'][i], 'num_points_in_gt': gt_points.shape[0],
                                   'iou': iou_labels,
                                   'score': pred['score'][i]}
                        self.all_db_infos[names[i]].append(db_info)
                    
                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_y_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_y'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['rot_angle'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['scale'][unlabeled_mask, ...]
                )

                pseudo_ious = []
                pseudo_accs = []
                pseudo_fgs = []
                for i, ind in enumerate(unlabeled_mask):
                    # statistics
                    anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        batch_dict['gt_boxes'][ind, ...][:, 0:7],
                        ori_unlabeled_boxes[i, :, 0:7])
                    cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                    unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    cls_pseudo = cls_pseudo[unzero_inds]
                    if len(unzero_inds) > 0:
                        iou_max, asgn = anchor_by_gt_overlap[unzero_inds, :].max(dim=1)
                        pseudo_ious.append(iou_max.unsqueeze(0))
                        acc = (ori_unlabeled_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                        pseudo_accs.append(acc.unsqueeze(0))
                        fg = (iou_max > 0.5).float().sum(dim=0, keepdim=True) / len(unzero_inds)

                        sem_score_fg = (pseudo_sem_score[unzero_inds] * (iou_max > 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max > 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                        sem_score_bg = (pseudo_sem_score[unzero_inds] * (iou_max < 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max < 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                        pseudo_fgs.append(fg)

                        # only for 100% label
                        if self.supervise_mode >= 1:
                            filter = iou_max > 0.3
                            asgn = asgn[filter]
                            batch_dict['gt_boxes'][ind, ...][:] = torch.zeros_like(batch_dict['gt_boxes'][ind, ...][:])
                            batch_dict['gt_boxes'][ind, ...][:len(asgn)] = ori_unlabeled_boxes[i, :].gather(dim=0, index=asgn.unsqueeze(-1).repeat(1, 8))

                            if self.supervise_mode == 2:
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                    else:
                        ones = torch.ones((1), device=unlabeled_mask.device)
                        sem_score_fg = ones
                        sem_score_bg = ones
                        pseudo_ious.append(ones)
                        pseudo_accs.append(ones)
                        pseudo_fgs.append(ones)

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum() + loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss_point = loss_point[labeled_mask, ...].sum()
            loss_rcnn_cls = loss_rcnn_cls[labeled_mask, ...].sum()

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum() + loss_rcnn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]

            tb_dict_['pseudo_ious'] = torch.cat(pseudo_ious, dim=0).mean()
            tb_dict_['pseudo_accs'] = torch.cat(pseudo_accs, dim=0).mean()
            tb_dict_['sem_score_fg'] = sem_score_fg.mean()
            tb_dict_['sem_score_bg'] = sem_score_bg.mean()

            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            ret_dict = {
                'loss': loss
            }
            disp_dict['infos'] = self.all_db_infos
            disp_dict['path'] = self.dbinfo_path
            disp_dict['points'] = pointslist
            disp_dict['filepath'] = pointpath
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))