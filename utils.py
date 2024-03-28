from torchvision import ops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
import torch
import os


class ModelUtils:
    @staticmethod
    def generate_proposals(anchors, offsets, image_size=(416, 416), device=torch.device('cpu')):
        # Generate proposal anchor boxes
        anchors = anchors.to(device)
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

        # apply offsets to anchors to create proposals
        proposals_ = torch.zeros_like(anchors)
        proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
        proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
        proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])
        proposals_ = proposals_.to(device)

        # change format of proposals back from 'cxcywh' to 'xyxy'
        proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')
        proposals = ops.clip_boxes_to_image(proposals, size=image_size)
        return proposals

    @staticmethod
    def gen_anc_centers(out_size):
        out_h, out_w = out_size
        anc_pts_x = torch.arange(0, out_w) + 0.5
        anc_pts_y = torch.arange(0, out_h) + 0.5
        return anc_pts_x, anc_pts_y

    @staticmethod
    def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size, device=torch.device('cpu')):
        n_anc_boxes = len(anc_scales) * len(anc_ratios)
        anc_base = torch.zeros(1, anc_pts_x.size(dim=0),
                               anc_pts_y.size(dim=0),
                               n_anc_boxes,
                               4)  # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

        for ix, xc in enumerate(anc_pts_x):
            for jx, yc in enumerate(anc_pts_y):
                anc_boxes = torch.zeros((n_anc_boxes, 4))
                c = 0
                for i, scale in enumerate(anc_scales):
                    for j, ratio in enumerate(anc_ratios):
                        w = scale * ratio
                        h = scale
                        xmin = xc - w / 2
                        ymin = yc - h / 2
                        xmax = xc + w / 2
                        ymax = yc + h / 2
                        anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                        c += 1
                anc_boxes = anc_boxes.to(device)
                anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
        return anc_base

    @staticmethod
    def _scale_bbox_data(bboxes, width_scale_factor, height_scale_factor, mode):
        if mode == 'a2p':
            # activation map to pixel image
            bboxes[:, :, [0, 2]] *= width_scale_factor
            bboxes[:, :, [1, 3]] *= height_scale_factor
        else:
            # pixel image to activation map
            bboxes[:, :, [0, 2]] /= width_scale_factor
            bboxes[:, :, [1, 3]] /= height_scale_factor
        return bboxes

    @staticmethod
    def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
        # Box format = xyxy
        assert mode in ['a2p', 'p2a']
        if len(bboxes) <= 0:
            return None
        batch_size = bboxes.size(dim=0)
        proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
        invalid_bbox_mask = (proj_bboxes == -1)  # indicating padded bboxes

        proj_bboxes = ModelUtils._scale_bbox_data(bboxes=proj_bboxes, width_scale_factor=width_scale_factor,
                                                  height_scale_factor=height_scale_factor, mode=mode)

        proj_bboxes.masked_fill_(invalid_bbox_mask, -1)  # fill padded bboxes back with -1
        proj_bboxes.resize_as_(bboxes)
        return proj_bboxes

    @staticmethod
    def get_iou_mat(batch_size, all_anc_boxes, gt_bboxes_all, device=torch.device('cpu')):
        anc_boxes_flat = all_anc_boxes.reshape(batch_size, -1, 4)  # flatten
        tot_anc_boxes = anc_boxes_flat.size(dim=1)

        ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))  # placeholder for IoUs
        for i in range(batch_size):
            gt_bboxes = gt_bboxes_all[i]
            anc_boxes = anc_boxes_flat[i]
            anc_boxes, gt_bboxes = anc_boxes.to(device), gt_bboxes.to(device)
            ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
        return ious_mat

    @staticmethod
    def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping, device=torch.device('cpu'), in_fmt='xyxy', out_fmt='cxcywh'):
        pos_anc_coords, gt_bbox_mapping = pos_anc_coords.to(device), gt_bbox_mapping.to(device)
        pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt=in_fmt, out_fmt=out_fmt)
        gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt=in_fmt, out_fmt=out_fmt)

        gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[
                                                                                                        :, 3]
        anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[
                                                                                                         :, 3]
        tx_ = (gt_cx - anc_cx) / anc_w
        ty_ = (gt_cy - anc_cy) / anc_h
        tw_ = torch.log(gt_w / anc_w)
        th_ = torch.log(gt_h / anc_h)
        return torch.stack([tx_, ty_, tw_, th_], dim=-1)

    @staticmethod
    def _get_pos_anchor_box(iou_mat, labels, gt_bboxes_all, all_anc_boxes, max_iou_per_gt_box, pos_thresh, batch_size,
                            gt_box_num, tot_anc_boxes, device=torch.device('cpu')):
        # Get anchor box with the max iou for every gt bbox
        positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
        # Get anchor boxes with iou above a threshold with any of the gt bboxes
        positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)

        positive_anc_ind_sep = torch.where(positive_anc_mask)[0]  # get separate indices in the batch
        positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
        positive_anc_ind = torch.where(positive_anc_mask)[0]

        max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)  # get anchor box with max iou to GT
        max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

        GT_conf_scores = max_iou_per_anc[positive_anc_ind]  # IoU scores
        gt_classes_expand = labels.view(batch_size, 1, gt_box_num).expand(batch_size, tot_anc_boxes, gt_box_num)
        gt_classes_expand, max_iou_per_anc_ind = gt_classes_expand.to(device), max_iou_per_anc_ind.to(device)
        GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
        GT_class = GT_class.flatten(start_dim=0, end_dim=1)
        GT_class_pos = GT_class[positive_anc_ind]
        gt_bboxes_expand = gt_bboxes_all.view(batch_size, 1, gt_box_num, 4).expand(batch_size, tot_anc_boxes,
                                                                                   gt_box_num, 4)
        gt_bboxes_expand = gt_bboxes_expand.to(device)
        GT_bboxes = torch.gather(gt_bboxes_expand, -2,
                                 max_iou_per_anc_ind.reshape(batch_size, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
        GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
        GT_bboxes_pos = GT_bboxes[positive_anc_ind]
        anc_boxes_flat = all_anc_boxes.flatten(start_dim=0, end_dim=-2)  # flatten all the anchor boxes
        positive_anc_coords = anc_boxes_flat[positive_anc_ind]
        GT_offsets = ModelUtils.calc_gt_offsets(positive_anc_coords, GT_bboxes_pos, device=device)

        return positive_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, positive_anc_ind_sep, \
               max_iou_per_anc, anc_boxes_flat

    @staticmethod
    def _get_neg_anchor_box(max_iou_per_anc, positive_anc_ind, neg_thresh, anc_boxes_flat):
        negative_anc_mask = (max_iou_per_anc < neg_thresh)
        negative_anc_ind = torch.where(negative_anc_mask)[0]
        # Select samples to match the positive samples
        negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
        negative_anc_coords = anc_boxes_flat[negative_anc_ind]
        return negative_anc_ind, negative_anc_coords

    @staticmethod
    def get_req_anchors(all_anc_boxes, gt_bboxes_all, labels, pos_thresh=0.7, neg_thresh=0.2,
                        device=torch.device('cpu')):
        '''
        Prepare necessary data required for training

        Input
        ------
        anc_boxes_all - torch.Tensor of shape (B, w_anc, h_anc, n_anchor_boxes, 4)
            all anchor boxes for a batch of images
        gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
            padded ground truth boxes for a batch of images
        gt_classes_all - torch.Tensor of shape (B, max_objects)
            padded ground truth classes for a batch of images

        Returns
        ---------
        positive_anc_ind -  torch.Tensor of shape (n_pos,)
            flattened positive indices for all the images in the batch
        negative_anc_ind - torch.Tensor of shape (n_pos,)
            flattened positive indices for all the images in the batch
        GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
        GT_offsets -  torch.Tensor of shape (n_pos, 4),
            offsets between +ve anchors and their corresponding ground truth boxes
        GT_class_pos - torch.Tensor of shape (n_pos,)
            mapped classes of +ve anchors
        positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
        negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
        positive_anc_ind_sep - list of indices to keep track of +ve anchors
        '''

        # anchor boxes size
        B, w_anc, h_anc, A, _ = all_anc_boxes.shape
        N = gt_bboxes_all.shape[1]  # max number of GT bboxes in a batch
        tot_anc_boxes = A * w_anc * h_anc  # total anc box in an image

        iou_mat = ModelUtils.get_iou_mat(B, all_anc_boxes, gt_bboxes_all, device=device)
        max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

        # Get positive anchor boxes
        positive_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, positive_anc_ind_sep, \
        max_iou_per_anc, anc_boxes_flat = ModelUtils._get_pos_anchor_box(
            iou_mat=iou_mat, labels=labels, gt_bboxes_all=gt_bboxes_all, all_anc_boxes=all_anc_boxes,
            max_iou_per_gt_box=max_iou_per_gt_box, pos_thresh=pos_thresh, batch_size=B, gt_box_num=N,
            tot_anc_boxes=tot_anc_boxes, device=device)

        # Get negative anchor boxes
        negative_anc_ind, negative_anc_coords = ModelUtils._get_neg_anchor_box(
            max_iou_per_anc=max_iou_per_anc, positive_anc_ind=positive_anc_ind, neg_thresh=neg_thresh,
            anc_boxes_flat=anc_boxes_flat)

        return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, \
               negative_anc_coords, positive_anc_ind_sep

    @staticmethod
    def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
        target_pos = torch.ones_like(conf_scores_pos)
        target_neg = torch.zeros_like(conf_scores_neg)

        target = torch.cat((target_pos, target_neg))
        inputs = torch.cat((conf_scores_pos, conf_scores_neg))

        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size

        return loss

    @staticmethod
    def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
        assert gt_offsets.size() == reg_offsets_pos.size()
        loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size
        return loss


class DataUtils:
    @staticmethod
    def load_annotations_text_file(file_path: str):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    @staticmethod
    def write_annotations_text_file(file_path: str, data):
        with open(file_path, "w") as file:
            for item in data:
                file.write(str(item) + "\n")
        file.close()
        print(f'New data for {file_path}')

    @staticmethod
    def create_custom_dir(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return directory_path
        else:
            count = 1
            while True:
                new_directory_path = f"{directory_path}_{count}"
                if not os.path.exists(new_directory_path):
                    os.makedirs(new_directory_path)
                    return new_directory_path
                count += 1

    @staticmethod
    def get_unique_filename(file_path):
        # Append a number to the base filename if it already exists.
        if not os.path.exists(file_path):
            return file_path

        file_name, file_ext = os.path.splitext(file_path)
        count = 1
        while os.path.exists(f"{file_name}_{count}{file_ext}"):
            count += 1

        return f"{file_name}_{count}{file_ext}"


class Visualizer:
    @staticmethod
    def visualize_data(image, bboxes, is_save_img=False, viz_path=''):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = Visualizer._display_img(image=image, ax=ax)
        bboxes = ops.box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh')
        fig, _ = Visualizer._display_bbox(bboxes, fig, ax)
        if is_save_img:
            fig.savefig(os.path.join(DataUtils.get_unique_filename(os.path.join(viz_path, f'image.png'))))
        else:
            fig.show()

    @staticmethod
    def _display_img(image, ax):
        ax.imshow(image)
        return ax

    @staticmethod
    def _display_bbox(bboxes, fig, ax, classes=None, color='y', line_width=3):
        if classes is not None and bboxes is not None:
            assert len(bboxes) == len(classes)
        for i, box in enumerate(bboxes):
            x, y, w, h = box.numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if classes is not None:
                if classes[i] == 'pad':
                    continue
                ax.text(x + 5, y + 20, classes[i], bbox=dict(facecolor=color, alpha=0.5))
        return fig, ax

    @staticmethod
    def compare_line_plot(data, data_label, compare_data, compare_data_label):
        plt.plot(data, label=data_label)
        plt.plot(compare_data, label=compare_data_label)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
