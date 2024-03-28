import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import ops
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
import time
import os

from utils import DataUtils, Visualizer, ModelUtils


class CustomObjectDetector:
    def __init__(self, train_data=None, test_data=None, val_data=None, roi_size=(2, 2), anc_scales=(2, 4, 6),
                 anc_ratios=(0.5, 1, 1.5), pos_thresh=0.9, neg_thresh=0.1, early_stopping_patience=25):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

        self.epochs, self.batch_size, self.learning_rate = 0, 0, 0

        self.image_size = train_data.image_size
        self.n_class = train_data.n_class
        self.roi_size = roi_size
        self.anc_scales = anc_scales
        self.anc_ratios = anc_ratios
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        self.model = TwoStageDetector(n_classes=self.n_class,
                                      roi_size=self.roi_size,
                                      anc_scales=self.anc_scales,
                                      anc_ratios=self.anc_ratios,
                                      pos_thresh=self.pos_thresh,
                                      neg_thresh=self.neg_thresh,
                                      image_size=self.image_size,
                                      transform=self.train_data.transform,
                                      device=self.device)
        self.model = self.model.to(self.device)

        self.model_save_path = 'models'
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.date_now_srt = datetime.now().strftime("%m%d%Y")

    def __training_loop(self, batch, optimizer, running_loss):
        image, labels, box_data = batch
        image, labels, box_data = image.to(self.device), labels.to(self.device), box_data.to(self.device)
        loss = self.model(image, box_data, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        return running_loss, optimizer

    def __validation_loop(self, batch, val_loss):
        image, labels, box_data = batch
        image, labels, box_data = image.to(self.device), labels.to(self.device), box_data.to(self.device)

        loss = self.model(image, box_data, labels)
        val_loss += loss.item()
        return val_loss

    def _validate(self, average_loss, losses, epoch):
        self.model.eval()
        data_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=True)
        val_loss = 0
        with torch.no_grad():
            with tqdm(data_loader, desc=f"Validating:", unit="batch") as tbar:
                for batch in tbar:
                    val_loss = self.__validation_loop(batch, val_loss)
            avg_val_loss = val_loss / len(data_loader)
        self.model.train()
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss}, Val Loss: {avg_val_loss}")
        losses['validation_losses'].append(avg_val_loss)
        return avg_val_loss, losses

    def _save_best_model(self, val_loss, epoch, full_model_save_path, best_model_stats):
        if val_loss < self.best_val_loss and epoch > 1:
            self.best_val_loss = val_loss
            best_model = self.model.state_dict()
            best_model_stats['best_epoch'] = epoch
            best_model_stats['no_improvement_count'] = 0
            best_model_full_save_path = os.path.join(full_model_save_path, 'best_model.pt')
            print(f'Best weights: Validation loss: {self.best_val_loss}')
            print(f'Best model will be saved to {best_model_full_save_path}.')
            torch.save(obj=best_model, f=best_model_full_save_path)
        else:
            best_model_stats['no_improvement_count'] += 1
        return best_model_stats

    def _get_training_results(self, training_loss, val_loss, log_data_path, epoch):
        log_data = []
        if os.path.exists(log_data_path):
            with open(log_data_path, 'r') as file:
                log_data = [line.strip() for line in file.readlines()]
        else:
            log_data.append(f'Image Size: {self.image_size}')
            log_data.append(f'Batch Size: {self.batch_size}')
            log_data.append(f'Epochs: {self.epochs}')
            log_data.append(f'Learning rate: {self.learning_rate}')
            log_data.append(f'Anchor Scales: {self.anc_scales}')
            log_data.append(f'Anchor Ratios: {self.anc_ratios}')
            log_data.append('Training Logs: ------------------------------')
        log_data.append(f'e={epoch} ;tl={training_loss} ;vl={val_loss}')
        return log_data

    def _save_training_data(self, full_model_save_path, epoch, training_loss, val_loss, start_time, model_name='model'):
        if not os.path.exists(full_model_save_path):
            os.makedirs(full_model_save_path)
        torch.save(obj=self.model.state_dict(), f=os.path.join(full_model_save_path, f'{model_name}.pt'))

        log_data_path = os.path.join(full_model_save_path, 'train_logs.txt')
        log_data = self._get_training_results(training_loss=training_loss, val_loss=val_loss,
                                              log_data_path=log_data_path, epoch=epoch)

        if epoch + 1 >= self.epochs:
            end_time = time.time()
            total_time = end_time - start_time
            log_data.append(f'Total train Time: {total_time}')
            print(f'Total Train time: {total_time}')
        with open(log_data_path, 'w') as file:
            for item in log_data:
                file.write(str(item) + '\n')

    def _is_early_stop(self, no_improvement_count, epoch):
        if 0 < self.early_stopping_patience <= no_improvement_count:
            print(f'No improvements after {self.early_stopping_patience} epochs. '
                  f'Training will stop at epoch {epoch}/{self.epochs}.')
            return True
        return False

    def _train_epoch(self, data_loader, optimizer, epoch, losses, full_model_save_path, best_model_stats, start_time):
        running_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as tbar:
            for batch in tbar:
                running_loss, optimizer = self.__training_loop(batch=batch, optimizer=optimizer,
                                                               running_loss=running_loss)
            average_loss = running_loss / len(data_loader)
            losses['training_losses'].append(average_loss)

        # Validation
        val_loss, losses = self._validate(average_loss=average_loss, losses=losses, epoch=epoch)

        # Save the model if it has the best validation loss so far
        best_model_stats = self._save_best_model(val_loss=val_loss, epoch=epoch,
                                                 full_model_save_path=full_model_save_path,
                                                 best_model_stats=best_model_stats)

        self._save_training_data(epoch=epoch, training_loss=average_loss, val_loss=val_loss, start_time=start_time,
                                 full_model_save_path=full_model_save_path)

        if self._is_early_stop(no_improvement_count=best_model_stats['no_improvement_count'], epoch=epoch):
            return None

        return optimizer, losses, best_model_stats

    def train(self, epochs, batch_size, learning_rate=1e-3):
        losses = {'training_losses': [], 'validation_losses': []}
        best_model_stats = {'no_improvement_count': 0, 'best_epoch': 0}
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        full_model_save_path = DataUtils.get_unique_filename(
            os.path.join(
                self.model_save_path,
                f'model_{self.date_now_srt}_{epochs}e_{len(self.train_data)}d'
            )
        )
        print(f'Model will be saved to {full_model_save_path}.')

        data_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=False)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            train_results = self._train_epoch(
                data_loader=data_loader, optimizer=optimizer, epoch=epoch, losses=losses, start_time=start_time,
                full_model_save_path=full_model_save_path, best_model_stats=best_model_stats)
            if train_results:
                optimizer, losses, best_model_stats = train_results
            else:
                # Early Stop
                break

        Visualizer.compare_line_plot(data=losses['training_losses'],
                                     data_label='Training loss',
                                     compare_data=losses['validation_losses'],
                                     compare_data_label='Validation loss')
        print(f'Training Done. '
              f'Best weights: Validation loss: {self.best_val_loss} on epoch: {best_model_stats["best_epoch"]}.')
        return losses['training_losses'], losses['validation_losses']

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(f=model_path))
        print(f'Model loaded from {model_path}')

    def test_images(self, num_images=1, is_save_img=True):
        self.model.eval()
        data_loader = DataLoader(dataset=self.test_data, batch_size=num_images, shuffle=True)
        viz_path = DataUtils.create_custom_dir(directory_path='test_results\\test_viz')
        for images, labels, box_data in data_loader:
            images = images.to(self.device)
            img_h, img_w = images.size()[2:]
            proposals_final, conf_scores_final, classes_final, confs, out_size = self.model.inference(images,
                                                                                                      conf_thresh=0.95,
                                                                                                      nms_thresh=0.05)
            if not any(var is None for var in [proposals_final, conf_scores_final, classes_final, confs, out_size]):
                scale_factor_h, scale_factor_w = img_h // out_size[0], img_w // out_size[1]
                images = images * 255
                for i, image in enumerate(images):
                    proposed_bboxes = [ModelUtils.project_bboxes(proposals_final[i], scale_factor_w,
                                                                 scale_factor_h, mode='a2p')]
                    if proposed_bboxes != [None]:
                        proposed_bboxes = pad_sequence(proposed_bboxes, batch_first=True, padding_value=-1).cpu()
                        image = image.cpu().permute(1, 2, 0)
                        #proposed_bboxes = ops.box_convert(proposed_bboxes, in_fmt='xyxy', out_fmt='xywh')
                        Visualizer.visualize_data(image=image, bboxes=proposed_bboxes[0], is_save_img=is_save_img,
                                                  viz_path=viz_path)
                    else:
                        print('No Pred')
                break


class TwoStageDetector(nn.Module):
    def __init__(self, n_classes, roi_size, anc_scales, anc_ratios, pos_thresh,
                 neg_thresh, image_size, transform, device):
        super().__init__()
        self.image_size = (3,) + image_size
        self.transform = transform
        self.device = device

        # Stage 1 model
        self.rpn = RegionProposalNetwork(anc_scales=anc_scales,
                                         anc_ratios=anc_ratios,
                                         pos_thresh=pos_thresh,
                                         neg_thresh=neg_thresh,
                                         device=self.device)
        self.rpn = self.rpn.to(self.device)

        # Stage 2 model
        self.classifier = ClassificationModule(out_channels=self.rpn.out_channels,
                                               n_classes=n_classes,
                                               roi_size=roi_size)
        self.classifier = self.classifier.to(self.device)

    def forward(self, images, gt_bboxes, gt_classes):
        images, gt_bboxes, gt_classes = images.to(self.device), gt_bboxes.to(self.device), gt_classes.to(self.device)
        total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos = \
            self.rpn(images, gt_bboxes, gt_classes)

        # Get separate proposals for each sample
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)
        feature_map, GT_class_pos = feature_map.to(self.device), GT_class_pos.to(self.device)
        cls_loss = self.classifier(feature_map, pos_proposals_list, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss

        return total_loss

    def inference(self, images, conf_thresh=0.9, nms_thresh=0.7):
        images = self.__validate_images(images=images)
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map, confs, out_size = self.rpn.inference(
            images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map, proposals_final)

        cls_probs = F.softmax(cls_scores, dim=-1)
        classes_all = torch.argmax(cls_probs, dim=-1)

        classes_final = []
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i])  # Get the number of proposals for each image
            classes_final.append(classes_all[c: c + n_proposals])
            c += n_proposals

        return proposals_final, conf_scores_final, classes_final, confs, out_size

    def __validate_images(self, images):
        new_images = []
        for image in images:
            if type(image) != Image.Image:
                if image.shape != self.image_size:
                    new_images.append(self.transform(image))
                else:
                    new_images.append(image)
            else:
                new_images.append(self.transform(image))
        new_images = torch.stack(new_images)
        new_images = new_images.to(self.device)
        return new_images


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        self.num_features = list(req_layers[-1][-1].children())[-2].num_features

    def forward(self, img_data):
        return self.backbone(img_data)


class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3, device=torch.device('cpu')):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)
        self.device = device

    def _verify_mode(self, pos_anc_ind, neg_anc_ind, pos_anc_coords):
        return 'eval' if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None else 'train'

    def _generate_train_proposals(self, conf_scores_pred, reg_offsets_pred, pos_anc_coords, pos_anc_ind, neg_anc_ind):
        conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
        conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
        offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
        proposals = ModelUtils.generate_proposals(anchors=pos_anc_coords, offsets=offsets_pos, device=self.device)
        return conf_scores_pos, conf_scores_neg, offsets_pos, proposals

    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        mode = self._verify_mode(pos_anc_ind=pos_anc_ind, neg_anc_ind=neg_anc_ind, pos_anc_coords=pos_anc_coords)

        out = self.dropout(self.conv1(feature_map))
        out = F.relu(out)

        reg_offsets_pred = self.reg_head(out)  # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out)  # (B, A, hmap, wmap)

        if mode == 'train':
            return self._generate_train_proposals(conf_scores_pred=conf_scores_pred, reg_offsets_pred=reg_offsets_pred,
                                                  pos_anc_coords=pos_anc_coords, pos_anc_ind=pos_anc_ind,
                                                  neg_anc_ind=neg_anc_ind)
        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred


class RegionProposalNetwork(nn.Module):
    def __init__(self, anc_scales, anc_ratios, pos_thresh, neg_thresh, device=torch.device('cpu')):
        super().__init__()
        self.device = device

        # Anchor boxes parameters
        self.anc_scales = anc_scales
        self.anc_ratios = anc_ratios
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)

        # IoU thresholds
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        # weights for loss
        self.w_conf = 1
        self.w_reg = 5

        self.feature_extractor = FeatureExtractor()
        self.out_channels = self.feature_extractor.num_features
        self.proposal_module = ProposalModule(in_features=self.out_channels,
                                              n_anchors=self.n_anc_boxes,
                                              device=self.device)

    def _generate_anchors(self, out_size, batch_size):
        anc_pts_x, anc_pts_y = ModelUtils.gen_anc_centers(out_size=out_size)
        anc_base = ModelUtils.gen_anc_base(anc_pts_x=anc_pts_x,
                                           anc_pts_y=anc_pts_y,
                                           anc_scales=self.anc_scales,
                                           anc_ratios=self.anc_ratios,
                                           out_size=out_size,
                                           device=self.device)
        return anc_base.repeat(batch_size, 1, 1, 1, 1)

    def _get_anchors(self, gt_bboxes, gt_classes, anc_boxes_all, scale_factor_w, scale_factor_h):
        gt_bboxes_proj = ModelUtils.project_bboxes(gt_bboxes, scale_factor_w, scale_factor_h, mode='p2a')
        return ModelUtils.get_req_anchors(all_anc_boxes=anc_boxes_all, gt_bboxes_all=gt_bboxes_proj, labels=gt_classes,
                                          device=self.device)

    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        img_h, img_w = images.size()[2:]
        feature_map = self.feature_extractor(images)
        out_h, out_w = feature_map.size()[2:]
        scale_factor_h, scale_factor_w = img_h // out_h, img_w // out_w

        # Generate Anchors
        anc_boxes_all = self._generate_anchors(out_size=(out_h, out_w), batch_size=batch_size)

        # Get Positive and Negative Anchors
        positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = self._get_anchors(
            gt_bboxes=gt_bboxes, gt_classes=gt_classes, anc_boxes_all=anc_boxes_all, scale_factor_w=scale_factor_w,
            scale_factor_h=scale_factor_h)

        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(
            feature_map=feature_map, pos_anc_ind=positive_anc_ind, neg_anc_ind=negative_anc_ind,
            pos_anc_coords=positive_anc_coords)

        cls_loss = ModelUtils.calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = ModelUtils.calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss

        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos

    def _filter_confidence(self, proposals, conf_scores, conf_thresh):
        conf_idx = torch.where(conf_scores >= conf_thresh)[0]
        conf_scores_pos = conf_scores[conf_idx]
        proposals_pos = proposals[conf_idx]
        confs = conf_scores[conf_idx]
        return conf_scores_pos, proposals_pos, confs

    def _filter_nms(self, proposals_pos, conf_scores_pos, nms_thresh):
        nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
        conf_scores_pos = conf_scores_pos[nms_idx]
        proposals_pos = proposals_pos[nms_idx]
        return conf_scores_pos, proposals_pos

    def _filter_proposals(self, conf_thresh, nms_thresh, batch_size, conf_scores_pred, offsets_pred, anc_boxes_flat):
        proposals_final = []
        conf_scores_final = []
        for i in range(batch_size):
            conf_scores = torch.sigmoid(conf_scores_pred[i])
            offsets = offsets_pred[i]
            anc_boxes = anc_boxes_flat[i]
            proposals = ModelUtils.generate_proposals(anchors=anc_boxes, offsets=offsets, device=self.device)

            # filter based on confidence threshold
            conf_scores_pos, proposals_pos, confs = self._filter_confidence(proposals=proposals,
                                                                            conf_scores=conf_scores,
                                                                            conf_thresh=conf_thresh)
            conf_scores_pos, proposals_pos = self._filter_nms(proposals_pos=proposals_pos,
                                                              conf_scores_pos=conf_scores_pos,
                                                              nms_thresh=nms_thresh)
            proposals_final.append(proposals_pos)
            conf_scores_final.append(conf_scores_pos)
        return proposals_final, conf_scores_final, confs

    def inference(self, images, conf_thresh=0.9, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)
            out_h, out_w = feature_map.size()[2:]
            out_size = (out_h, out_w)

            anc_boxes_all = self._generate_anchors(out_size=(out_h, out_w), batch_size=batch_size)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            # Get confidence scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # Filter out low confidence proposals
            proposals_final, conf_scores_final, confs = self._filter_proposals(conf_thresh=conf_thresh,
                                                                               nms_thresh=nms_thresh,
                                                                               batch_size=batch_size,
                                                                               conf_scores_pred=conf_scores_pred,
                                                                               offsets_pred=offsets_pred,
                                                                               anc_boxes_flat=anc_boxes_flat)
        return proposals_final, conf_scores_final, feature_map, confs, out_size


class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()
        self.roi_size = roi_size

        # Hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

        # Define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, feature_map, proposals_list, gt_classes=None):
        mode = 'eval' if gt_classes is None else 'train'
        # Apply roi pooling on proposals followed by avg pooling
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))

        # Get the classification scores
        cls_scores = self.cls_head(out)

        if mode == 'eval':
            return cls_scores
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        return cls_loss
