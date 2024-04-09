import torch
import torch.nn as nn
import torch.nn.functional as F


class RandLANetContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.2, num_points=4096*11, num_classes=20):
        super(RandLANetContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.num_points = num_points
        self.num_classes = num_classes
        #self.randlanet = RandLANet(num_classes=self.num_classes, num_points=self.num_points)

    def forward(self, x, label, class_weights):
        # Compute the embeddings and the logits
        batch_size, num_points, feature_size = x.shape
        feature = x.double()
        positive = F.one_hot(label, num_classes=self.num_classes).double()
        negative = class_weights.double()

        # compute the similarity mask for positive and negative samples
        pos_mask = positive.unsqueeze(0) == positive.unsqueeze(1)
        neg_mask = ~pos_mask

        pos_loss = torch.matmul(feature, positive.transpose(1,2))
        neg_loss = torch.matmul(feature, negative.transpose(1,2))

        exp_loss = torch.sum(torch.exp((pos_loss * pos_mask - neg_loss * neg_mask) / self.temperature), axis=2)
        log_loss = torch.sum(torch.log(1 + exp_loss), axis=2)

        # compute the total loss
        loss = pos_loss + neg_loss

        """
        for b in range(batch_size):
            for n in range(self.num_points):
                f1 = x[b,n].float()
                f2 = (class_weights[b,n]).float()

                # compute the pairwise similarity between all pairs of feature vectors
                similarity = torch.dist(f1.T,f2, p=2)

                # compute the similarity mask for positive and negative samples
                #pos_mask = (F.one_hot(label[n], num_classes=self.num_classes)>0)
                pos_mask = (F.one_hot(label[b,n], num_classes=self.num_classes)>0).unsqueeze(0)
                pos_mask = pos_mask & (F.one_hot(label[b,n], num_classes=self.num_classes)>0).unsqueeze(1)
                #pos_mask = pos_mask.T
                neg_mask = ~pos_mask

                # compute the loss for positive and negative samples
                #pos_loss = 0.5 * torch.sum(pos_mask.float() * torch.square(torch.max(torch.tensor(0.0), self.margin - similarity)))
                #neg_loss = 0.5 * torch.sum(neg_mask.float() * torch.square(similarity))

                distances = similarity
                pos_loss = torch.exp(distances / self.temperature) * pos_mask
                pos_loss = torch.sum(pos_loss, dim=1) / torch.sum(pos_mask, dim=1)
                pos_loss = -torch.log(pos_loss)

                neg_loss = torch.exp(self.margin - distances / self.temperature) * neg_mask
                neg_loss = torch.sum(neg_loss, dim=1) / torch.sum(neg_mask, dim=1)
                neg_loss = -torch.log(neg_loss)

                # compute the total loss
                loss = pos_loss + neg_loss
                loss_tot += loss
        """
        loss_tot = loss_tot / self.num_points
        return loss_tot


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5, num_points=4096*11, num_classes_coarse=7, num_classes_fine=20):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.num_points = num_points
        self.num_classes_coarse = num_classes_coarse
        self.num_classes_fine = num_classes_fine
        #self.randlanet = RandLANet(num_classes=self.num_classes_fine, num_points=self.num_points)

    def forward(self, x, y_coarse, y_fine):
        # Compute the embeddings and the predictions
        output = x #self.randlanet(x)
        global_feature = output[:, :, 0]
        local_feature = output[:, :, 1:]
        logits_fine = torch.mean(local_feature, dim=1)
        logits_coarse = torch.zeros((logits_fine.shape[0], self.num_classes_coarse)).to(x.device)
        for i in range(self.num_classes_coarse):
            indices = torch.where(y_coarse == i)[0]
            if len(indices) > 0:
                logits_coarse[:, i] = torch.mean(logits_fine[indices, :], dim=0)
        probs_fine = nn.functional.softmax(logits_fine / self.temperature, dim=1)
        probs_coarse = nn.functional.softmax(logits_coarse / self.temperature, dim=1)

        # Compute the knowledge distillation loss
        loss_fine = nn.functional.kl_div(probs_fine.log(), nn.functional.softmax(y_fine / self.temperature, dim=1),
                                         reduction='batchmean')
        loss_coarse = nn.functional.kl_div(probs_coarse.log(),
                                           nn.functional.softmax(y_coarse / self.temperature, dim=1),
                                           reduction='batchmean')
        loss = self.alpha * loss_fine + (1 - self.alpha) * loss_coarse

        return global_feature, loss


class c2f_kdc(nn.Module):
    def __init__(self, ids_mapping=None, same_kd_lambda=True):
        super(c2f_kdc, self).__init__()
        self.ids_mapping = ids_mapping
        self.same_kd_lambda = same_kd_lambda

    def forward(self, nw_out, nw_out_old):
        # compute the target distribution
        labels = torch.softmax(nw_out_old, dim=1)
        # log(softmax) -> denominator = difference
        den = torch.logsumexp(nw_out, dim=1, keepdim=True)
        # align distributions plane-wise using the mappings
        planes = []
        if self.same_kd_lambda == 1:
            for ids in self.ids_mapping:
                if len(ids) == 1:
                    planes.append(nw_out[:, ids, ...] - den)
                else:
                    planes.append(torch.logsumexp(nw_out[:, ids, ...], dim=1, keepdim=True) - den)
            # concatenate the logs
            out = torch.cat(planes, dim=1)
            # compute point-wise cross-entropy
            loss = (labels * out).sum(dim=1)
            # compute the loss
            return -loss.mean()
        else:
            id_macro_ff = []
            id_macro_fc = []
            planes_ff = []
            planes_fc = []
            for id_m, ids in enumerate(self.ids_mapping):
                if len(ids) == 1:
                    id_macro_ff.append(id_m)
                    planes_ff.append(nw_out[:, ids, ...] - den)
                else:
                    id_macro_fc.append(id_m)
                    planes_fc.append(torch.logsumexp(nw_out[:, ids, ...], dim=1, keepdim=True) - den)
            # concatenate the logs
            if planes_ff:
                out_f = torch.cat(planes_ff, dim=1)
                loss_f = (labels[:, id_macro_ff, ...] * out_f).sum(dim=1)

            if planes_fc:
                out_c = torch.cat(planes_fc, dim=1)
                loss_c = (labels[:, id_macro_fc, ...] * out_c).sum(dim=1)

            if planes_ff and planes_fc:
                return -loss_f.mean(), -loss_c.mean()
            elif planes_ff:
                return -loss_f.mean(), torch.tensor([0.], requires_grad=True, device=nw_out.device)
            elif planes_fc:
                return torch.tensor([0.], requires_grad=True, device=nw_out.device), -loss_c.mean()


class c2f_distance_kd(nn.Module):
    def __init__(self, ids_mapping=None, use_logits=False, metric='L2'):
        super(c2f_distance_kd, self).__init__()
        self.ids_mapping = ids_mapping
        self.use_logits = use_logits

        if metric == 'L2':
            self.metric = nn.MSELoss()
        elif metric == 'L1':
            self.metric = nn.L1Loss()
        else:
            raise ValueError("Unrecognized alignment loss, must be ['L1', 'L2']")

    def forward(self, nw_out, nw_out_old):
        if self.use_logits:
            nw_out = torch.softmax(nw_out, dim=1)
            nw_out_old = torch.softmax(nw_out_old, dim=1)

        # extract the appropriate planes
        planes = []
        for ids in self.ids_mapping:
            if len(ids) == 1:
                planes.append(nw_out[:, ids, ...])
            else:
                planes.append(torch.sum(nw_out[:, ids, ...], dim=1, keepdim=True))
        # concatenate the planes
        out = torch.cat(planes, dim=1)

        return self.metric(out, nw_out_old)