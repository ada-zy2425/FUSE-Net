import copy
import numpy as np
import torch
import torch.nn as nn
from utils import to_gpu, time_desc_decorator
import models 
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

class Solver:
    def __init__(self, train_cfg, dev_cfg, test_cfg, train_loader, dev_loader, test_loader, is_train=True):
        self.trc, self.dvc, self.tsc = train_cfg, dev_cfg, test_cfg
        self.trl, self.dvl, self.tsl = train_loader, dev_loader, test_loader
        self.is_train = is_train
        self.model = None

    @time_desc_decorator('Build')
    def build(self):
        use_pp = getattr(self.trc, 'use_fusenetpp', False)
        if use_pp:
            from fusenetpp import FUSENetPP
            self.model = FUSENetPP(self.trc)
        else:
            import models  # 指 FUSENet/models.py
            self.model = models.FUSENet(self.trc)

        if torch.cuda.is_available(): self.model.cuda()

        if self.is_train:
            bert_params = [p for n, p in self.model.named_parameters() if 'bert' in n and p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() if 'bert' not in n and p.requires_grad]
            
            optimizer_grouped_parameters = [
                {'params': bert_params, 'lr': self.trc.bert_learning_rate, 'weight_decay': self.trc.weight_decay},
                {'params': other_params, 'lr': self.trc.learning_rate, 'weight_decay': self.trc.weight_decay}
            ]
            
            self.opt = self.trc.optimizer(optimizer_grouped_parameters)
            
            if self.trc.scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=self.trc.T_0, T_mult=self.trc.T_mult, eta_min=1e-7)
            elif self.trc.scheduler_type == 'plateau':
               
                self.scheduler = ReduceLROnPlateau(self.opt, mode='min', patience=self.trc.patience, factor=0.5)
            else:
                self.scheduler = None

    @time_desc_decorator('Train')
    def train(self):
        is_reg = self.trc.data.lower() in ['mosi', 'mosei', 'ch-sims']
        task_criterion = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()

        best_dev_loss = float('inf')
        best_weights = None
        patience_counter = self.trc.patience

        for epoch in range(1, self.trc.n_epoch + 1):
            self.model.train()
            for s, v, a, y, l, bs, bt, bm in self.trl:
                s, v, a, y, l = map(to_gpu, (s, v, a, y, l))
                if bs is not None: bs, bm = map(to_gpu, (bs, bm))
                if bt is not None: bt = to_gpu(bt)

                self.opt.zero_grad()
                out, representations = self.model(s, v, a, l, bs, bt, bm)
                
                L_task = task_criterion(out.squeeze(), y.float().squeeze())
                L_info = self.compute_infoNCE(representations['shared'], representations['private'])
                L_siam = self.compute_siamese(representations['noise'])
                L_info_gain = self.compute_info_gain(representations, y, task_criterion, is_reg)
                L_cycle = self.compute_cycle(representations['shared'], representations['private'])
                L_mrc = self.compute_mrc_loss(representations['mrc'])
                
                loss = (
                    self.trc.task_weight * L_task
                    + self.trc.info_weight * L_info
                    + self.trc.siamese_weight * L_siam
                    + self.trc.info_gain_weight * L_info_gain
                    + self.trc.cycle_weight * L_cycle
                    + self.trc.recon_weight * L_mrc
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trc.clip)
                self.opt.step()
                if self.scheduler and self.trc.scheduler_type == 'cosine': self.scheduler.step()

            dev_loss, _ = self.eval('dev')
            print(f"Epoch {epoch}/{self.trc.n_epoch} | Dev Loss: {dev_loss:.4f}")
            
            if self.scheduler and self.trc.scheduler_type == 'plateau':
               
                old_lr = self.opt.param_groups[0]['lr']
                self.scheduler.step(dev_loss)
                new_lr = self.opt.param_groups[0]['lr']
                if new_lr < old_lr:
                    print(f"Epoch {epoch}: ReduceLROnPlateau reducing learning rate to {new_lr:.6f}.")

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = self.trc.patience
                print("Found new best model on dev set!")
            else:
                patience_counter -= 1
                if patience_counter <= 0:
                    print("Patience exhausted, early stopping.")
                    break
        
        if best_weights:
            self.model.load_state_dict(best_weights)
            print("\nTesting with best model...")
            _, test_metrics = self.eval('test', to_print=True)
            return test_metrics
        return None

    def eval(self, mode, to_print=False):
        self.model.eval()
        loader = self.dvl if mode == 'dev' else self.tsl
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for s, v, a, y, l, bs, bt, bm in loader:
                s, v, a, y, l = map(to_gpu, (s, v, a, y, l))
                if bs is not None: bs, bm = map(to_gpu, (bs, bm))
                if bt is not None: bt = to_gpu(bt)
                out, _ = self.model(s, v, a, l, bs, bt, bm)
                all_preds.append(out.cpu())
                all_trues.append(y.cpu())
        
        preds = torch.cat(all_preds).numpy()
        trues = torch.cat(all_trues).numpy()
        
        metrics = self.calc_metrics(trues, preds)
        loss = metrics['mae']
        
        if to_print:
            print(f"--- Results on {mode} set ---")
            for k, v in metrics.items(): print(f"{k}: {v:.4f}")
            
        return loss, metrics

    def compute_mrc_loss(self, mrc_outputs):
        recon_criterion = nn.MSELoss()
        total_recon_loss, total_kl_loss = 0.0, 0.0
        for m in ['t', 'v', 'a']:
            o = mrc_outputs[m]
            total_recon_loss += recon_criterion(o['recon'], o['target'])
            kl_div = -0.5 * torch.sum(1 + o['log_var'] - o['mu'].pow(2) - o['log_var'].exp(), dim=1).mean()
            total_kl_loss += kl_div
        return (total_recon_loss / 3.0) + self.trc.vib_beta * (total_kl_loss / 3.0)

    def compute_cycle(self, repr_shared, repr_private):
        total = 0.0
        for m in ['t', 'v', 'a']:
            Hs, Hp = repr_shared[m], repr_private[m]
            Gm, Gm_inv = self.model.G[m], self.model.G_inv[m]
            total += (Hs - Gm(Hp)).pow(2).mean() + (Hp - Gm_inv(Hs)).pow(2).mean()
        return total / 3.0

    def compute_info_gain(self, reps, y, criterion, is_reg):
        total = 0.0
        rep_s, rep_p, rep_n = reps['shared'], reps['private'], reps['noise']
        for m in ['t','v','a']:
            rs_m, rp_m, rn_m = rep_s[m], rep_p[m], rep_n[m]
            yt = y.float().squeeze() if is_reg else y.long()
            pred_s = self.model.info_gain_head_s(rs_m).squeeze()
            pred_p = self.model.info_gain_head_h(rp_m).squeeze()
            pred_n = self.model.info_gain_head_n(rn_m).squeeze()
            ce_s, ce_p, ce_n = criterion(pred_s, yt), criterion(pred_p, yt), criterion(pred_n, yt)
            total += (-ce_s - ce_p + ce_n)
        return total / 3.0
        
    def compute_infoNCE(self, h_s, h_p):
        total, count = 0.0, 0
        for m1 in ['t', 'v', 'a']:
            for m2 in ['t', 'v', 'a']:
                if m1 == m2: continue
                sim_shared = torch.exp(torch.cosine_similarity(h_s[m1], h_s[m2], dim=1) / self.trc.temperature)
                sim_private = torch.exp(torch.cosine_similarity(h_s[m1], h_p[m1], dim=1) / self.trc.temperature)
                total += (-torch.log(sim_shared / (sim_shared + sim_private))).mean()
                count += 1
        return total / count if count > 0 else 0.0

    def compute_siamese(self, Rn):
        if self.trc.siamese_weight == 0: return 0.0
        B = next(iter(Rn.values())).size(0)
        if B <= 1: return 0.0
        sum1 = sum(torch.norm(Rn[m][i] - Rn[m][j], p=2) for i in range(B) for j in range(B) if i != j for m in ['t', 'v', 'a'])
        term1 = sum1 / (3 * B * (B - 1))
        sum2 = sum(torch.relu(self.trc.epsilon - torch.norm(Rn[a][i] - Rn[b][i], p=2)) for i in range(B) for a in ['t', 'v', 'a'] for b in ['t', 'v', 'a'] if a != b)
        term2 = sum2 / (6 * B)
        return term1 + term2

    def calc_metrics(self, y_true, y_pred):
        preds, trues = y_pred.squeeze(), y_true.squeeze()
        mae = np.mean(np.abs(preds - trues))
        corr = np.corrcoef(preds, trues)[0, 1] if len(preds) > 1 else 0.0
        
        acc7 = np.mean(np.round(np.clip(preds, -3, 3)) == np.round(np.clip(trues, -3, 3)))
        acc5 = np.mean(np.round(np.clip(preds, -2, 2)) == np.round(np.clip(trues, -2, 2)))
        
        non_zeros = trues != 0
        f1_pn = f1_score(trues[non_zeros] > 0, preds[non_zeros] > 0, average='weighted') if np.any(non_zeros) else 0.0
        acc2_pn = accuracy_score(trues[non_zeros] > 0, preds[non_zeros] > 0) if np.any(non_zeros) else 0.0
        
        acc2_nn = accuracy_score(trues >= 0, preds >= 0)
        
        return {"mae": mae, "pearson": corr, "acc7": acc7, "acc5": acc5, 
                "f1_pos_neg": f1_pn, "acc2_pos_neg": acc2_pn, "acc2_nonneg_neg": acc2_nn}