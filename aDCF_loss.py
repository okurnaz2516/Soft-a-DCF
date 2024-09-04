import torch

class adcf_loss:
    def __init__(self, Pspf=0.05, Pnontrg=0.05, Ptrg=0.9, Cmiss=1, Cfa_asv=10, Cfa_cm=20):
        self.Pspf = Pspf
        self.Pnontrg = Pnontrg
        self.Ptrg = Ptrg
        self.Cmiss = Cmiss
        self.Cfa_asv = Cfa_asv
        self.Cfa_cm = Cfa_cm
        self.sigmoid = torch.nn.Sigmoid()

    def calculate_a_dcf(self, score, label, threshold):
        
        trg = score[label == 0]
        nontrg = score[label == 1]
        spf = score[label == 2]
        
        far_asvs, far_cms, frrs = self.compute_a_det_curve(trg, nontrg, spf, threshold)
        
        far_asvs = far_asvs.to('cpu')
        far_cms = far_cms.to('cpu')
        frrs = frrs.to('cpu')
        
        a_dcfs = torch.tensor([self.Cmiss * self.Ptrg]) * frrs + \
                 torch.tensor([self.Cfa_asv * self.Pnontrg]) * far_asvs + \
                 torch.tensor([self.Cfa_cm * self.Pspf]) * far_cms

        a_dcfs_normed = self.normalize(a_dcfs)

        min_a_dcf_idx = torch.argmin(a_dcfs_normed)
        min_a_dcf = a_dcfs_normed[min_a_dcf_idx]
        # min_a_dcf_thresh = a_dcf_thresh[min_a_dcf_idx]

        return min_a_dcf

    def normalize(self, a_dcfs):
        a_dcf_all_accept = torch.tensor([self.Cfa_asv * self.Pnontrg + \
                                     self.Cfa_cm * self.Pspf])
        a_dcf_all_reject = torch.tensor([self.Cmiss * self.Ptrg])

        a_dcfs_normed = a_dcfs / torch.min(a_dcf_all_accept, a_dcf_all_reject)

        return a_dcfs_normed
    
    def soft_miss_rate(self, P, threshold):
        return 1 / len(P) * sum(self.sigmoid(threshold - x) for x in P)

    def soft_false_accept_rate(self, N, threshold):
        return 1 / len(N) * sum(self.sigmoid(x - threshold) for x in N)

    def compute_a_det_curve(self, trg_scores, nontrg_scores, spf_scores, threshold):
        
        far_asvs = self.soft_false_accept_rate(nontrg_scores, threshold)
        far_cms = self.soft_false_accept_rate(spf_scores, threshold)
        frrs = self.soft_miss_rate(trg_scores, threshold)
            
    
        return far_asvs, far_cms, frrs
