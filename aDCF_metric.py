import numpy as np
import torch

Pspf: float = 0
Pnontrg: float = 0.5
Ptrg: float = 0.5
Cmiss: float = 1
Cfa_asv: float = 1
Cfa_cm: float = 1
threshold = 0.5


def sigmoid(z):
    return 1/(1+np.exp(-z))

def soft_miss_rate(P, threshold):
    return 1 / len(P) * sum(sigmoid(threshold - x) for x in P)

def soft_false_accept_rate(N, threshold):
    return 1 / len(N) * sum(sigmoid(x - threshold) for x in N)

def calculate_adcf_hard_act(score, label, threshold):
    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = score[label == 0]
    nontrg = score[label == 1]
    spf = score[label == 2]
    
    far_asvs_act = len(np.where(nontrg>threshold)[0]) / len(nontrg)
    far_cms_act = len(np.where(spf>threshold)[0]) / len(spf)
    frrs_act = len(np.where(trg<=threshold)[0]) / len(trg)
    
    a_dcfs = np.array([Cmiss * Ptrg]) * np.array(frrs_act) + \
        np.array([Cfa_asv * Pnontrg]) * np.array(far_asvs_act) + \
        np.array([Cfa_cm * Pspf]) * np.array(far_cms_act)
        
    a_dcfs_normed = normalize(a_dcfs)

    return far_asvs_act, far_cms_act, frrs_act, a_dcfs_normed

def calculate_adcf_soft_act(score, label, threshold):
    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = score[label == 0]
    nontrg = score[label == 1]
    spf = score[label == 2]
    
    far_asvs = soft_false_accept_rate(nontrg, threshold)
    far_cms = soft_false_accept_rate(spf, threshold)
    frrs = soft_miss_rate(trg, threshold)
    
    a_dcfs = np.array([Cmiss * Ptrg]) * np.array(frrs) + \
        np.array([Cfa_asv * Pnontrg]) * np.array(far_asvs) + \
        np.array([Cfa_cm * Pspf]) * np.array(far_cms)
        
    a_dcfs_normed = normalize(a_dcfs)

    return far_asvs, far_cms, frrs, a_dcfs_normed

# min values from hard and soft case

def calculate_adcf_hard_min(score, label):
    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = score[label == 0]
    nontrg = score[label == 1]
    spf = score[label == 2]
    
    far_asvs, far_cms, frrs, fars, a_dcf_thresh = compute_det_curve_hard(trg, nontrg, spf)
    
    a_dcfs = np.array([Cmiss * Ptrg]) * np.array(frrs) + \
        np.array([Cfa_asv * Pnontrg]) * np.array(far_asvs) + \
        np.array([Cfa_cm * Pspf]) * np.array(far_cms)
        
    a_dcfs_normed = normalize(a_dcfs)
    min_a_dcf_idx = np.argmin(a_dcfs_normed)
    min_a_dcf = a_dcfs_normed[min_a_dcf_idx]
    min_a_dcf_thresh = a_dcf_thresh[min_a_dcf_idx]
    
    return far_asvs, far_cms, frrs, fars, a_dcfs_normed, min_a_dcf, a_dcf_thresh, min_a_dcf_thresh, \
    far_asvs[min_a_dcf_idx], far_cms[min_a_dcf_idx], frrs[min_a_dcf_idx]

def compute_det_curve_hard(trg_scores, nontrg_scores, spf_scores):
    
    all_scores = np.concatenate((trg_scores, nontrg_scores, spf_scores))
    labels = np.concatenate(
        (np.zeros_like(trg_scores), np.ones_like(nontrg_scores), np.ones_like(spf_scores) + 1))
    # Sort labels based on scores
    indices = np.argsort(all_scores.squeeze(), kind='mergesort')
    labels = labels[indices]
    scores_sorted = all_scores[indices]
    
    fp_nontrg, fp_spf, fp, fn = len(nontrg_scores), len(spf_scores), len(nontrg_scores)+len(spf_scores), 0
    far_asvs, far_cms, fars, frrs, a_dcf_thresh = [], [], [], [], []
    for sco, lab in zip(scores_sorted, labels):
        if lab == 1: # non-target
            fp_nontrg -= 1 # false alarm for accepting nontarget trial
            fp -= 1
        elif lab == 0: # target
            fn += 1 # miss for rejecting target trial
        elif lab == 2: # spoof
            fp_spf -= 1 # false alarm for accepting spof trial
            fp -= 1
        else:
            raise ValueError ("Label should be one of (0, 1, 2).")
        far_asvs.append(fp_nontrg / len(nontrg_scores))
        far_cms.append(fp_spf / len(spf_scores))
        frrs.append(fn / len(trg_scores))
        fars.append(fp / (len(spf_scores) + len(nontrg_scores)))
        a_dcf_thresh.append(sco[0])
        
    return far_asvs, far_cms, frrs, fars, a_dcf_thresh

def calculate_adcf_soft_min(score, label):
    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = score[label == 0]
    nontrg = score[label == 1]
    spf = score[label == 2]
    
    far_asvs, far_cms, frrs, fars, a_dcf_thresh = compute_det_curve_soft_min(trg, nontrg, spf)
    
    a_dcfs = np.array([Cmiss * Ptrg]) * np.array(frrs) + \
        np.array([Cfa_asv * Pnontrg]) * np.array(far_asvs) + \
        np.array([Cfa_cm * Pspf]) * np.array(far_cms)
        
    a_dcfs_normed = normalize(a_dcfs)
    min_a_dcf_idx = np.argmin(a_dcfs_normed)
    min_a_dcf = a_dcfs_normed[min_a_dcf_idx]
    min_a_dcf_thresh = a_dcf_thresh[min_a_dcf_idx]

    return far_asvs, far_cms, frrs, fars, a_dcfs_normed, min_a_dcf, a_dcf_thresh, min_a_dcf_thresh, \
    far_asvs[min_a_dcf_idx], far_cms[min_a_dcf_idx], frrs[min_a_dcf_idx]

def compute_det_curve_soft_min(trg_scores, nontrg_scores, spf_scores):
    
    all_scores = np.concatenate((trg_scores, nontrg_scores, spf_scores))
    labels = np.concatenate(
        (np.zeros_like(trg_scores), np.ones_like(nontrg_scores), np.ones_like(spf_scores) + 1))
    # Sort labels based on scores
    indices = np.argsort(all_scores.squeeze(), kind='mergesort')
    labels = labels[indices]
    scores_sorted = all_scores[indices]
    
    far_asvs, far_cms, frrs, fars, a_dcf_thresh = [], [], [], [], []
    for sco, lab in zip(scores_sorted, labels):
        far_asvs.append((sum(sigmoid(nontrg_scores - sco[0]))[0]) / len(nontrg_scores)) # false alarm for accepting nontarget trial
        frrs.append((sum(sigmoid(sco[0] - trg_scores))[0]) / len(trg_scores)) # miss for rejecting target trial
        far_cms.append((sum(sigmoid(spf_scores - sco[0]))[0]) / len(spf_scores)) # false alarm for accepting spoof trial
        fars.append(((sum(sigmoid(spf_scores - sco[0]))[0]) + (sum(sigmoid(nontrg_scores - sco[0]))[0])) / (len(spf_scores) + len(nontrg_scores)))
        a_dcf_thresh.append(sco[0]) 
        
    return far_asvs, far_cms, frrs, fars, a_dcf_thresh


def normalize(a_dcfs: np.ndarray) -> np.ndarray:
    
    a_dcf_all_accept = np.array([Cfa_asv * Pnontrg + \
        Cfa_cm * Pspf])                             
    a_dcf_all_reject = np.array([Cmiss * Ptrg])   
    
    a_dcfs_normed = a_dcfs / min(a_dcf_all_accept, a_dcf_all_reject)
    
    return a_dcfs_normed

