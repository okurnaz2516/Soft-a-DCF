import torch
from torch.utils.data import DataLoader
from dataset import SASV_Dataset
from models import Baseline2
from metrics import get_all_EERs_my
from aDCF_metric import calculate_adcf_hard_act, calculate_adcf_hard_min, calculate_adcf_soft_act, calculate_adcf_soft_min
import matplotlib.pyplot as plt

def calculation(model, loader):
    model.eval()
    with torch.no_grad():
        preds, keys = [], []
        for num, data_minibatch in enumerate(loader, 0):
            asv1, asv2, cm1, ans, key = data_minibatch
            if torch.backends.mps.is_available():
                asv1 = asv1.to(device)
                asv2 = asv2.to(device)
                cm1 = cm1.to(device)
                ans = ans.to(device)
    
            pred = model(asv1, asv2, cm1)
            preds.append(pred)
            keys.append(key)
    
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        keys = torch.cat(keys, dim=0).detach().cpu().numpy()
    
    return preds, keys


# sys can be either 'wBCE' (aDCF loss with BCE) or 'woBCE' (aDCF loss without BCE)  

# sys = 'woBCE' # S2 system in paper (aDCF loss without BCE)
sys = 'wBCE'  # S3 system in paper (aDCF loss with BCE)
epoch_size = 250
bs = 1024
threshold = 0.5
lr = 0.001
model = Baseline2()
folder_path = "results/" + str(model.name) + "_" + sys + "_" + str(bs) + "_" + str(epoch_size) + "_" + str(lr)
model.load_state_dict(torch.load(folder_path+"/best_model/soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".pt"))

#%%  development set results

dev_set = SASV_Dataset("dev")
dev_loader = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False, drop_last=False, pin_memory=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

preds_dev, keys_dev = calculation(model, dev_loader)

dev_far_asvs_hard_act, dev_far_cms_hard_act, dev_frrs_hard_act, dev_a_dcfs_hard_act = calculate_adcf_hard_act(preds_dev, keys_dev, threshold)

dev_far_asvs_soft_act, dev_far_cms_soft_act, dev_frrs_soft_act, dev_a_dcfs_soft_act = calculate_adcf_soft_act(preds_dev, keys_dev, threshold)

dev_far_asvs_hard, dev_far_cms_hard, dev_frrs_hard, dev_fars_hard, dev_a_dcfs_hard, dev_a_dcfs_hard_min, dev_a_dcf_thresh_hard, \
dev_a_dcf_thresh_hard_min, dev_far_asvs_hard_min, dev_far_cms_hard_min, dev_frrs_hard_min = calculate_adcf_hard_min(preds_dev, keys_dev)

dev_far_asvs_soft, dev_far_cms_soft, dev_frrs_soft, dev_fars_soft, dev_a_dcfs_soft, dev_a_dcfs_soft_min, dev_a_dcf_thresh_soft, \
dev_a_dcf_thresh_soft_min, dev_far_asvs_soft_min, dev_far_cms_soft_min, dev_frrs_soft_min = calculate_adcf_soft_min(preds_dev, keys_dev)

sasv_eer_dev, sv_eer_dev, spf_eer_dev = get_all_EERs_my(preds_dev, keys_dev)
print("dev: sasv_eer_dev: %0.3f, sv_eer_dev: %0.3f, spf_eer_dev: %0.3f" % (100 * sasv_eer_dev, 100 * sv_eer_dev, 100 * spf_eer_dev))

print("%0.4f" % (dev_far_asvs_hard_act))
print("%0.4f" % (dev_far_cms_hard_act))
print("%0.4f" % (dev_frrs_hard_act))
print("%0.4f" % (dev_a_dcfs_hard_act[0]))
print(0.5)
print("************************")
print("%0.4f" % (dev_far_asvs_soft_act[0]))
print("%0.4f" % (dev_far_cms_soft_act[0]))
print("%0.4f" % (dev_frrs_soft_act[0]))
print("%0.4f" % (dev_a_dcfs_soft_act[0]))
print(0.5)
print("************************")
print("%0.4f" % (dev_far_asvs_hard_min))
print("%0.4f" % (dev_far_cms_hard_min))
print("%0.4f" % (dev_frrs_hard_min))
print("%0.4f" % (dev_a_dcfs_hard_min))
print("%0.4f" % (dev_a_dcf_thresh_hard_min))
print("************************")
print("%0.4f" % (dev_far_asvs_soft_min))
print("%0.4f" % (dev_far_cms_soft_min))
print("%0.4f" % (dev_frrs_soft_min))
print("%0.4f" % (dev_a_dcfs_soft_min))
print("%0.4f" % (dev_a_dcf_thresh_soft_min))

with open(folder_path+"/text_files/dev_results_soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".txt", "w") as file:
    file.write("dev_far_asvs_hard_act: %0.4f\n" % dev_far_asvs_hard_act)
    file.write("dev_far_cms_hard_act: %0.4f\n" % dev_far_cms_hard_act)
    file.write("dev_frrs_hard_act: %0.4f\n" % dev_frrs_hard_act)
    file.write("dev_a_dcfs_hard_act: %0.4f\n" % dev_a_dcfs_hard_act[0])
    file.write("dev_far_asvs_soft_act: %0.4f\n" % dev_far_asvs_soft_act[0])
    file.write("dev_far_cms_soft_act: %0.4f\n" % dev_far_cms_soft_act[0])
    file.write("dev_frrs_soft_act: %0.4f\n" % dev_frrs_soft_act[0])
    file.write("dev_a_dcfs_soft_act: %0.4f\n" % dev_a_dcfs_soft_act[0])
    file.write("dev_far_asvs_hard_min: %0.4f\n" % dev_far_asvs_hard_min)
    file.write("dev_far_cms_hard_min: %0.4f\n" % dev_far_cms_hard_min)
    file.write("dev_frrs_hard_min: %0.4f\n" % dev_frrs_hard_min)
    file.write("dev_a_dcfs_hard_min: %0.4f\n" % dev_a_dcfs_hard_min)
    file.write("dev_a_dcf_thresh_hard_min: %0.4f\n" % dev_a_dcf_thresh_hard_min)
    file.write("dev_far_asvs_soft_min: %0.4f\n" % dev_far_asvs_soft_min)
    file.write("dev_far_cms_soft_min: %0.4f\n" % dev_far_cms_soft_min)
    file.write("dev_frrs_soft_min: %0.4f\n" % dev_frrs_soft_min)
    file.write("dev_a_dcfs_soft_min: %0.4f\n" % dev_a_dcfs_soft_min)
    file.write("dev_a_dcf_thresh_soft_min: %0.4f\n" % dev_a_dcf_thresh_soft_min)
    file.write("sasv_eer_dev: %0.4f\n" % sasv_eer_dev)
    file.write("sv_eer_dev: %0.4f\n" % sv_eer_dev)
    file.write("spf_eer_dev: %0.4f\n" % spf_eer_dev)

file.close()

plt.plot(dev_a_dcf_thresh_hard, dev_a_dcfs_hard, color='r', label='Hard case')
plt.plot(dev_a_dcf_thresh_soft, dev_a_dcfs_soft, color='b', label='Soft case')
plt.xlabel('Threshold')
plt.ylabel('a-DCF')
plt.title('a-DCF values concerning fixed threshold (0.5) in development')
plt.legend()

plt.scatter([threshold], [dev_a_dcfs_hard_act], color='red', marker='x', s=100, label='Hard Act a-DCF with thres. 0.5')
plt.scatter([threshold], [dev_a_dcfs_soft_act], color='blue', marker='.', s=100, label='Soft Act a-DCF with thres. 0.5')

plt.scatter([dev_a_dcf_thresh_hard_min], [dev_a_dcfs_hard_min], color='red', marker='*', s=100, label='Hard min a-DCF')
plt.scatter([dev_a_dcf_thresh_soft_min], [dev_a_dcfs_soft_min], color='blue', marker='1', s=100, label='Soft min a-DCF')

plt.legend()

plt.savefig(folder_path+"/figures/dev_soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".jpg")
plt.show()


#%% eval set results

eval_set = SASV_Dataset("eval")
eval_loader = DataLoader(eval_set, batch_size=len(eval_set), shuffle=False, drop_last=False, pin_memory=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

preds_eval, keys_eval = calculation(model, eval_loader)

eval_far_asvs_hard_act, eval_far_cms_hard_act, eval_frrs_hard_act, eval_a_dcfs_hard_act = calculate_adcf_hard_act(preds_eval, keys_eval, threshold)

eval_far_asvs_soft_act, eval_far_cms_soft_act, eval_frrs_soft_act, eval_a_dcfs_soft_act = calculate_adcf_soft_act(preds_eval, keys_eval, threshold)

eval_far_asvs_hard, eval_far_cms_hard, eval_frrs_hard, eval_fars_hard, eval_a_dcfs_hard, eval_a_dcfs_hard_min, eval_a_dcf_thresh_hard, \
eval_a_dcf_thresh_hard_min, eval_far_asvs_hard_min, eval_far_cms_hard_min, eval_frrs_hard_min = calculate_adcf_hard_min(preds_eval, keys_eval)

eval_far_asvs_soft, eval_far_cms_soft, eval_frrs_soft, eval_fars_soft, eval_a_dcfs_soft, eval_a_dcfs_soft_min, eval_a_dcf_thresh_soft, \
eval_a_dcf_thresh_soft_min, eval_far_asvs_soft_min, eval_far_cms_soft_min, eval_frrs_soft_min = calculate_adcf_soft_min(preds_eval, keys_eval)

sasv_eer_eval, sv_eer_eval, spf_eer_eval = get_all_EERs_my(preds_eval, keys_eval)
print("eval: sasv_eer_eval: %0.3f, sv_eer_eval: %0.3f, spf_eer_eval: %0.3f" % (100 * sasv_eer_eval, 100 * sv_eer_eval, 100 * spf_eer_eval))

print("%0.4f" % (eval_far_asvs_hard_act))
print("%0.4f" % (eval_far_cms_hard_act))
print("%0.4f" % (eval_frrs_hard_act))
print("%0.4f" % (eval_a_dcfs_hard_act[0]))
print(0.5)
print("************************")
print("%0.4f" % (eval_far_asvs_soft_act[0]))
print("%0.4f" % (eval_far_cms_soft_act[0]))
print("%0.4f" % (eval_frrs_soft_act[0]))
print("%0.4f" % (eval_a_dcfs_soft_act[0]))
print(0.5)
print("************************")
print("%0.4f" % (eval_far_asvs_hard_min))
print("%0.4f" % (eval_far_cms_hard_min))
print("%0.4f" % (eval_frrs_hard_min))
print("%0.4f" % (eval_a_dcfs_hard_min))
print("%0.4f" % (eval_a_dcf_thresh_hard_min))
print("************************")
print("%0.4f" % (eval_far_asvs_soft_min))
print("%0.4f" % (eval_far_cms_soft_min))
print("%0.4f" % (eval_frrs_soft_min))
print("%0.4f" % (eval_a_dcfs_soft_min))
print("%0.4f" % (eval_a_dcf_thresh_soft_min))

with open(folder_path+"/text_files/eval_results_soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".txt", "w") as file:
    file.write("eval_far_asvs_hard_act: %0.4f\n" % eval_far_asvs_hard_act)
    file.write("eval_far_cms_hard_act: %0.4f\n" % eval_far_cms_hard_act)
    file.write("eval_frrs_hard_act: %0.4f\n" % eval_frrs_hard_act)
    file.write("eval_a_dcfs_hard_act: %0.4f\n" % eval_a_dcfs_hard_act[0])
    file.write("eval_far_asvs_soft_act: %0.4f\n" % eval_far_asvs_soft_act[0])
    file.write("eval_far_cms_soft_act: %0.4f\n" % eval_far_cms_soft_act[0])
    file.write("eval_frrs_soft_act: %0.4f\n" % eval_frrs_soft_act[0])
    file.write("eval_a_dcfs_soft_act: %0.4f\n" % eval_a_dcfs_soft_act[0])
    file.write("eval_far_asvs_hard_min: %0.4f\n" % eval_far_asvs_hard_min)
    file.write("eval_far_cms_hard_min: %0.4f\n" % eval_far_cms_hard_min)
    file.write("eval_frrs_hard_min: %0.4f\n" % eval_frrs_hard_min)
    file.write("eval_a_dcfs_hard_min: %0.4f\n" % eval_a_dcfs_hard_min)
    file.write("eval_a_dcf_thresh_hard_min: %0.4f\n" % eval_a_dcf_thresh_hard_min)
    file.write("eval_far_asvs_soft_min: %0.4f\n" % eval_far_asvs_soft_min)
    file.write("eval_far_cms_soft_min: %0.4f\n" % eval_far_cms_soft_min)
    file.write("eval_frrs_soft_min: %0.4f\n" % eval_frrs_soft_min)
    file.write("eval_a_dcfs_soft_min: %0.4f\n" % eval_a_dcfs_soft_min)
    file.write("eval_a_dcf_thresh_soft_min: %0.4f\n" % eval_a_dcf_thresh_soft_min)
    file.write("sasv_eer_eval: %0.4f\n" % sasv_eer_eval)
    file.write("sv_eer_eval: %0.4f\n" % sv_eer_eval)
    file.write("spf_eer_eval: %0.4f\n" % spf_eer_eval)

file.close()

plt.plot(eval_a_dcf_thresh_hard, eval_a_dcfs_hard, color='r', label='Hard case')
plt.plot(eval_a_dcf_thresh_soft, eval_a_dcfs_soft, color='b', label='Soft case')
plt.xlabel('Threshold')
plt.ylabel('a-DCF')
plt.title('a-DCF values concerning fixed threshold (0.5) in evaluation')
plt.legend()

plt.scatter([threshold], [eval_a_dcfs_hard_act], color='red', marker='x', s=100, label='Hard Act a-DCF with thres. 0.5')
plt.scatter([threshold], [eval_a_dcfs_soft_act], color='blue', marker='.', s=100, label='Soft Act a-DCF with thres. 0.5')

plt.scatter([eval_a_dcf_thresh_hard_min], [eval_a_dcfs_hard_min], color='red', marker='*', s=100, label='Hard min a-DCF')
plt.scatter([eval_a_dcf_thresh_soft_min], [eval_a_dcfs_soft_min], color='blue', marker='1', s=100, label='Soft min a-DCF')

plt.legend()

plt.savefig(folder_path+"/figures/eval_soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".jpg")
plt.show()

