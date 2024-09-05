import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import SASV_Dataset
from models import Baseline2
import torch.nn as nn
from aDCF_loss import adcf_loss
from aDCF_metric import calculate_adcf_soft_act
import os


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)

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

trn_set = SASV_Dataset("trn")
trn_loader = DataLoader(trn_set, batch_size=bs, shuffle=True, drop_last=False, pin_memory=True)

dev_set = SASV_Dataset("dev")
dev_loader = DataLoader(dev_set, batch_size=len(dev_set), shuffle=False, drop_last=False, pin_memory=True)

model = Baseline2()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
lr = 0.001
params = list(model.parameters())
optimizer = optim.Adam(params, lr=lr, weight_decay=0.001)
lr_schedular = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: keras_decay(step))
adcf_loss = adcf_loss()
bce_loss = nn.BCELoss()

folder_path = "results/" + str(model.name) + "_" + sys + "_" + str(bs) + "_" + str(epoch_size) + "_" + str(lr)
if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)

if not os.path.exists(folder_path+"/best_model"):
    os.makedirs(folder_path+"/best_model", exist_ok=True)
    
if not os.path.exists(folder_path+"/text_files"):
    os.makedirs(folder_path+"/text_files", exist_ok=True)
    
if not os.path.exists(folder_path+"/figures"):
    os.makedirs(folder_path+"/figures", exist_ok=True)


f1 = open(folder_path+"/text_files/train_loss.txt", "w")

trn_loss = []
adcf_losses = []
bce_losses = []
adcf = 1e4
threshold = 0.5
for epoch in range(epoch_size):
    model.train()
    run_loss = 0.
    adcf_run_loss = 0.
    bce_run_loss = 0.
    preds, keys = [], []
    for num, data_minibatch in enumerate(trn_loader, 0):
        asv1, asv2, cm1, ans, key = data_minibatch
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            if torch.backends.mps.is_available():
                asv1 = asv1.to(device)
                asv2 = asv2.to(device)
                cm1 = cm1.to(device)
                ans = ans.to(device)
                key = key.to(device)

            pred = model(asv1, asv2, cm1)
            loss_adcf = adcf_loss.calculate_a_dcf(pred, key, threshold)
            if sys == 'woBCE':
                loss_adcf = loss_adcf.to(device)
                final_loss = loss_adcf
                final_loss.backward()
                optimizer.step()
                run_loss += final_loss
                adcf_run_loss += loss_adcf
                preds.append(pred)
                keys.extend(list(key))
            elif sys == 'wBCE':
                loss_adcf = loss_adcf.to(device)
                loss_bce = bce_loss(pred, ans.unsqueeze(1).float())
                final_loss = (loss_adcf + loss_bce) / 2
                final_loss.backward()
                optimizer.step()
                run_loss += final_loss
                adcf_run_loss += loss_adcf
                bce_run_loss += loss_bce
                preds.append(pred)
                keys.extend(list(key))
            else:
                print('Loss type error')
    
    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    lr_schedular.step()
    
    trn_loss.append(float((run_loss/len(trn_loader)).data.cpu().numpy()))
    adcf_losses.append(float((adcf_run_loss/len(trn_loader)).data.cpu().numpy()))
    if sys == 'wBCE':
        bce_losses.append((bce_run_loss/len(trn_loader)).data.cpu().numpy())

    if sys == 'woBCE':
        print("\nTrn: Epoch-%d aDCF Loss: %0.5f Total Loss: %0.5f" % (epoch+1, float((adcf_run_loss/len(trn_loader)).data.cpu().numpy()), float((run_loss/len(trn_loader)).data.cpu().numpy())))
        f1.writelines('Epoch: '+str(epoch+1)+ ' aDCF Loss: ' + str(float((adcf_run_loss/len(trn_loader)).detach().cpu().numpy())) + ' Train Loss: ' + str(float((run_loss/len(trn_loader)).detach().cpu().numpy())))
        f1.writelines('\n')
    else:
        print("\nTrn: Epoch-%d aDCF Loss: %0.5f BCE Loss: %0.5f Total Loss: %0.5f" % (epoch+1, float((adcf_run_loss/len(trn_loader)).data.cpu().numpy()), float((bce_run_loss/len(trn_loader)).data.cpu().numpy()), float((run_loss/len(trn_loader)).data.cpu().numpy())))
        f1.writelines('Epoch: '+str(epoch+1)+ ' aDCF Loss: ' + str(float((adcf_run_loss/len(trn_loader)).detach().cpu().numpy())) + ' BCE Loss: ' + str(float((bce_run_loss/len(trn_loader)).detach().cpu().numpy())) + ' Train Loss: ' + str(float((run_loss/len(trn_loader)).detach().cpu().numpy())))
        f1.writelines('\n')
    
    preds_dev, keys_dev = calculation(model, dev_loader)
    
    dev_far_asvs_soft_act, dev_far_cms_soft_act, dev_frrs_soft_act, dev_adcf_soft_act = calculate_adcf_soft_act(preds_dev, keys_dev, threshold)
    
    if dev_adcf_soft_act[0] < adcf:
        torch.save(model.state_dict(), folder_path+"/best_model/soft_aDCF_" + str(bs) + "_" + str(epoch_size) + ".pt")
        adcf = dev_adcf_soft_act[0]
        best_epoch = epoch+1
        print(f'\nEpoch-{epoch+1} min aDCF: {adcf:.6f}')
    
f1.close()

