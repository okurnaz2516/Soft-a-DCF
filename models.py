import torch
from torch import nn

class Baseline2(nn.Module):
    def __init__(self, num_nodes=[256, 128, 64]):
        super().__init__()
        self.name = "Baseline2"
        self.enh_DNN = self._make_layers(544, num_nodes)
        self.fc_out = torch.nn.Linear(num_nodes[-1], 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        asv_enr = torch.squeeze(embd_asv_enr, 1) # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1) # shape: (bs, 192)
        cm_tst = torch.squeeze(embd_cm_tst, 1) # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim = 1)) # shape: (bs, 544)
        x = self.sigmoid(self.fc_out(x))

        return x

    def _make_layers(self, in_dim, l_nodes):
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(torch.nn.Linear(in_features = in_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(torch.nn.Linear(in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(torch.nn.LeakyReLU(negative_slope = 0.3))
        return torch.nn.Sequential(*l_fc)
