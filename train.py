from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

from Dataset import BasicDataset
from Metrics import metric
from Metrics import ConfusionMatrix
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from MsanlfNet import MsanlfNet

num_classes = 6
max_epoch = 50
lr = 0.001
batchsize = 16
dataset = 'Vaihingen'

train_dir_img = fr'data/{dataset}/train/'
train_dir_mask = fr'data/{dataset}/train_labels/'
test_dir_img = fr'data/{dataset}/test/'
test_dir_mask = fr'data/{dataset}/test_labels/'

train = BasicDataset(train_dir_img, train_dir_mask, 1)
test = BasicDataset(test_dir_img, test_dir_mask, 1)

train_loader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MsanlfNet(num_classes=num_classes)
net.to(device=device)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

criterion = nn.CrossEntropyLoss()

for epoch in range(max_epoch):
    net.train()
    epoch_loss = 0

    for id1, batch in enumerate(tqdm(train_loader)):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32)
        masks_pred = net(imgs)

        ce_loss = criterion(masks_pred, true_masks.long())
        loss = ce_loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    torch.save(net.state_dict(), 'MsanlfNet.pth')

net.eval()
matrix_ = [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]
for id2, batch in enumerate(tqdm(test_loader)):
    imgs = batch['image'].to(device=device, dtype=torch.float32)
    true_masks = batch['mask'].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_pred = net(imgs)
        predict_all = mask_pred.argmax(dim=1).flatten().cpu()
        label_all = true_masks.flatten().cpu()
        matrix = ConfusionMatrix(numClass=6, imgPredict=predict_all, Label=label_all)
    matrix_ += matrix
precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)

print('precision:{}\nrecall:{}\nOA:{}\n'
      'IoU:{}\nFWIoU:{}\nmIoU:{}\nf1score:{}'.format(precision, recall, OA, IoU, FWIOU, mIOU, f1score))
