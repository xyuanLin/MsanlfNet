from torch.utils.data import DataLoader
import torch

from Dataset import BasicDataset
from Metrics import metric
from Metrics import ConfusionMatrix
from tqdm import tqdm
from models.MsanlfNet import MsanlfNet

num_classes = 6
max_epoch = 50
lr = 0.001
batchsize = 16
dataset = 'Vaihingen'

test_dir_img = fr'data/{dataset}/test/'
test_dir_mask = fr'data/{dataset}/test_labels/'

test = BasicDataset(test_dir_img, test_dir_mask)

test_loader = DataLoader(test, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MsanlfNet(num_classes=num_classes)
net.to(device=device)
pretrained_dict = torch.load('MsanlfNet.pth')
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

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
