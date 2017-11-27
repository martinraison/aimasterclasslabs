from __future__ import division
from __future__ import print_function
import argparse
import torch
from torchvision import transforms
from torch.autograd import Variable
import datasets
from predictor import Predictor

# Training settings
parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument("--split", default='val')
parser.add_argument("--root", default='agirecole')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.AgirEcole(args.root, args.split, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1722,), (0.3309,))
                    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


predictor = Predictor()
test_loss = 0
correct = 0
for data, target in test_loader:
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = predictor(data)
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)

print("%.1f" % (100 * correct / len(test_loader.dataset)))
