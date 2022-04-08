import argparse
import math
from test import test_cifar10, test_cifar100

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# set random seeds
seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_accuracy_fn(test_logits, test_loader):
    labels_arr = torch.tensor([])
    labels_arr = labels_arr.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        labels_arr = torch.cat((labels_arr, labels), dim=0)
    print("test accuracy: %.3f" % accuracy(test_logits, labels_arr)[0])


def get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
    num_warmup_steps=0,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(
            args, args.datapath
        )
        PATH = args.bestmodel
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(
            args, args.datapath
        )
        PATH = args.bestmodel

    if not(args.test):

        args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
        best_acc = 0
        unlabeled_dataset_train, unlabeled_dataset_validate = train_test_split(
            unlabeled_dataset, test_size=0.02, random_state=84
        )
        labeled_loader = iter(
            DataLoader(
                labeled_dataset,
                batch_size=args.train_batch,
                shuffle=True,
                num_workers=args.num_workers,
            )
        )
        unlabeled_loader = iter(
            DataLoader(
                unlabeled_dataset_train,
                batch_size=args.train_batch,
                shuffle=True,
                num_workers=args.num_workers,
            )
        )

        validation_loader = DataLoader(
            unlabeled_dataset_validate,
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.num_workers,
        )
        del unlabeled_dataset_train
        del unlabeled_dataset_validate
        args.start_epoch = 0
        model = WideResNet(
            args.model_depth, args.num_classes, widen_factor=args.model_width
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
        )
        softmax_var = torch.nn.Softmax(dim=-1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.total_iter)
        model = model.to(device)

        ############################################################################
        # TODO: SUPPLY your code
        ############################################################################

        for epoch in range(args.start_epoch, args.epoch):
            pseudo_data = torch.tensor([], device=device)
            pseudo_label = torch.tensor([], device=device)
            model.train()
            for i in range(args.iter_per_epoch):
                try:
                    x_l, y_l = next(labeled_loader)

                except StopIteration:
                    labeled_loader = iter(
                        DataLoader(
                            labeled_dataset,
                            batch_size=args.train_batch,
                            shuffle=True,
                            num_workers=args.num_workers,
                        )
                    )
                    x_l, y_l = next(labeled_loader)
                x_l, y_l = x_l.to(device), y_l.to(device)
                y_l = y_l.long()
                optimizer.zero_grad()
                outputs = model(x_l)
                loss = criterion(outputs, y_l)
                loss.backward()
                optimizer.step()
                scheduler.step()

                try:
                    x_ul, _ = next(unlabeled_loader)
                except StopIteration:
                    unlabeled_loader = iter(
                        DataLoader(
                            unlabeled_dataset,
                            batch_size=args.train_batch,
                            shuffle=True,
                            num_workers=args.num_workers,
                        )
                    )
                    x_ul, _ = next(unlabeled_loader)
                x_ul = x_ul.to(device)
                with torch.no_grad():
                    pred = model(x_ul)
                    pred = softmax_var(pred)
                    correct_idx_mask = torch.max(pred, axis=-1).values >= args.threshold
                    x_ul = x_ul[correct_idx_mask]
                    pred_confirm = torch.masked_select(
                        torch.argmax(pred, axis=-1), correct_idx_mask
                    )
                    pseudo_data = torch.cat((pseudo_data, x_l), dim=0)
                    pseudo_label = torch.cat((pseudo_label, y_l), dim=0)
                    pseudo_data = torch.cat((pseudo_data, x_ul), dim=0)
                    pseudo_label = torch.cat((pseudo_label, pred_confirm), dim=0)

            print(
                "[%d/%d] loss: %.3f, accuracy: %.3f"
                % (i, epoch, loss.item(), accuracy(outputs, y_l)[0])
            )
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Accuracy/train", accuracy(outputs, y_l)[0], epoch)
            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            model.eval()
            predicted_arr = torch.tensor([], device=device)
            labels_arr = torch.tensor([], device=device)
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    predicted_arr = torch.cat((predicted_arr, outputs), dim=0)
                    labels_arr = torch.cat((labels_arr, labels), dim=0)
            #############
            test_acc = accuracy(predicted_arr, labels_arr)[0]
            print("Accuracy of the network on the validation images: %.3f" % (test_acc))
            writer.add_scalar("Accuracy/validation", test_acc, epoch)
            model.train()
            pseudo_dataset = TensorDataset(
                pseudo_data.to("cpu"), pseudo_label.to("cpu")
            )
            labeled_loader = iter(
                DataLoader(
                    pseudo_dataset,
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.num_workers,
                )
            )
            torch.cuda.empty_cache()
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            if is_best:
                torch.save(model, PATH)
        writer.close()
    else:
        # test
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.dataset == "cifar10":
            test_logits = test_cifar10(test_dataset, filepath=PATH)
            test_accuracy_fn(test_logits, test_loader)
        elif args.dataset == "cifar100":
            test_logits = test_cifar100(test_dataset, filepath=PATH)
            test_accuracy_fn(test_logits, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch"
    )
    parser.add_argument(
        "--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    parser.add_argument(
        "--datapath",
        default="./data/",
        type=str,
        help="Path to the CIFAR-10/100 dataset",
    )
    parser.add_argument(
        "--num-labeled", type=int, default=4000, help="Total number of labeled samples"
    )
    parser.add_argument(
        "--lr", default=1.5, type=float, help="The initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="Optimizer momentum"
    )
    parser.add_argument("--wd", default=0.00005, type=float, help="Weight decay")
    parser.add_argument(
        "--expand-labels", action="store_true", help="expand labels to fit eval steps"
    )
    parser.add_argument("--train-batch", default=64, type=int, help="train batchsize")
    parser.add_argument("--test-batch", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--total-iter",
        default=2 * 2,
        type=int,
        help="total number of iterations to run",
    )
    parser.add_argument(
        "--iter-per-epoch",
        default=2,
        type=int,
        help="Number of iterations to run per epoch",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers to launch during training",
    )
    parser.add_argument(
        "--test",
        default=0,
        type=int,
        help="flag, If only test needs to be performed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence Threshold for pseudo labeling",
    )
    parser.add_argument(
        "--bestmodel",
        type=str,
        default="./model/model_best.pth.tar",
        help="Path to save log files",
    )
    parser.add_argument(
        "--model-depth", type=int, default=28, help="model depth for wide resnet"
    )
    parser.add_argument(
        "--model-width", type=int, default=2, help="model width for wide resnet"
    )

    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    args = parser.parse_args()

    main(args)  # for testing only send second argument as 0
