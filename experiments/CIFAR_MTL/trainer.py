import logging
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from experiments.CIFAR_MTL.data import CIFAR100, one_hot
from tqdm import trange

from BayesAgg_MTL import BayesAggMTL
from experiments.utils import (
    set_seed,
    set_logger,
    common_parser,
    get_device,
    parse_eval_summary,
)
from experiments.CIFAR_MTL.models import CNN

set_logger()


@torch.no_grad()
def evaluate(net, n_tasks, loader):

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    total = 0.0
    correct = 0.0
    correct_per_task = torch.zeros(n_tasks, dtype=torch.float32)
    losses_per_task = torch.zeros(n_tasks, dtype=torch.float32)

    for i, batch in enumerate(loader):
        net.eval()
        batch = (t.to(device) for t in batch)
        img, _, label_coarse = batch
        ys = one_hot(label_coarse, n_tasks)

        # prediction per task
        logits = net(img)
        losses = criterion(logits, ys).mean(dim=0).cpu()

        # looks only at the task and according to that decides
        correct_per_task += (logits > 0).eq(ys).sum(dim=0).cpu()
        # looks at all tasks simultaneously. The true class should be the best one
        correct += logits.argmax(1).eq(label_coarse).sum(dim=0).cpu()
        total += len(ys)

        losses *= len(ys)

        losses_per_task += losses

    tasks_accs = (correct_per_task / total).numpy()
    mean_tasks_accs = correct / total
    tasks_losses = (losses_per_task / total).numpy()

    eval_summary = {"losses": tasks_losses, "mean_accs": mean_tasks_accs, "accs": tasks_accs}
    return eval_summary


def main(args, n_tasks: int, device: torch.device):
    # ----
    # Nets
    # ---

    # Get only feature ex layers since we have class heads per task
    net: nn.Module = CNN(
        n_tasks=n_tasks, activation=args.activation
    )
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # dataset and dataloaders
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = CIFAR100(args.data_path, train=True, download=True, transform=transform)
    test_set = CIFAR100(args.data_path, train=False, download=True, transform=transform)

    assert 0 < args.val_size < 1, f"val_pct should be in (0, 1), got {args.val_pct}."
    indices = list(range(len(train_set)))
    train_indices, val_indices = train_test_split(indices, test_size=args.val_size)

    val_set = torch.utils.data.Subset(train_set, val_indices)
    train_set = torch.utils.data.Subset(train_set, train_indices)

    logging.info(
        f"train size: {len(train_set)}, val size: {len(val_set)}, test size: {len(test_set)}"
    )

    # TODO: benchmark num_workers
    num_workers = 4

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # weight method
    method = BayesAggMTL(num_tasks=args.n_tasks, n_outputs_per_task_group=[args.n_tasks],
                         task_types=['binary_tasks'],
                         agg_scheme_hps={},
                         cls_hps={'gamma': args.gamma, 'sqrt_power': args.sqrt_power_cls,
                                  'num_mc_samples': args.num_mc_samples})

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=net.shared_parameters(), lr=args.lr),
            dict(params=net.task_specific_parameters(), lr=args.lr),
        ],
        lr=args.lr,
        weight_decay=args.wd,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.n_epochs * 0.6), int(args.n_epochs * 0.8)], gamma=0.1
    )
    epochs = args.n_epochs
    epoch_iter = trange(epochs)

    step = -1
    best_val_acc = 0.0
    best_val_task_losses = best_test_task_losses = best_val_task_accs = best_test_task_accs = np.inf, np.inf, 0.0, 0.0
    best_mean_val_loss, best_mean_test_loss, best_mean_val_acc, best_mean_test_acc = np.inf, np.inf, 0.0, 0.0
    mean_val_loss, mean_test_loss = np.inf, np.inf
    mean_val_acc, mean_test_acc = 0.0, 0.0
    train_features = train_labels = None

    for epoch in epoch_iter:
        lr = scheduler.optimizer.param_groups[0]["lr"]
        for j, batch in enumerate(train_loader):
            step += 1

            # init properties
            net.train()
            optimizer.zero_grad()
            batch = (t.to(device) for t in batch)
            img, _, label_coarse = batch

            ys = one_hot(label_coarse, n_tasks)

            # prediction per task
            logits, features = net(img, return_representation=True)
            losses = criterion(logits, ys).mean(dim=0)

            # calculate weighted loss + update
            if epoch < args.ls_epochs:
                loss = torch.sum(losses)
                loss.backward()
            else:
                method.backward(
                    losses=losses,
                    last_layer_params=list(net.task_specific_parameters()),
                    representation=features,
                    labels=[ys],
                    full_train_features=train_features if j == 0 else None,
                    full_train_labels=[train_labels] if j == 0 else None
                )

            torch.nn.utils.clip_grad_norm_(net.task_specific_parameters(), 50)
            # if epoch >= args.ls_epochs:
            #     torch.nn.utils.clip_grad_norm_(net.shared_parameters(), 0.2)

            optimizer.step()

            # logging
            description = f"Mean train loss: {losses.mean().item():.3f}, lr: {lr:.5f}"
            epoch_iter.set_description(description)

            online_train_features = features.detach().clone() if j == 0 else \
                torch.cat((online_train_features, features.detach().clone()), dim=0)
            online_train_labels = ys if j == 0 else torch.cat((online_train_labels, ys), dim=0)

        scheduler.step()

        train_features = online_train_features.clone()
        train_labels = online_train_labels.clone()

        if (epoch + 1) % args.eval_every == 0:
            val_summary = evaluate(net, n_tasks, val_loader)
            val_task_losses = val_summary["losses"]
            mean_val_loss = val_task_losses.mean().item()
            val_task_accs = val_summary["accs"]
            mean_val_acc = val_summary["mean_accs"]

            test_summary = evaluate(net, n_tasks, test_loader)
            test_task_losses = test_summary["losses"]
            mean_test_loss = test_task_losses.mean().item()
            test_task_accs = test_summary["accs"]
            mean_test_acc = test_summary["mean_accs"]

            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                best_val_task_losses = val_task_losses
                best_test_task_losses = test_task_losses
                best_val_task_accs = val_task_accs
                best_test_task_accs = test_task_accs
                best_mean_val_loss = mean_val_loss
                best_mean_test_loss = mean_test_loss
                best_mean_val_acc = mean_val_acc
                best_mean_test_acc = mean_test_acc

            # metric, split, values
        print_dict = {}
        print_dict.update(parse_eval_summary("loss", "test", test_task_losses))
        print_dict.update(parse_eval_summary("acc", "test", test_task_accs))
        print_dict.update(parse_eval_summary("best_loss", "test", best_test_task_losses))
        print_dict.update(parse_eval_summary("best_acc", "test", best_test_task_accs))
        print_dict.update(
            {
                "val_acc": mean_val_acc.item(),
                "best_val_acc": best_mean_val_acc.item(),
                "test_acc": mean_test_acc.item(),
                "best_test_acc": best_mean_test_acc.item(),
            }
        )

        logging.info(print_dict)


if __name__ == "__main__":
    parser = ArgumentParser("CIFAR-MTL", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset/CIFAR_MTL",
        lr=1e-3,
        wd=0.0001,
        n_epochs=50,
        batch_size=128,
        ls_epochs=3,
        gamma=0.0005,
        sqrt_power_cls=0.0005,
    )
    parser.add_argument(
        "--n-tasks", type=int, default=20, choices=[20], help="num. tasks"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="pct of training examples for val set.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=["relu", "leaky_relu", "elu"],
        help="activation function in cnn",
    )

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)
    main(
        args=args,
        n_tasks=args.n_tasks,
        device=device,
    )
