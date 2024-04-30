from argparse import ArgumentParser
import logging
import numpy as np
import torch
import torch.nn as nn
from experiments.ChestX_ray14.dataset.build import build_loader
from experiments.ChestX_ray14.models import ResNet34
from BayesAgg_MTL import BayesAggMTL
from experiments.ChestX_ray14.calc_delta import delta_fn
from sklearn.metrics import roc_auc_score

from tqdm import trange

from experiments.utils import (
    set_seed,
    set_logger,
    common_parser,
    get_device,
    parse_eval_summary,
)

set_logger()


@torch.no_grad()
def evaluate(model, loader, n_tasks, dataset="val"):

    model.eval()
    data_size = 0
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    losses_per_task = torch.zeros(n_tasks, dtype=torch.float32)
    acc_per_task = torch.zeros(n_tasks, dtype=torch.float32)

    preds = np.zeros((len(loader.dataset), n_tasks))
    labels = np.zeros((len(loader.dataset), n_tasks))

    for data in loader:
        x, ys, _ = (datum for datum in data)
        x = x.to(device)
        ys = torch.stack(ys, dim=1).to(device=device, dtype=torch.float32)
        bs = x.shape[0]

        logits = model(x)

        losses_per_task += criterion(logits, ys).sum(dim=0).cpu()
        acc_per_task += ((logits >= 0.0).eq(ys)).sum(dim=0).cpu()  # gender

        preds[data_size: data_size + bs, :] = torch.sigmoid(logits).cpu().numpy()
        labels[data_size: data_size + bs, :] = ys.cpu().numpy()

        data_size += bs

    model.train()

    # Report meV instead of eV.
    avg_task_losses = losses_per_task.numpy() / data_size
    avg_task_accs = acc_per_task.numpy() / data_size
    auc_per_task = roc_auc_score(labels, preds, average=None)

    delta_m = delta_fn(auc_per_task, dataset=dataset)

    return dict(
        avg_task_losses=avg_task_losses,
        avg_task_accs=avg_task_accs,
        auc_per_task=auc_per_task,
        delta_m=delta_m
    )


def main(args, device):

    n_tasks = 14
    model = ResNet34(n_tasks=n_tasks, activation=args.activation).to(device)

    train_loader, val_loader, test_loader = (
        build_loader(data_root=args.data_path.as_posix(),
                     train_csv_path='./dataset/partition/train.csv',
                     validation_csv_path='./dataset/partition/validation.csv',
                     test_csv_path='./dataset/partition/test.csv',
                     batch_size=args.batch_size, test_batch_size=args.batch_size * 2,
                     num_workers=args.n_workers))

    criterion = nn.BCEWithLogitsLoss(reduction="none",)

    # weight method
    method = BayesAggMTL(num_tasks=n_tasks, n_outputs_per_task_group=[n_tasks],
                         task_types=['binary_tasks'],
                         agg_scheme_hps={},
                         cls_hps={'gamma': args.gamma, 'sqrt_power': args.sqrt_power_cls,
                                  'num_mc_samples': args.num_mc_samples})

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.shared_parameters(), lr=args.lr),
            dict(params=model.task_specific_parameters(), lr=args.lr),
        ],
        lr=args.lr,
        weight_decay=args.wd,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.n_epochs * 0.6), int(args.n_epochs * 0.8)], gamma=0.1
    )
    epochs = args.n_epochs
    epoch_iter = trange(epochs)

    best_val_task_losses = best_test_task_losses = np.inf, np.inf
    val_loss, test_loss = np.inf, np.inf
    val_delta, test_delta, best_test_delta, best_val_delta = np.inf, np.inf, np.inf, np.inf
    best_val_task_aucs = np.asarray([0] * n_tasks)
    best_test_task_aucs = np.asarray([0] * n_tasks)
    best_val_task_accs = np.asarray([0] * n_tasks)
    best_test_task_accs = np.asarray([0] * n_tasks)
    train_features = train_labels = None

    step = -1

    for epoch in epoch_iter:
        lr = scheduler.optimizer.param_groups[0]["lr"]
        for j, data in enumerate(train_loader):
            model.train()
            step += 1

            x, ys, _ = (datum for datum in data)
            x = x.to(device)
            ys = torch.stack(ys, dim=1).to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            logits, features = model(x, return_representation=True)
            losses = criterion(logits, ys).mean(dim=0)

            # calculate weighted loss + update
            if epoch < args.ls_epochs:
                loss = losses.sum()
                loss.backward()
            else:
                method.backward(
                    losses=losses,
                    last_layer_params=list(model.task_specific_parameters()),
                    representation=features,
                    labels=[ys],
                    full_train_features=train_features if j == 0 else None,
                    full_train_labels=[train_labels] if j == 0 else None
                )

            torch.nn.utils.clip_grad_norm_(model.task_specific_parameters(), 50)
            # if epoch >= args.ls_epochs:
            #     torch.nn.utils.clip_grad_norm_(model.shared_parameters(), 1)

            optimizer.step()

            # logging
            description = f"Mean train loss: {losses.mean().item():.3f}, lr: {lr:.5f}"
            epoch_iter.set_description(description)

            online_train_features = features.detach().clone() if j == 0 else \
                torch.cat((online_train_features, features.detach().clone()), dim=0)
            online_train_labels = ys if j == 0 else torch.cat((online_train_labels, ys), dim=0)

        train_features = online_train_features.clone()
        train_labels = online_train_labels.clone()

        scheduler.step()
        if (epoch + 1) % args.eval_every == 0:
            logging.info("\nEvaluate on Val-Test")

            val_loss_dict = evaluate(model, val_loader, n_tasks=n_tasks, dataset="val")
            val_loss = val_loss_dict["avg_task_losses"]
            val_acc = val_loss_dict["avg_task_accs"]
            val_auc = val_loss_dict["auc_per_task"]
            val_delta = val_loss_dict["delta_m"]

            test_loss_dict = evaluate(model, test_loader, n_tasks=n_tasks, dataset="test")
            test_loss = test_loss_dict["avg_task_losses"]
            test_acc = test_loss_dict["avg_task_accs"]
            test_auc = test_loss_dict["auc_per_task"]
            test_delta = test_loss_dict["delta_m"]

            best_val_criteria = val_delta <= best_val_delta

            if best_val_criteria:
                best_val_task_losses = val_loss
                best_val_task_accs = val_acc
                best_val_delta = val_delta
                best_val_task_aucs = val_auc

                best_test_task_losses = test_loss
                best_test_task_accs = test_acc
                best_test_delta = test_delta
                best_test_task_aucs = test_auc

        print_dict = {}
        print_dict.update(parse_eval_summary("loss", "test", test_loss))
        print_dict.update(parse_eval_summary("auc", "test", test_auc))
        print_dict.update(parse_eval_summary("best_loss", "test", best_test_task_losses))
        print_dict.update(parse_eval_summary("best_auc", "test", best_test_task_aucs))
        print_dict.update(
            {
                "val_delta": val_delta,
                "best_val_delta": best_val_delta,
                "test_delta": test_delta,
                "best_test_delta": best_test_delta,
                "mean_test_loss": test_loss.mean().item(),
                "mean_test_auc": test_auc.mean().item(),
            }
        )

        logging.info(print_dict)


if __name__ == "__main__":
    parser = ArgumentParser("ChestX-ray14", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset/ChestX-ray",
        lr=1e-3,
        wd=0.0,
        n_epochs=100,
        batch_size=256,
        n_workers=8,
        ls_epochs=1,
        gamma=0.005,
        sqrt_power_cls=0.0005,
    )

    parser.add_argument("--activation",
                        default="elu",
                        type=str, choices=["relu", "elu"],
                        help="last layer activation")

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(args=args, device=device)
