from argparse import ArgumentParser
import logging
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import trange
from pathlib import Path
from experiments.utkface.data import UTKFacesData
from experiments.utkface.models import ResNetEncoder
from BayesAgg_MTL import BayesAggMTL
from experiments.utkface.calc_delta import delta_fn

from experiments.utils import (
    get_device,
    common_parser,
    set_seed,
    set_logger
)

set_logger()


class UTKFaceMultiTaskLoss(nn.modules.loss._Loss):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target) -> torch.Tensor:
        # TODO: Note that the target is not scaled back to original values by the std

        loss = []
        loss.append(self.mse(input[0].squeeze(), target[0].squeeze()))
        input = input[1:]
        target = target[1:]
        if isinstance(input, (list, tuple)):
            for i, t in zip(input, target):
                loss.append(self.ce(i, t))
        elif isinstance(input, torch.Tensor):
            loss.append(self.ce(input, target))
        else:
            raise Exception("unrecognized input type for loss calculation")

        return torch.stack(loss, dim=1)


def parse_eval_summary(metric, split, values):
    d = {}
    d.update({f"{split}/{metric}_task_{i}": j for i, j in enumerate(values)})
    return d


@torch.no_grad()
def evaluate(model, loader, n_tasks, dataset="val", mean_w_age=None):

    model.eval()
    data_size = 0.0
    losses_per_task = torch.zeros(n_tasks, dtype=torch.float32, device=device)
    acc_per_task = torch.zeros(n_tasks, dtype=torch.float32, device=device)
    loss_module = UTKFaceMultiTaskLoss()
    logits_gr = []
    targets_gr = []

    for data in loader:
        x, a, g, r = (datum.to(device) for datum in data)
        logits, features = model(x, return_representation=True)
        if mean_w_age is not None:
            if args.include_bias:
                features = torch.cat((features, torch.ones(features.shape[0],
                                                           device=features.device).unsqueeze(-1)), dim=-1)
            logits[0] = features @ mean_w_age.t()

        losses_per_task += loss_module.forward(logits, (a, g, r)).sum(0)
        acc_per_task[1] += logits[1].argmax(1).eq(g).sum(dim=0).cpu().item()  # gender
        acc_per_task[2] += logits[2].argmax(1).eq(r).sum(dim=0).cpu().item()  # race

        data_size += x.shape[0]
        logits_gr.append(torch.cat((logits[1], logits[2]), dim=1))
        targets_gr.append(torch.cat((g.view(-1, 1), r.view(-1, 1)), dim=1))

    model.train()

    logits_gr = torch.cat(logits_gr)
    targets_gr = torch.cat(targets_gr)

    avg_task_losses = losses_per_task.detach().cpu().numpy() / data_size
    avg_task_accs = acc_per_task.detach().cpu().numpy() / data_size
    delta_m = delta_fn(np.asarray([avg_task_losses[0], avg_task_accs[1], avg_task_accs[2]]), dataset=dataset)

    return dict(
        avg_task_losses=avg_task_losses,
        avg_task_accs=avg_task_accs,
        delta_m=delta_m
    )


def main(args, device):

    n_tasks = 3
    model = ResNetEncoder(n_tasks=n_tasks, activation=args.activation).to(device)

    dataset = UTKFacesData(data_dir=args.data_path.as_posix())
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=args.batch_size,
                                                                test_batch_size=args.batch_size * 2,
                                                                num_workers=args.n_workers)

    loss_module = UTKFaceMultiTaskLoss()

    # weight method
    method = BayesAggMTL(num_tasks=n_tasks, n_outputs_per_task_group=[1, 2, 5],
                         task_types=['regression', 'multiclass', 'multiclass'],
                         agg_scheme_hps={},
                         reg_hps={'sqrt_power': args.sqrt_power, 'obs_noise': args.obs_noise},
                         cls_hps={'gamma': args.gamma, 'sqrt_power': args.sqrt_power_cls,
                                  'num_mc_samples': args.num_mc_samples,}
                        )

    optimizer = torch.optim.Adam(
        [
            dict(params=model.shared_parameters(), lr=args.lr),
            dict(params=model.task_specific_parameters(), lr=args.lr),
        ],
        weight_decay=args.wd
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.n_epochs * 0.6), int(args.n_epochs * 0.8)], gamma=0.1
    )

    epoch_iterator = trange(args.n_epochs)

    best_val_task_losses = best_test_task_losses = np.inf, np.inf
    val_loss, test_loss = np.inf, np.inf
    val_delta, test_delta, best_test_delta, best_val_delta = np.inf, np.inf, np.inf, np.inf
    best_val_task_accs = np.asarray([0] * 3)
    best_test_task_accs = np.asarray([0] * 3)
    step = -1
    test_logits_gr, test_targets_gr = None, None

    for epoch in epoch_iterator:
        lr = scheduler.optimizer.param_groups[0]["lr"]
        for j, data in enumerate(train_loader):
            model.train()
            step += 1

            x, a, g, r = (datum.to(device) for datum in data)
            optimizer.zero_grad()

            logits, features = model(x, return_representation=True)
            losses = loss_module.forward(logits, (a, g, r)).mean(dim=0)

            if epoch < args.ls_epochs:
                loss = losses.sum()
                loss.backward()
            else:
                method.backward(
                    losses=losses,
                    last_layer_params=list(model.task_specific_parameters()),
                    representation=features,
                    labels=[a.unsqueeze(-1), g, r],
                    full_train_features=train_features if j == 0 else None,
                    full_train_labels=train_labels if j == 0 else None
                )

            torch.nn.utils.clip_grad_norm_(model.task_specific_parameters(), 10)
            if epoch >= args.ls_epochs:
                torch.nn.utils.clip_grad_norm_(model.shared_parameters(), 5)

            optimizer.step()

            description = f"Mean train loss: {torch.mean(losses).item():.3f}"
            epoch_iterator.set_description(description)

            online_train_features = features.detach().clone() if j == 0 else \
                torch.cat((online_train_features, features.detach().clone()), dim=0)
            online_train_labels_a = a.unsqueeze(-1) if j == 0 else torch.cat(
                (online_train_labels_a, a.unsqueeze(-1)), dim=0)
            online_train_labels_g = g if j == 0 else torch.cat((online_train_labels_g, g), dim=0)
            online_train_labels_r = r if j == 0 else torch.cat((online_train_labels_r, r), dim=0)

        scheduler.step()

        train_features = online_train_features.clone()
        train_labels = [online_train_labels_a.clone(),
                        online_train_labels_g.clone(),
                        online_train_labels_r.clone()]

        mean_w_age = None
        if epoch >= (args.ls_epochs - 1):
            logging.info("\nCompute full data posterior")
            output_dim = method.posterior_modules[0].num_outputs
            full_data_prior_mean, full_data_prior_precision = \
                method.posterior_modules[0].prior(train_features, output_dim)
            full_data_posterior = method.posterior_modules[0].posterior(train_features, train_labels[0],
                                                                        full_data_prior_mean,
                                                                        full_data_prior_precision)
            mean_w_age = full_data_posterior.mean

        if (epoch + 1) % args.eval_every == 0:
            logging.info("Evaluate on Val-Test")

            val_loss_dict = evaluate(model, val_loader, n_tasks=n_tasks, dataset="val", mean_w_age=mean_w_age)
            val_loss = val_loss_dict["avg_task_losses"]
            val_acc = val_loss_dict["avg_task_accs"]
            val_delta = val_loss_dict["delta_m"]

            test_loss_dict = evaluate(model, test_loader, n_tasks=n_tasks, dataset="test", mean_w_age=mean_w_age)
            test_loss = test_loss_dict["avg_task_losses"]
            test_acc = test_loss_dict["avg_task_accs"]
            test_delta = test_loss_dict["delta_m"]

            if val_delta <= best_val_delta:
                best_val_task_losses = val_loss
                best_val_task_accs = val_acc
                best_val_delta = val_delta

                best_test_task_losses = test_loss
                best_test_task_accs = test_acc
                best_test_delta = test_delta

            print_dict = {}
            print_dict.update(parse_eval_summary("loss", "test", test_loss))
            print_dict.update(parse_eval_summary("acc", "test", test_acc))
            print_dict.update(parse_eval_summary("best_loss", "test", best_test_task_losses))
            print_dict.update(parse_eval_summary("best_acc", "test", best_test_task_accs))
            print_dict.update(
                {
                    "val_delta": val_delta,
                    "best_val_delta": best_val_delta,
                    "test_delta": test_delta,
                    "best_test_delta": best_test_delta,
                }
            )

            logging.info(print_dict)


if __name__ == "__main__":

    from os.path import exists

    parser = ArgumentParser("UTKFace", parents=[common_parser])
    parser.set_defaults(
        data_path="./dataset/UTKFace",
        lr=1e-3,
        wd=1e-4,
        n_epochs=100,
        batch_size=128,
        ls_epochs=5,
        n_workers=8,
    )
    parser.add_argument("--activation",
                        default="elu",
                        type=str, choices=["relu", "elu"],
                        help="last layer activation")

    args = parser.parse_args()
    device = get_device(gpus=args.gpu)
    main(args=args, device=device)