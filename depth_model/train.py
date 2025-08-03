import torch
from torch import nn
from torchvision.transforms import ToTensor

import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile
from pathlib import Path

from depth_model.dataset import DepthDataset
from depth_model.segnet import SegNet


def main(num_samples=10, max_num_epochs=10):
    # if not ray.is_initialized():
    #     ray.init(num_cpus=2, num_gpus=1 if torch.cuda.is_available() else 0)

    # config = {
    #     "features": tune.choice([16, 32, 64]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([4, 8, 16, 32])
    # }
    config = {"features": 32, "lr": 0.001, "batch_size": tune.choice([16, 32])}

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        train_depth,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        },
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        verbose=2,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = SegNet(
        in_channels=3, out_channels=1, features=best_trial.config["features"]
    )
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(
        trial=best_trial,
        metric="accuracy",
        mode="min",  # AbsRel should be minimized
    )
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        test_acc = test_accuracy(best_trained_model)
        print("Best trial test set accuracy: {}".format(test_acc))

        torch.save(
            {
                "model_state_dict": best_trained_model.state_dict(),
                "config": best_trial.config,
                "test_accuracy": test_acc,
            },
            "best_depth_model.pth",
        )


def train_depth(config):
    model = SegNet(in_channels=3, out_channels=1, features=config["features"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    torch.manual_seed(42)
    dataset = DepthDataset(
        images_dir="/content/data/depth_data/images",
        depth_maps_dir="/content/data/depth_data/depth_maps",
        transform=ToTensor(),
    )
    trainset, valset, _ = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    for epoch in range(start_epoch, 10):
        for (X, y) in train_loader:
            X, y = X.to(device).float(), y.to(device).float()

            # zero the parameter gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            model.train()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # validation
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        epsilon = 1e-8
        for (X, y) in val_loader:
            with torch.inference_mode():
                X, y = X.to(device).float(), y.to(device).float()
                model.eval()
                y_pred = model(X)
                loss = criterion(y_pred, y)
            val_loss += loss.cpu().numpy()
            val_steps += 1
            val_acc += torch.mean(
                torch.abs(y_pred.squeeze() - y) / (y + epsilon)
            ).item()

        # Report metrics to Ray Tune
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {
                    "loss": val_loss / val_steps,
                    "accuracy": val_acc / val_steps,
                },
                checkpoint=checkpoint,
            )


def test_accuracy(model):
    torch.manual_seed(42)
    dataset = DepthDataset(
        images_dir="/content/data/depth_data/images",
        depth_maps_dir="/content/data/depth_data/depth_maps",
        transform=ToTensor(),
    )
    _, _, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=4, shuffle=False, num_workers=0
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    test_acc = 0.0
    test_steps = 0
    epsilon = 1e-8
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device).float(), y.to(device).float()
            model.eval()
            y_pred = model(X)
            test_acc += torch.mean(
                torch.abs(y_pred.squeeze() - y) / (y + epsilon)
            ).item()
            test_steps += 1

    return test_acc / test_steps


if __name__ == "__main__":
    main(num_samples=2, max_num_epochs=10)
