import torch
from torch import nn
from torchvision.transforms import ToTensor

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile
from pathlib import Path

from dataset import DepthDataset
from segnet import SegNet


def main():
    num_samples = 30
    max_num_epochs = 10
    grace_period = 1
    reduction_factor = 2

    param_space = {
        "features": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )

    tune_config = tune.TuneConfig(
        num_samples=num_samples,
        scheduler=scheduler,
    )

    train_with_resources = tune.with_resources(
        train_depth, resources={"cpu": 8, "gpu": 1}
    )

    tuner = tune.Tuner(
        train_with_resources,
        tune_config=tune_config,
        param_space=param_space,
        run_config=tune.RunConfig(
            storage_path="~/depth-model/results", name="test_experiment"
        ),
    )
    result = tuner.fit()

    best_result = result.get_best_result("loss", "min", "last")
    print(f"Best result config: {best_result.config}")
    print(f"Best result final validation loss: {best_result.metrics['loss']}")

    best_trained_model = SegNet(
        in_channels=3, out_channels=1, features=best_result.config["features"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        test_loss_value = test_loss(best_trained_model)
        print("Best trial test set loss: {}".format(test_loss_value))

        torch.save(
            {
                "model_state_dict": best_trained_model.state_dict(),
                "config": best_result.config,
                "test_loss": test_loss_value,
            },
            "best_depth_model.pth",
        )


def train_depth(config):
    model = SegNet(in_channels=3, out_channels=1, features=config["features"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        images_dir="~/data/depth_data/images",
        depth_maps_dir="~/data/depth_data/depth_maps",
        transform=ToTensor(),
    )
    train_set, val_set, _ = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    for epoch in range(start_epoch, 10):
        for X, y in train_loader:
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
        val_steps = 0
        for X, y in val_loader:
            with torch.inference_mode():
                X, y = X.to(device).float(), y.to(device).float()
                model.eval()
                y_pred = model(X)
                loss = criterion(y_pred, y)
            val_loss += loss.cpu().numpy()
            val_steps += 1

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
                },
                checkpoint=checkpoint,
            )


def test_loss(model):
    torch.manual_seed(42)
    dataset = DepthDataset(
        images_dir="/home/ubuntu/data/depth_data/images",
        depth_maps_dir="/home/ubuntu/data/depth_data/depth_maps",
        transform=ToTensor(),
    )
    _, _, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss = 0.0
    test_steps = 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device).float(), y.to(device).float()
            model.eval()
            y_pred = model(X)
            test_loss += torch.mean((y_pred.squeeze() - y) ** 2).cpu().numpy()
            test_steps += 1

    return test_loss / test_steps


if __name__ == "__main__":
    main()
