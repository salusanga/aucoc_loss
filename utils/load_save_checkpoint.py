import os
import torch


def load_checkpoint(net, optimizer, scheduler, model_name, seed, checkpoint_dir):
    """Load pre-trained checkpoint."""
    checkpoint_file = "{}/{}_{:02d}_last_epoch.pth".format(
        checkpoint_dir, model_name, seed
    )
    if os.path.isfile(checkpoint_file):
        print("=> Loading checkpoint from '{}'".format(checkpoint_file))
        checkpoint_dict = torch.load(checkpoint_file)
        start_epoch = checkpoint_dict["epoch"] + 1
        net.load_state_dict(checkpoint_dict["net_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        train_stats_best = checkpoint_dict["train_stats_best"]
        val_stats_best = checkpoint_dict["val_stats_best"]
        settings = checkpoint_dict["settings"]
        print(
            "=> Loaded checkpoint of '{}' (epoch {})".format(
                checkpoint_file, checkpoint_dict["epoch"]
            )
        )
    else:
        print("=> No checkpoint found at '{}'".format(checkpoint_file))

    return (
        settings,
        start_epoch,
        net,
        optimizer,
        scheduler,
        train_stats_best,
        val_stats_best,
    )


def save_checkpoint(
    epoch,
    net,
    optimizer,
    scheduler,
    train_stats_best,
    val_stats_best,
    settings,
    checkpoint_dir,
    checkpoint_file,
):
    """Saves a checkpoint of the network and other variables."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    state = {
        "epoch": epoch,
        "net_state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "settings": settings,
        "train_stats_best": train_stats_best,
        "val_stats_best": val_stats_best,
    }

    torch.save(state, checkpoint_file)
    print("Saving checkpoint at {}.".format(checkpoint_file))


def load_model(net, settings):
    """Load pre-trained model."""
    checkpoint_pretrained = "{}/{}_{:02d}_last_epoch.pth".format(
        settings.checkpoint_pretrained_dir,
        settings.model_pretrained_name,
        settings.seed,
    )
    if os.path.isfile(checkpoint_pretrained):
        print("=> Loading checkpoint from '{}'".format(checkpoint_pretrained))
        checkpoint_dict = torch.load(checkpoint_pretrained)
        net.load_state_dict(checkpoint_dict["net_state_dict"])
        print("=> Loaded checkpoint from '{}'.".format(checkpoint_pretrained))
    else:
        print("=> No checkpoint found at '{}'".format(checkpoint_pretrained))

    return net
