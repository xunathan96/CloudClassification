from transforms import *
from utils import *
from dataset import *
from model import *
from catalyst.contrib.criterion import DiceLoss, IoULoss

logdir = "../logs/segmentation/" + NETWORK + "_" + ENCODER

def main():
    model, dataset, loaders = initialize_model(NETWORK, 'train')

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3}
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }

    device = utils.get_device()
    print(f"device: {device}")
    runner = SupervisedRunner(device=device)

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # Aggregate all losses into one loss based on weights.
        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="weighted_sum", # can be "sum", "weighted_sum" or "mean"
            loss_keys={"loss_dice": 1, "loss_iou": 0, "loss_bce": 1}
        ),

        # metrics
        DiceCallback(),
        IouCallback(),
        EarlyStoppingCallback(patience=5, min_delta=0.001)
    ]
    
    num_epochs = 25

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    # PLOT LOSS/SCORE GRAPHS
    utils.plot_metrics(
        logdir=logdir,
        # specify which metrics we want to plot
        metrics=["loss", "dice", "bce", 'lr', '_base/lr']
    )

    # FREE GPU CACHE
    del model, dataset, loaders, runner
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
