import argparse
import numpy
import torch
from torch.utils.data import DataLoader
from utils import EarlyStopping
from train_sampling import *


def main(args):
    # acm data
    if args["dataset"] == "ACMRaw":
        from utils import load_data

        (
            g,
            features,
            labels,
            n_classes,
            train_nid,
            val_nid,
            test_nid,
            train_mask,
            val_mask,
            test_mask,
        ) = load_data("ACMRaw")
        metapath_list = [["pa", "ap"], ["pf", "fp"]]
    else:
        raise NotImplementedError(
            "Unsupported dataset {}".format(args["dataset"])
        )

    # Is it need to set different neighbors numbers for different meta-path based graph?
    num_neighbors = args["num_neighbors"]
    han_sampler = HANSampler(g, metapath_list, num_neighbors)
    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid,
        batch_size=args["batch_size"],
        collate_fn=han_sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    model = HAN(
        num_metapath=len(metapath_list),
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=n_classes,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    ).to(args["device"])

    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total trainable params: {:d}".format(total_trainable_params))

    stopper = EarlyStopping(patience=args["patience"])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    for epoch in range(args["num_epochs"]):
        model.train()
        for step, (seeds, blocks) in enumerate(dataloader):
            h_list = load_subtensors(blocks, features)
            blocks = [block.to(args["device"]) for block in blocks]
            hs = [h.to(args["device"]) for h in h_list]

            logits = model(blocks, hs)
            loss = loss_fn(
                logits, labels[numpy.asarray(seeds)].to(args["device"])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print info in each batch
            train_acc, train_micro_f1, train_macro_f1 = score(
                logits, labels[numpy.asarray(seeds)]
            )
            print(
                "Epoch {:d} | loss: {:.4f} | train_acc: {:.4f} | train_micro_f1: {:.4f} | train_macro_f1: {:.4f}".format(
                    epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1
                )
            )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model,
            g,
            metapath_list,
            num_neighbors,
            features,
            labels,
            val_nid,
            loss_fn,
            args["batch_size"],
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
            "Epoch {:d} | Val loss {:.4f} | Val Accuracy {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model,
        g,
        metapath_list,
        num_neighbors,
        features,
        labels,
        test_nid,
        loss_fn,
        args["batch_size"],
    )
    print(
        "Test loss {:.4f} | Test Accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_acc, test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mini-batch HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_neighbors", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_heads", type=list, default=[8])
    parser.add_argument("--hidden_units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ACMRaw")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--client_nums", type=int, default=20)

    args = parser.parse_args().__dict__
    # set_random_seed(args['seed'])

    main(args)
