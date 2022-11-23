import argparse
import torch
from torch.nn.modules.loss import NLLLoss
from model import model_3DCNN
from data_loader import data_pull_and_load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" ) if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description='Base_paper 1 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--data_source', type=str, default='MCIC')
    parser.add_argument('--mri_type', type=str, default='T1')
    parser.add_argument('--pix_dimension', type=int, default=[2,2,2])    
    parser.add_argument('--resize_spatial_size', type=int, default=[99,99,99])
    parser.add_argument('--test_split', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    
    train_ds, train_loader, val_loader = data_pull_and_load(
        data_source= args.data_source, 
        mri_type = args.mri_type, 
        pix_dimension=(args.pix_dimension[0], args.pix_dimension[1], args.pix_dimension[2]), 
        resize_spatial_size=(args.resize_spatial_size[0], args.resize_spatial_size[1], args.resize_spatial_size[2]),
        test_split=args.test_split,
        batch_size=args.batch_size  
    )

    model = model_3DCNN(dropout = args.dropout)

    # loss_function = torch.nn.BCEWithLogitsLoss()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    train_best_metric = -1
    train_best_metric_epoch = -1
    train_accuracy = -1
    epoch_loss_values = []
    epoch_train_accuracy_values = []
    train_metric_values = []
    metric_values = []
    # writer = SummaryWriter()
    max_epochs = args.epoch

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_train_accuracy = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labe = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = []
            for i in labe:
                if i == 0:
                    i = [1,0]
                    labels.append(i)
                elif i == 1:
                    i = [0,1]
                    labels.append(i)
            label = np.array(labels)
            labels = torch.from_numpy(label)
            labels = labels.float().to(device)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            train_num_correct = 0.0
            train_metric_count = 0
            train_value = torch.eq(outputs.argmax(dim=1), labe)
            train_metric_count += len(train_value)
            train_num_correct += train_value.sum().item()
            train_accuracy = train_num_correct/train_metric_count
            epoch_train_accuracy += train_accuracy
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_accuracy: {train_accuracy:.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        # train_metric = train_num_correct / train_metric_count
        # train_metric_values.append(train_metric)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        epoch_train_accuracy /= step
        epoch_train_accuracy_values.append(epoch_train_accuracy)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"epoch {epoch + 1} average accuracy: {epoch_train_accuracy:.4f}")

        if epoch_train_accuracy > train_best_metric:
            train_best_metric = epoch_train_accuracy
            train_best_metric_epoch = epoch + 1
        # torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
        print("saved new training best metric model")
        print(f"Best train accuracy: {train_best_metric:.4f} at epoch {train_best_metric_epoch}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new validation best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best validation accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            # writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, training best_metric: {train_best_metric:.4f} at epoch: {train_best_metric_epoch}")
    print(f"Training completed, validation best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    return epoch_loss_values, epoch_train_accuracy_values, metric_values
if __name__ == '__main__':
	main()
