import argparse
import torch
from ignite.metrics import Accuracy, Precision, Recall
from torch.nn.modules.loss import NLLLoss
from model import model_3DCNN
from data_loader import data_pull_and_load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

device = torch.device("cuda" ) if torch.cuda.is_available() else "cpu"

def main():
	parser = argparse.ArgumentParser(description='Base_paper 1 3D convolution for action recognition')
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--epoch', type=int, default=50)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	parser.add_argument('--dropout', type=float, default=0.3)

	parser.add_argument('--data_source', type=str, default='MCIC')
	parser.add_argument('--mri_type', type=str, default='T1')
	parser.add_argument('--pix_dimension', type=int, default=[2,2,2])    
	parser.add_argument('--resize_spatial_size', type=int, default=[99,99,99])
	parser.add_argument('--test_split', type=float, default=0.3)
	parser.add_argument('--ratio', type=str, default='NO')

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
	optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
	val_interval = 1
	best_metric = -1
	best_metric_epoch = -1
	train_best_metric = -1
	train_best_metric_epoch = -1
	train_accuracy = -1
	epoch_loss_values = []
	epoch_train_accuracy_values = []
	epoch_train_precision_values = []
	epoch_train_recall_values = []
	epoch_train_F1_values = []
	epoch_val_accuracy_values = []
	epoch_val_precision_values = []
	epoch_val_recall_values = []
	epoch_val_F1_values = []
	# writer = SummaryWriter()
	max_epochs = args.epoch

	for epoch in range(max_epochs):
		print("-" * 10)
		print(f"epoch {epoch + 1}/{max_epochs}")
		model.train()
		epoch_loss = 0
		epoch_train_accuracy = 0
		epoch_train_precision = 0
		epoch_train_recall= 0
		epoch_train_F1 = 0
		epoch_val_accuracy = 0
		epoch_val_precision = 0
		epoch_val_recall= 0
		epoch_val_F1 = 0
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

			accuracy = Accuracy()
			accuracy.reset()
			accuracy.update((outputs.argmax(dim=1), labe))
			batch_acc = accuracy.compute()
			epoch_train_accuracy += batch_acc

			precision = Precision()
			precision.reset()
			precision.update((outputs.argmax(dim=1), labe))
			batch_precision = precision.compute()
			epoch_train_precision += batch_precision

			recall = Recall()
			recall.reset()
			recall.update((outputs.argmax(dim=1), labe))
			batch_recall = recall.compute()
			epoch_train_recall += batch_recall

			batch_F1 = (batch_precision * batch_recall * 2 / (batch_precision + batch_recall))
			if batch_F1!=batch_F1:
				batch_F1 = 0
			else: 
				batch_F1 =batch_F1

			epoch_train_F1 += batch_F1

			print(f"{step}/{epoch_len}, train loss: {loss.item():.4f}, train_accuracy: {batch_acc:.4f}, train_precision: {batch_precision:.4f}, train_recall: {batch_recall:.4f}, train_F1: {batch_F1:.4f}")

		epoch_loss /= step
		epoch_loss_values.append(epoch_loss)
		epoch_train_accuracy /= step
		epoch_train_accuracy_values.append(epoch_train_accuracy)
		epoch_train_precision /= step
		epoch_train_precision_values.append(epoch_train_precision)
		epoch_train_recall /= step
		epoch_train_recall_values.append(epoch_train_recall)
		epoch_train_F1 /= step
		epoch_train_F1_values.append(epoch_train_F1)
		print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, average accuracy: {epoch_train_accuracy:.4f}, average precision: { epoch_train_precision:.4f}, average recall: {epoch_train_recall:.4f}, average F1: {epoch_train_F1:.4f}")


		if epoch_train_F1 > train_best_metric:
			train_best_metric = epoch_train_F1
			train_best_metric_epoch = epoch + 1
			# torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
			# print("saved new training best metric model")
		print(f"Best train F1 score: {train_best_metric:.4f} at epoch {train_best_metric_epoch}")

		if (epoch + 1) % val_interval == 0:
			model.eval()
			num_correct = 0.0
			metric_count = 0
			val_step = 0
			for val_data in val_loader:
				val_step +=1
				val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
				with torch.no_grad():
					val_outputs = model(val_images)
					val_accuracy = Accuracy()
					val_accuracy.reset()
					val_accuracy.update((val_outputs, val_labels))
					val_batch_acc = val_accuracy.compute()
					epoch_val_accuracy += val_batch_acc

					val_precision = Precision()
					val_precision.reset()
					val_precision.update((val_outputs.argmax(dim=1), val_labels))
					val_batch_precision = val_precision.compute()
					epoch_val_precision += val_batch_precision

					val_recall = Recall()
					val_recall.reset()
					val_recall.update((val_outputs.argmax(dim=1), val_labels))
					val_batch_recall = val_recall.compute()
					epoch_val_recall += val_batch_recall

					val_batch_F1 = (val_batch_precision * val_batch_recall * 2 / (val_batch_precision + val_batch_recall))
					if val_batch_F1!=val_batch_F1:
						val_batch_F1 = 0
					else: 
						val_batch_F1 =val_batch_F1

					epoch_val_F1 += val_batch_F1
					print(f"{val_step}, val_accuracy: {val_batch_acc:.4f}, val_precision: {val_batch_precision:.4f}, val_recall: {val_batch_recall:.4f}, val_F1: {val_batch_F1:.4f}")

			epoch_val_accuracy /= val_step
			epoch_val_accuracy_values.append(epoch_val_accuracy)
			epoch_val_precision /= val_step
			epoch_val_precision_values.append(epoch_val_precision)
			epoch_val_recall /= val_step
			epoch_val_recall_values.append(epoch_val_recall)
			epoch_val_F1 /= val_step
			epoch_val_F1_values.append(epoch_val_F1)

			if epoch_val_F1 > best_metric:
				best_metric = epoch_val_F1
				best_metric_epoch = epoch + 1
				# torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
				print("saved new validation best metric model")

			print(f"val accuracy: {epoch_val_accuracy:.4f}, val precision: { epoch_val_precision:.4f}, val recall: {epoch_val_recall:.4f}, val F1: {epoch_val_F1:.4f}")
			print(f"Best validation F1 score: {best_metric:.4f} at epoch {best_metric_epoch}")
		# # writer.add_scalar("val_accuracy", metric, epoch + 1)

	print(f"Training completed, training best_metric: {train_best_metric:.4f} at epoch: {train_best_metric_epoch}, validation best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


	plt.figure('train', (12,6))
	plt.subplot(1,2,1)
	plt.title("Epoch Average Loss")
	x = [i+1 for i in range(len(epoch_loss_values))]
	y = epoch_loss_values
	x1 = [i+1 for i in range(len(epoch_train_accuracy_values))]
	y1 = epoch_train_accuracy_values
	x2 = [i+1 for i in range(len(epoch_train_F1_values))]
	y2 = epoch_train_F1_values
	
	x3 = [(i+1) for i in range(len(epoch_val_accuracy_values))]
	y3 = epoch_val_accuracy_values
	x4 = [i+1 for i in range(len(epoch_val_F1_values))]
	y4 = epoch_train_F1_values
	
	plt.plot(x, y, label='Training Loss')
	
	plt.plot(x1,y1, label='Training Accuracy')

	plt.plot(x2, y2, label='Training F1 score')
	
	plt.plot(x3, y3, label='Validation Accuracy')
	
	plt.plot(x4, y4, label='epoch_train_F1_values')
	
	plt.title('Training and Validation Accurccy')

	plt.legend(loc='upper right')
	plt.savefig('TrainingandValidationAccurccy.png')


if __name__ == '__main__':
	main()
