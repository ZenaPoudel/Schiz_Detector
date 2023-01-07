import argparse
import torch
from ignite.metrics import Accuracy, Precision, Recall
from torch.nn.modules.loss import NLLLoss
from model import model_3DCNN
from data_loader import data_pull_and_load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import sys
random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(random_seed)


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
	parser.add_argument('--resize_spatial_size', type=str, default='99,99,99')
	parser.add_argument('--test_split', type=float, default=0.3)
	parser.add_argument('--ratio', type=str, default='NO')

	args = parser.parse_args()
	resize_spatial_size = [int(item) for item in args.resize_spatial_size.split(',')]
	train_ds, train_loader, val_loader = data_pull_and_load(
	      data_source= args.data_source, 
	      mri_type = args.mri_type, 
	      pix_dimension=(args.pix_dimension[0], args.pix_dimension[1], args.pix_dimension[2]), 
	      resize_spatial_size=(resize_spatial_size[0], resize_spatial_size[1], resize_spatial_size[2]),
	      test_split=args.test_split,
	      batch_size=args.batch_size,
	      ratio=args.ratio
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
	epoch_train_balace_accuracy_values=[]
	epoch_train_precision_values = []
	epoch_train_recall_values = []
	epoch_train_specificity_values = []
	epoch_train_F1_values = []
	epoch_val_accuracy_values = []
	epoch_val_balace_accuracy_values=[]
	epoch_val_precision_values = []
	epoch_val_recall_values = []
	epoch_val_specificity_values = []
	epoch_val_F1_values = []
	# writer = SummaryWriter()
	max_epochs = 2


	for epoch in range(max_epochs):
	  y_pred = []
	  y_true = []
	  val_y_pred = []
	  val_y_true = []
	  print("-" * 10)
	  print(f"epoch {epoch + 1}/{max_epochs}")
	  model.train()
	  epoch_loss = 0
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
		i=[1,0]
		labels.append(i)
	      elif i == 1:
		i=[0,1]
		labels.append(i)
	    label = np.array(labels)
	    labels = torch.from_numpy(label)
	    labels = labels.float().to(device)

	# 			print(labels, outputs)

	    loss = loss_function(outputs, labels)
	    loss.backward()
	    optimizer.step()
	    epoch_loss += loss.item()
	    epoch_len = len(train_ds) // train_loader.batch_size


	# 			print(f"train loss: {loss.item():.4f}")

	    output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
	    y_pred.extend(output) # Save Prediction


	    labels = labe.data.cpu().numpy()
	    y_true.extend(labels) # Save Truth


	  epoch_loss /= step
	  epoch_loss_values.append(epoch_loss)

	  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	  acc = accuracy_score(y_true, y_pred)
	  epoch_train_accuracy_values.append(acc)


	  bal_acc= balanced_accuracy_score(y_true, y_pred)
	  epoch_train_balace_accuracy_values.append(bal_acc)

	  precision = precision_score(y_true, y_pred)
	  epoch_train_precision_values.append(precision)

	  recall= recall_score(y_true, y_pred)
	  epoch_train_precision_values.append(recall)

	  specificity = tn / (tn+fp)
	  epoch_train_specificity_values.append(specificity)

	  f1_score = (precision * recall * 2 )/ (precision +recall)

	  if f1_score!=f1_score:
	    f1_score = 0
	  else: 
	    f1_score =f1_score

	  epoch_train_F1_values.append(f1_score)

	  print(f"epoch {epoch + 1} average epoch loss: {epoch_loss:.4f}, epoch train accuracy: {acc:.4f}, epoch train balanced_Acc:{bal_acc:.4f} epoch train precision: { precision:.4f}, epoch train recall: {recall:.4f}, epoch train F1: {f1_score:.4f}, epoch train specificity: {specificity:.4f}, epoch train auc score: {roc_auc_score(y_true, y_pred)}")

	  if (epoch + 1) % val_interval == 0:
	    model.eval()
	    with torch.no_grad():
	      num_correct = 0.0
	      metric_count = 0
	      val_step = 0
	      for val_data in val_loader:
			val_step +=1
			val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
			val_outputs = model(val_images)

			val_output = (torch.max(torch.exp(val_outputs), 1)[1]).data.cpu().numpy()
			val_y_pred.extend(val_output) # Save Prediction


			val_labels = val_labels.data.cpu().numpy()
			val_y_true.extend(val_labels) # Save Truth

	    vtn, vfp, vfn, vtp = confusion_matrix(val_y_true, val_y_pred).ravel()

	    val_acc = accuracy_score(val_y_true, val_y_pred)
	    epoch_val_accuracy_values.append(val_acc)


	    val_bal_acc= balanced_accuracy_score(val_y_true, val_y_pred)
	    epoch_val_balace_accuracy_values.append(val_bal_acc)

	    val_precision = precision_score(val_y_true, val_y_pred)
	    epoch_val_precision_values.append(val_precision)

	    val_recall= recall_score(val_y_true, val_y_pred)
	    epoch_val_precision_values.append(val_recall)

	    val_specificity = vtn / (vtn+vfp)
	    epoch_val_specificity_values.append(val_specificity)

	    val_f1_score = (val_precision * val_recall * 2 )/ (val_precision + val_recall)

	    if val_f1_score!=val_f1_score:
	      val_f1_score = 0
	    else: 
	      val_f1_score =val_f1_score

	    epoch_val_F1_values.append(val_f1_score)

	    if val_f1_score > best_metric:
	      best_metric = val_f1_score
	      best_metric_epoch = epoch + 1
	      torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
	      print("saved new validation best metric model")

	    print(f"epoch {epoch + 1}, val confusion matrix:{(vtn, vfp, vfn, vtp)} epoch val accuracy: {val_acc:.4f}, epoch val balanced_Acc:{val_bal_acc:.4f} epoch precision: { val_precision:.4f}, epoch recall: {val_recall:.4f}, epoch F1: {val_f1_score:.4f}, epoch specificity: {val_specificity:.4f}, epoch val auc score: {roc_auc_score(val_y_true, val_y_pred)}")

	print(f"Training completed, validation best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



	# customizing runtime configuration stored
	# in matplotlib.rcParams
	plt.rcParams["figure.figsize"] = [7.00, 3.50]
	plt.rcParams["figure.autolayout"] = True

	train_loss_acc = plt.figure('loss_acc', (12,6))
	plt.subplot(1,2,1)
	plt.title("Epoch Average Loss and accuracy")
	x = [i+1 for i in range(len(epoch_loss_values))]
	y = epoch_loss_values
	x1 = [i+1 for i in range(len(epoch_train_accuracy_values))]
	y1 = epoch_train_accuracy_values
	x3 = [(i+1) for i in range(len(epoch_val_accuracy_values))]
	y3 = epoch_val_accuracy_values
	plt.plot(x, y, label='Training Loss')
	plt.plot(x1,y1, label='Training Accuracy')
	plt.plot(x3, y3, label='Validation Accuracy')
	plt.title('Training loss, Training and Validation Accurccy')

	plt.legend(loc='upper left')
	plt.savefig('TrainingandValidationAccurccy.png')

	F1_score = plt.figure('F1', (12,6))
	plt.subplot(1,2,1)

	plt.title("Epoch Average f1")

	x2 = [i+1 for i in range(len(epoch_train_F1_values))]
	y2 = epoch_train_F1_values

	x4 = [i+1 for i in range(len(epoch_val_F1_values))]
	y4 = epoch_val_F1_values


	plt.plot(x2, y2, label='Training F1 score')
	plt.plot(x4, y4, label='validation_F1_score')

	plt.title('Training and Validation F1 score')

	plt.legend(loc='upper left')
	plt.savefig('TrainingandValidationF1.png')

if __name__ == '__main__':
	main()
