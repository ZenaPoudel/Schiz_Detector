import matplotlib.pyplot as plt
import argparse

def plot_curve(epoch_loss_values, epoch_train_accuracy_values, metric_values):
  
  parser = argparse.ArgumentParser(description='view loss and accuracy curve')
  parser.add_argument('--epoch_loss_values', type=int, default=[0,0,0])
  parser.add_argument('--epoch_train_accuracy_values', type=int, default=[0,0,0])
  parser.add_argument('--metric_values', type=int, default=[0,0,0])
  
  plt.figure('train', (12,6))
  plt.subplot(1,2,1)
  plt.title("Epoch Average Loss")
  x = [i+1 for i in range(len(args.epoch_loss_values))]
  y = args.epoch_loss_values
  plt.xlabel('epoch')
  plt.plot(x, y)
  plt.show()
  plt.title("Training and Validation: Accuracy_curve")
  x1 = [i+1 for i in range(len(args.epoch_train_accuracy_values))]
  y1 = args.epoch_train_accuracy_values
  plt.xlabel('epoch')
  plt.plot(x1, y1)
  
  plt.title("Validation: Accuracy_Curve")
  x2 = [(i+1) for i in range(len(args.metric_values))]
  y2 = args.metric_values
  plt.xlabel('epoch')
  plt.plot(x2,y2)
  plt.show() 		
  hold off
