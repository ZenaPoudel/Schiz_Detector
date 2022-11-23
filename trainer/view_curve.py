import matplotlib.pyplot as plt

def plot_curve(epoch_loss_values, epoch_train_accuracy_values, metric_values)
  plt.figure('train', (12,6))
  plt.subplot(1,2,1)
  plt.title("Epoch Average Loss")
  x = [i+1 for i in range(len(epoch_loss_values))]
  y = epoch_loss_values
  plt.xlabel('epoch')
  plt.plot(x, y)
  plt.show()
  plt.title("Training and Validation: Accuracy_curve")
  x1 = [i+1 for i in range(len(epoch_train_accuracy_values))]
  y1 = epoch_train_accuracy_values
  plt.xlabel('epoch')
  plt.plot(x1, y1)
  plt.show()
  hold on
  plt.title("Validation: Accuracy_Curve")
  x2 = [(i+1) for i in range(len(metric_values))]
  y2 = metric_values
  plt.xlabel('epoch')
  plt.plot(x2,y2)
  plt.show() 		
  hold off
