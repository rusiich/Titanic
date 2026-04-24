
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def draw_plots(train_losses, val_losses, metrics, lr_changes):

    # Learning rate changes
    plt.plot(range(len(lr_changes)), lr_changes, label='Learning Rate')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Changes of validation and train losses')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()

def build_confusion_matrix(predictions, ground_truth):
    """
    Builds confusion matrix from predictions and ground truth

    predictions: np array of ints, model predictions for all validation samples
    ground_truth: np array of ints, ground truth for all validation samples
    
    Returns:
    np array of ints, (10,10), counts of samples for predicted/ground_truth classes
    """
    
    confusion_matrix = np.zeros((10,10), dtype='int64')

    
    for pred, true in zip(predictions, ground_truth):
        confusion_matrix[pred, true] += 1
    return confusion_matrix

def visualize_confusion_matrix(predictions, ground_truth):
    """
    Visualizes confusion matrix
                     
    """
    # Adapted from 
    # https://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
    

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    size = max(len(np.unique(predictions)), len(np.unique(ground_truth)))
    confusion_matrix = np.zeros((size, size), dtype='int64')

    
    for pred, true in zip(predictions, ground_truth):
        confusion_matrix[true, pred] += 1

    fig = plt.figure(figsize=(size, size))
    plt.title("Confusion matrix")
    plt.ylabel("predicted")
    plt.xlabel("ground truth")
    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')



def save_model(config, model, epoch, current_metric, optimizer, epochs_since_improvement, 
                       name, scheduler, scaler, best_metric):
    '''Save PyTorch model.'''

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'metric': current_metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'best_metric': best_metric,
    }, os.path.join(config.paths.path_to_checkpoints, name))

