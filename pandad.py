import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('history.csv')


# plt.subplot(221)
df[['accuracy','val_accuracy']].plot()

plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.yscale('linear')
plt.xscale('linear')
plt.title("Xception Fold \n \n"
          "Training and Validation Accuracy")
plt.tight_layout()
plt.legend(loc='upper right')
plt.grid(True, color='w', linestyle='-', linewidth=2)
plt.gca().patch.set_facecolor('lightgrey')
plt.legend()
plt.savefig('AccFold.png')
plt.show()

# plt.subplot(222)
df[['loss','val_loss']].plot()
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.yscale('linear')
plt.xscale('linear')
plt.title("Xception Fold \n \n"
          "Training and Validation Loss")
plt.tight_layout()
plt.legend(loc='upper right')
plt.grid(True, color='w', linestyle='-', linewidth=2)
plt.gca().patch.set_facecolor('lightgrey')
plt.legend()
plt.savefig('LossFold.png')
plt.show()

def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(8,8), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    # ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.00)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

display_training_curves(df['accuracy'][1:], df['val_accuracy'][1:], 'accuracy', 211)
display_training_curves(df['loss'][1:], df['val_loss'][1:], 'loss', 212)

plt.show()