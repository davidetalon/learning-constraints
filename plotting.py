import matplotlib.pyplot as plt
import json

filename='models/metrics15-07-2019,13-03-49.json'
if filename:
    with open(filename, 'r') as f:
        metrics = json.load(f)

 # Plot training & validation accuracy values
plt.plot(metrics['gen_loss'], label='G')
plt.plot(metrics['discr_loss'], label="D")
plt.title('Model accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['gen', 'discr'], loc='upper left')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['gen_top'], label='G top')
plt.title('Model accuracy')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['gen_bottom'], label='G top')
plt.title('Model accuracy')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['discr_top'], label='G top')
plt.title('Model accuracy')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['discr_bottom'], label='G top')
plt.title('Model accuracy')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')