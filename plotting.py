import matplotlib.pyplot as plt
import json

filename='models/metrics14-07-2019,21-21-59.json'
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