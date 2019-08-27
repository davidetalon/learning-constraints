import matplotlib.pyplot as plt
import numpy as np
import json

filename='models/metrics23-08-2019,07-28-36.json'
if filename:
    with open(filename, 'r') as f:
        metrics = json.load(f)

 # Plot training & validation accuracy values
plt.plot(metrics['gen_loss'], label='G')
plt.plot(metrics['discr_loss'], label="D")
plt.title('Losses')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['gen', 'discr'], loc='upper left')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['gen_top'], label='G top')
plt.title('Gradient of Generator\'s top layer')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['gen_bottom'], label='G top')
plt.title('Gradient of Generator\'s bottom layer')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['discr_top'], label='G top')
plt.title('Gradient of Discriminator\'s top layer')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

 # Plot training & validation accuracy values
plt.plot(metrics['discr_bottom'], label='G top')
plt.title('Gradient of Discriminator\'s bottom layer')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.show()
# plt.savefig('plots/accuracy.png', format='png')

discr = np.array(metrics['discr_bottom'])
discr = discr / np.max(discr)

gen = np.array(metrics['gen_bottom'])
gen = gen /np.max(gen)

plt.plot(discr, label='discr top')
plt.plot(gen, label='gen top')
plt.title('Bottom layers grads')
plt.ylabel('Gradient')
plt.xlabel('Epoch')
plt.legend(['gen', 'discr'], loc='upper left')
plt.show()
