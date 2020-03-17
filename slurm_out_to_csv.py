

import re
import csv
import sys
import os

if __name__ == "__main__":

    try:
        gpu_re = re.compile(r'.*Created TensorFlow device.*name: (.*), pci.*')
        epoch_re = re.compile(r'Epoch (\d*): LearningRateScheduler setting learning rate to (\d*.\d*).')
        batch_re = re.compile(r'.*loss: (\d*.\d*) - _accuracy: (\d*.\d*).*')
        filename = sys.argv[1]
        if not os.path.isfile(filename):
            print('file %s not found' % filename)
            exit(1)

        with open(filename+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

            
            with open(filename) as f:
                for line in f:
                    
                    gpu_match = re.match(gpu_re, line)
                    if gpu_match:
                        gpu_name = gpu_match.group(1)
                        spamwriter.writerow([gpu_name])
                        continue

                    epoch_match = re.match(epoch_re, line)
                    if epoch_match:
                        nr_epoch = epoch_match.group(1)
                        learn_rate = epoch_match.group(2)
                        spamwriter.writerow([nr_epoch, learn_rate])
                        continue

                    batch_match = re.match(batch_re, line)
                    if batch_match:
                        loss = batch_match.group(1)
                        accuracy = batch_match.group(2)
                        spamwriter.writerow([loss, accuracy])
                        
    except Exception as e:
        print("error", e)