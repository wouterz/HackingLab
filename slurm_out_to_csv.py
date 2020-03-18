

import re
import csv
import sys
import os

if __name__ == "__main__":

    try:
        gpu_re = re.compile(r'.*Created TensorFlow device.*name: (.*), pci.*')
        epoch_re = re.compile(r'Epoch (\d*): LearningRateScheduler setting learning rate to (\d*.\d*(?:e-\d*)?).')
        batch_re = re.compile(r'.*loss: (\d*.\d*) - _accuracy: (\d*.\d*).*')
        filename = sys.argv[1]
        if not os.path.isfile(filename):
            print('file %s not found' % filename)
            exit(1)

        short = True

        if short:
            shortFlag = True

        with open(filename+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

            current_epoch = 0
            current_lr = 0
            with open(filename) as f:
                for line in f:
                    
                    gpu_match = re.match(gpu_re, line)
                    if gpu_match:
                        gpu_name = gpu_match.group(1)
                        spamwriter.writerow([gpu_name])
                        spamwriter.writerow(['Epoch', 'LR', 'Loss', 'Accuracy'])
                        continue

                    epoch_match = re.match(epoch_re, line)
                    if epoch_match:
                        current_epoch = epoch_match.group(1)
                        current_lr = epoch_match.group(2)
                        # spamwriter.writerow([nr_epoch, learn_rate])
                        continue

                    batch_match = re.match(batch_re, line)
                    if batch_match and current_epoch:
                        loss = batch_match.group(1)
                        accuracy = batch_match.group(2)
                        spamwriter.writerow([current_epoch, current_lr, loss, accuracy])
                        if shortFlag:
                            current_epoch = 0
                        
    except Exception as e:
        print("error", e)