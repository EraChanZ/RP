import numpy as np
import matplotlib.pyplot as plt
import re

# Initialize data storage
epochs = []
validation_epochs = []
validation_pck = []
validation_iou = []
source_pck, target_pck = [], []
source_iou, target_iou = [], []
joint_loss, coral_hand_loss = [], []
coral_body_feat = []

# Parse log file
with open('loss_log_pck535.txt', 'r') as f:
    current_epoch = -1
    joint_sum = coral_hand_sum = step_count = 0
    
    for line in f:
        # Handle training loss lines (e.g., "0	0	0.8223	0.7312	0.0910")
        if re.match(r'^\d+\t\d+\t[\d\.]+\t[\d\.]+\t[\d\.]+', line):
            cols = line.strip().split('\t')
            epoch = int(cols[0])
            
            # When we encounter a new epoch, store averages
            if epoch != current_epoch:
                if current_epoch != -1:  # Skip initial state
                    joint_loss.append(joint_sum / step_count)
                    coral_hand_loss.append(coral_hand_sum / step_count)
                current_epoch = epoch
                joint_sum = coral_hand_sum = step_count = 0
                
            # Accumulate losses (columns 3 and 4 are joint and coral_hand losses)
            joint_sum += float(cols[2])  # 3rd column is joint loss
            coral_hand_sum += float(cols[3])  # 4th column is coral hand loss
            step_count += 1

        # Handle validation lines
        elif 'VALIDATION_SOURCE' in line:
            source_pck.append(float(re.search(r'PCK:.*Mean: ([\d.]+)', line).group(1)))
            source_iou.append(float(re.search(r'IOU:.*Mean: ([\d.]+)', line).group(1)))
        elif 'VALIDATION_TARGET' in line:
            target_pck.append(float(re.search(r'PCK:.*Mean: ([\d.]+)', line).group(1)))
            target_iou.append(float(re.search(r'IOU:.*Mean: ([\d.]+)', line).group(1)))
        elif 'VALIDATION_CORAL' in line:
            coral_body_feat.append(float(re.search(r'coral_hand_feat:.*Mean: ([\d.]+)', line).group(1)))
        elif 'VALIDATION_EPOCH' in line:
            # Extract epoch number, PCK and IOU values
            epoch = int(line.split('_')[2].split('\t')[0])
            pck = float(re.search(r'PCK:\s+Mean:\s+([\d.]+)', line).group(1))
            iou = float(re.search(r'IOU:\s+Mean:\s+([\d.]+)', line).group(1))
            validation_epochs.append(epoch)
            validation_pck.append(pck)
            validation_iou.append(iou)

# Final epoch averages
if step_count > 0:
    joint_loss.append(joint_sum / step_count)
    coral_hand_loss.append(coral_hand_sum / step_count)

print(validation_pck)
# Create figure with 2 subplots side by side
plt.figure(figsize=(12, 6))

# Plot both PCK and IOU on the same graph
plt.plot(validation_epochs, validation_pck, 'blue', marker='o', label='Validation PCK')
plt.plot(validation_epochs, validation_iou, 'red', marker='s', label='Validation IOU')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend(prop={'size': 12})
plt.grid(True, alpha=0.3)
plt.title('Validation PCK and IOU per Epoch')

plt.tight_layout()
plt.savefig('handcoral_pck535_validation.png', dpi=300, bbox_inches='tight')
plt.close()

