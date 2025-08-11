import os
import cv2
from tqdm import tqdm
import albumentations as A

# Input directories for original images and labels
image_dir = 'dataset/images'
label_dir = 'dataset/labels'

# Output directories for augmented images and labels
out_image_dir = 'augmented/images'
out_label_dir = 'augmented/labels'
os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

# Define augmentation pipeline using Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),                              # Horizontal flip
    A.RandomBrightnessContrast(p=0.5),                    # Random brightness & contrast adjustment
    A.Rotate(limit=10, p=0.5),                            # Random rotation within ±10 degrees
    A.GaussianBlur(p=0.2),                                # Gaussian blur
    A.RandomGamma(p=0.3),                                 # Random gamma correction
    A.HueSaturationValue(p=0.3),                          # Change hue, saturation, and value
    A.CLAHE(p=0.2),                                       # Adaptive histogram equalization
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),  # Random color shift
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),         # Random occlusion (dropout)
    A.Perspective(scale=(0.02, 0.05), p=0.2),             # Perspective transformation
    A.ISONoise(p=0.3),                                    # Simulate ISO noise
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# Loop through all images in the input folder
for filename in tqdm(os.listdir(image_dir)):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + '.txt')

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        continue
    height, width = image.shape[:2]

    # Read corresponding label (.txt)
    if not os.path.exists(label_path):
        continue
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x, y, w, h = map(float, parts)
        bboxes.append([x, y, w, h])
        class_labels.append(int(cls_id))  # Ensure class ID is an integer

    # Generate 10 augmented versions per image
    for i in range(400):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Save the augmented image
        out_img_name = f"{filename.rsplit('.', 1)[0]}_aug{i}.jpg"
        out_img_path = os.path.join(out_image_dir, out_img_name)
        cv2.imwrite(out_img_path, aug_img)

        # Save the corresponding label (.txt)
        out_label_path = os.path.join(out_label_dir, out_img_name.replace('.jpg', '.txt'))
        with open(out_label_path, 'w') as f:
            for cls_id, bbox in zip(aug_labels, aug_bboxes):
                bbox_str = ' '.join(f'{v:.6f}' for v in bbox)
                f.write(f"{int(cls_id)} {bbox_str}\n")  # Ensure class ID is saved as an integer
