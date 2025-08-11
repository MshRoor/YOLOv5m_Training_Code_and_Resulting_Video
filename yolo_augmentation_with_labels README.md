# YOLO Image Augmentation with Label Preservation

This script performs offline data augmentation for object detection tasks using the YOLO format. It applies various visual transformations to each image and simultaneously transforms corresponding bounding boxes.

---

## Folder Structure

Make sure your dataset is organized like this:

```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
├── labels/
│   ├── img1.txt
│   ├── img2.txt
```

Each `.txt` file should contain YOLO-formatted annotations:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized between `0` and `1`.

---

## Output

The augmented images and labels will be saved in:

```
augmented/
├── images/
├── labels/
```

Each original image will produce 10 augmented variants like:

- `img1_aug0.jpg`
- `img1_aug0.txt`
- ...
- `img1_aug9.jpg`

---

## Applied Augmentations

This script uses [Albumentations](https://github.com/albumentations-team/albumentations) and includes the following:

- Horizontal and vertical flips
- Brightness and contrast adjustment
- Rotation
- Gaussian blur
- Gamma correction
- Hue/saturation shifts
- CLAHE (adaptive histogram equalization)
- RGB color shifts
- Coarse dropout (occlusion)
- Perspective distortion
- ISO noise simulation

---

## Requirements

Install all dependencies:

```bash
pip install albumentations opencv-python tqdm
```

---

## tqdm Installation

`tqdm` is used to show progress bars during image processing. Install it with:

- **Standard Python:**

```bash
pip install tqdm
```

- **Jupyter or Colab:**

```python
!pip install tqdm
```

- **Anaconda:**

```bash
conda install -c conda-forge tqdm
```

To verify installation:

```python
from tqdm import tqdm
for i in tqdm(range(100)):
    pass
```

---

## How to Run

```bash
python yolo_augmentation_with_labels.py
```

---

## Notes

- Make sure each image has a matching `.txt` file with the same name.
- Output files will overwrite if names already exist.
- Supports `.jpg`, `.png`, and `.jpeg` images.

---

## License

MIT License
