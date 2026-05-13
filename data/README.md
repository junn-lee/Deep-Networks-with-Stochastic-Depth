# Data

This project uses the **CIFAR-10** dataset.

CIFAR-10 is automatically downloaded by torchvision when you run any training script:

```python
torchvision.datasets.CIFAR10(root='./data', download=True)
```

Or from within the Colab notebooks, it is downloaded to `/content/data` at the start of each training run.

**Dataset details:**
- 60,000 32×32 color images across 10 classes
- 50,000 training images, 10,000 test images
- 45,000 / 5,000 train / validation split used in all experiments
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Official source:** https://www.cs.toronto.edu/~kriz/cifar.html
