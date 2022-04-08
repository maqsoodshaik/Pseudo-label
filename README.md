Implementation of "Dong-Hyun Lee et al. “Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks”. In: Workshop on challenges in representation learning, ICML. Vol. 3. 2. 2013, p. 896."


To test the provided models in the model folder for CIFAR-10 250 and 4000 labels and for CIFAR-100 2500 and 10000 labels, please provide the argument for main.py of appropriate model file name and the second argument of flag to set in test mode.
Along with the third argument of which dataset to be choosen("cifar10"or"cifar100")
(eg main.py --bestmodel "./model/model_best.pth.tar" --test 1 --dataset "cifar10")

Models saved:

model_threshold_0.95_10000_cifar100.pth.tar
model_threshold_0.95_4000_cifar10.pth.tar
model_threshold_0.95_2500_cifar100.pth.tar
model_threshold_0.95_250_cifar10.pth.tar
model_threshold_0.75_10000_cifar100.pth.tar
model_threshold_0.75_4000_cifar10.pth.tar
model_threshold_0.75_2500_cifar100.pth.tar
model_threshold_0.75_250_cifar10.pth.tar
model_threshold_0.6_10000_cifar100.pth.tar
model_threshold_0.6_250_cifar10.pth.tar
model_threshold_0.6_4000_cifar10.pth.tar
model_threshold_0.6_2500_cifar100.pth.tar
