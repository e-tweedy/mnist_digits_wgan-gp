# mnist_digits_wgan-gp
Implementation of a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) [See https://arxiv.org/abs/1701.07875, https://arxiv.org/pdf/1704.00028.pdf] which is trained to mimick the MNIST handwritten digit database.  This implementation follows a tutorial from the book Machine Learning with PyTorch and Scikit-Learn by Raschka, Liu and Mirjalili (2022, Packt Publishing).

The following files are included:

A Jupyter notebook demonstrating the implementation using PyTorch: WGAN-GP.ipynb

Saved generator and discriminator models:

mnist_WGAN-GP_gen_model.pth,
mnist_WGAN-GP_gen_model_weights.pth,
mnist_WGAN-GP_disc_model.pth,
mnist_WGAN-GP_disc_model_weights.pth

The generator network has the following architecture:

Sequential(

  (0): ConvTranspose2d(100, 128, kernel_size=(4, 4), stride=(1, 1), bias=False)
  
  (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  
  (2): LeakyReLU(negative_slope=0.2)
  
  (3): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  
  (4): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  
  (5): LeakyReLU(negative_slope=0.2)
  
  (6): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  
  (7): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  
  (8): LeakyReLU(negative_slope=0.2)
  
  (9): ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  
  (10): Tanh()
  
)

The discriminator or "critic" network has the following architecture:

Sequential(

    (0): Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    
    (1): LeakyReLU(negative_slope=0.2)
    
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    
    (3): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    
    (4): LeakyReLU(negative_slope=0.2)
    
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    (6): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    
    (7): LeakyReLU(negative_slope=0.2)
    
    (8): Conv2d(128, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    
    (9): Sigmoid()
    
)

