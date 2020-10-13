from torch import ones_like, zeros_like
import torch.nn.functional as F

def discriminator_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, 0.7*ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, 0.3*zeros_like(fake_output))  
    return real_loss + fake_loss

def generator_loss(fake_output):
    return F.binary_cross_entropy(fake_output, 0.7*ones_like(fake_output))

def target_loss():
    return True

def velocity_regulator_loss():
    return True