from torch import ones_like, zeros_like
import torch.nn.functional as F

def discriminator_loss(real_output, fake_output, params):
    real_loss = F.binary_cross_entropy(real_output, params['up_criterion'] * ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, params['down_criterion'] * zeros_like(fake_output))  
    return real_loss + fake_loss

def generator_loss(fake_output, fake_routes, real_routes, params):
    gen_adver = F.binary_cross_entropy(fake_output, params['up_criterion'] * ones_like(fake_output))
    gen_final = target_loss(fake_routes, real_routes)
    gen_veloc = velocity_regulator_loss(fake_routes, real_routes, params)
    return gen_adver * (1 - params['alpha'] - params['beta']) + gen_final * params['alpha'] + gen_veloc * params['beta']

def target_loss(fake_routes, real_routes):
    fr = fake_routes.permute(1, 0, 2).clone()
    last_fake = fr[-1]
    
    rr = real_routes.permute(1, 0, 2).clone()
    last_real = rr[-1]

    return F.mse_loss(last_fake, last_real)

def velocity_regulator_loss(fake_routes, real_routes, params):
    fr = fake_routes.permute(1, 0, 2).clone()
    fake_veloc = fr[1:params['predict_seq']] - fr[0:(params['predict_seq'] - 1)]
    fake_veloc = fake_veloc.permute(1, 0, 2)
    
    rr = real_routes.permute(1, 0, 2).clone()
    real_veloc = rr[1:params['predict_seq']] - rr[0:(params['predict_seq'] - 1)]
    real_veloc = real_veloc.permute(1, 0, 2)
    
    return F.mse_loss(fake_veloc, real_veloc)