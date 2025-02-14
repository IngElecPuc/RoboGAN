import torch
from losses import generator_loss, discriminator_loss
from metrics import ADE, FDE
from matplotlib import pyplot as plt
import time
import datetime
import json
import numpy as np

def trimm(batch, sequence_length, past_length, future_length, stride, trim_mode='absolute'):
    #This fuction will cut the entire sequence in chunks for the prediction
    #The Robot can only remember past_length steps of history for a prediction of size equal to future_length
    #The stride will serve to not overfitt the network to the training trajectories
    trimmed = {
        'noise' : [],
        'imgs' : [], 
        'past_traj' : [],
        'future_traj' : [],
        'past_vel' : [],
        'future_vel' : [],
        'past_target' : [],
        'future_target' : [],
        'trim_mode' : trim_mode,
        'batch_size' : 0,
        'steps' : 0
    }

    chunks = int((sequence_length - past_length - future_length) / (past_length - stride)) + 1
    jump = past_length - stride
    tensor_shape = batch['noise'].shape
    trimmed['batch_size'] = tensor_shape[0]
    trimmed['steps'] = chunks

    for i in range(chunks): #cuting chunks for prediction in a limited window
        #historical cuts
        trimmed['noise'].append(batch['noise'].narrow(1, i * jump, past_length).clone())
        trimmed['imgs'].append(batch['imgs'].narrow(1, i * jump, past_length).clone())
        trimmed['past_traj'].append(batch['trajectory'].narrow(1, i * jump, past_length).clone())
        trimmed['past_vel'].append(batch['velocity'].narrow(1, i * jump, past_length).clone())
        trimmed['past_target'].append(batch['target'].narrow(1, i * jump, past_length).clone())
        #future cuts
        trimmed['future_traj'].append(batch['trajectory'].narrow(1, i * jump + past_length, future_length).clone())
        trimmed['future_vel'].append(batch['velocity'].narrow(1, i * jump + past_length, future_length).clone())
        trimmed['future_target'].append(batch['target'].narrow(1, i * jump + past_length, future_length).clone())

    if trim_mode == 'relative':
        #Translating each chunk to current robot's position
        for (i, (past, futu)) in enumerate(zip(trimmed['past_traj'], trimmed['future_traj'])):
            past = past.permute(2, 1, 0)
            xn = past[0][-1] #last x coordinate of all the batch
            yn = past[1][-1]
            past[0] = past[0] - xn #Moving to the origin
            past[1] = past[1] - yn 
            trimmed['past_traj'][i] = past.permute(2, 1, 0)
            futu = futu.permute(2, 1, 0)
            x0 = futu[0][0] #first x coordinate of all the batch
            y0 = futu[1][0]
            futu[0] = futu[0] - x0 #Moving to the origin
            futu[1] = futu[1] - y0 
            trimmed['future_traj'][i] = futu.permute(2, 1, 0)
            #We dont do this to the target because it it supposed to be calculated from current position from the simulator 

        for (i, (past, futu)) in enumerate(zip(trimmed['past_target'], trimmed['future_target'])):
            past = past.permute(2, 1, 0)
            xn = past[0][-1] #last x coordinate of all the batch
            yn = past[1][-1]
            past[0] = past[0] - xn #Moving to the origin
            past[1] = past[1] - yn 
            trimmed['past_target'][i] = past.permute(2, 1, 0)
            futu = futu.permute(2, 1, 0)
            x0 = futu[0][0] #first x coordinate of all the batch
            y0 = futu[1][0]
            futu[0] = futu[0] - x0 #Moving to the origin
            futu[1] = futu[1] - y0 
            trimmed['future_target'][i] = futu.permute(2, 1, 0)

    return trimmed

def reconstruct(trimmed, sequence_length, past_length, stride):
    #This function will try to reconstruct a trimmed sequence of trajectories
    trajectory = torch.zeros((1, 1, 1))

    for (i, chunk) in enumerate(trimmed['past_traj']):
        chunk = chunk.permute(2, 1, 0)
        if trimmed['trim_mode'] == 'relative':
            if i == 0:
                x0 = chunk[0][0]
                y0 = chunk[1][0]
            else: #Corregir aquí según corresponda
                x0 += chunk[0][0]
                y0 += chunk[1][0]
            chunk[0] = chunk[0] - x0
            chunk[1] = chunk[1] - y0   
        #chunk = chunk.narrow(1, 0, past_length - stride)
        chunk = chunk.narrow(1, 0, past_length - 1)

        if i == 0:
            trajectory = chunk.clone()
        else:
            trajectory = torch.cat((trajectory, chunk.clone()),  axis=1)

    chunk = trimmed['future_traj'][-1].permute(2, 1, 0)
    trajectory = torch.cat((trajectory, chunk.clone()), axis=1)            
    
    return trajectory.permute(2, 1, 0)
        
def gan_epoch(gen, dis, loader, gen_opti, dis_opti, params, device, train_model=True):

    if (train_model):
        gen.train()
        dis.train()
    else:
        gen.eval()
        dis.eval()
        torch.no_grad()

    gen_mean_loss = []
    dis_mean_loss = []
    ADE_mean = []
    FDE_mean = []

    for (num_batch, batch) in enumerate(loader): #Adjust window of the seq to this method
       
        trimmed = trimm(batch, 
                        params['seq_len'], 
                        params['history'], 
                        params['predict_seq'], 
                        int(params['history']/2), 
                        trim_mode='relative')

        for i in range(trimmed['steps']): 
            imgs = trimmed['imgs'][i].to(device)
            z = trimmed['noise'][i].to(device)
            past_routes = trimmed['past_traj'][i].to(device)
            real_routes = trimmed['future_traj'][i].to(device)
            past_vel = trimmed['past_vel'][i].to(device)
            real_vel = trimmed['future_vel'][i].to(device)
            past_obj = trimmed['past_target'][i].to(device)
            real_obj = trimmed['future_target'][i].to(device)
            #print(f'Shape of past_routes {past_routes.shape}')
            #print(f'Shape of past_vel {past_vel.shape}')
            #print(f'Shape of past_obj {past_vel.shape}')
            past_routes = torch.cat((past_routes, past_vel), axis=2)
            past_routes = torch.cat((past_routes, past_obj), axis=2)
            real_routes = torch.cat((real_routes, real_vel), axis=2)
            real_routes = torch.cat((real_routes, real_obj), axis=2)
            #print(f'Shape of concatenated {past_routes.shape}')

            if imgs.shape[0] == 1: #Bug at some times
                continue

            gen.zero_grad()
            dis.zero_grad()
            fake_routes = gen(imgs, z, past_routes)
            real_output = dis(imgs, real_routes, past_routes)
            fake_output = dis(imgs, fake_routes, past_routes)
            
            ADE_mean.append(ADE(real_routes, fake_routes) / trimmed['batch_size'])
            FDE_mean.append(FDE(real_routes, fake_routes) / trimmed['batch_size'])

            dis_loss = discriminator_loss(real_output, fake_output, params)
            dis_opti.zero_grad()
            if (train_model):
                dis_loss.backward(retain_graph=True)
                dis_opti.step()
            dis_mean_loss.append(dis_loss.item() / trimmed['batch_size'])

            fake_output = dis(imgs, fake_routes, past_routes) #Check if this step is really necessary
            gen_loss = generator_loss(fake_output, fake_routes, real_routes, params)
            gen_opti.zero_grad()
            if (train_model):
                gen_loss.backward(retain_graph=True)
                gen_opti.step()
            gen_mean_loss.append(gen_loss.item() / trimmed['batch_size'])
        
        print('{:.2f} percent of current epoch'.format((num_batch+1)*loader.batch_size/len(loader.dataset)*100))

    return gen_mean_loss, dis_mean_loss, ADE_mean, FDE_mean

def train_gan(nepochs, gen, dis, train_loader, valid_loader, gen_opti, dis_opti, params, device, name):

    training_log = {}
    training_log['gen_t_loss'] = []
    training_log['dis_t_loss'] = []
    training_log['ADE_t'] = []
    training_log['FDE_t'] = []
    training_log['gen_v_loss'] = []
    training_log['dis_v_loss'] = []
    training_log['ADE_v'] = []
    training_log['FDE_v'] = []
    print('Starting training')

    for epoch in range(nepochs):
    
        start = time.time()

        gen_t_loss, dis_t_loss, ADE_t, FDE_t = gan_epoch(gen, dis, train_loader, gen_opti, dis_opti, params, device, train_model=True)
        gen_v_loss, dis_v_loss, ADE_v, FDE_v = gan_epoch(gen, dis, valid_loader, gen_opti, dis_opti, params, device, train_model=False)
        training_log['gen_t_loss'] += gen_t_loss
        training_log['dis_t_loss'] += dis_t_loss
        training_log['ADE_t'] += ADE_t
        training_log['FDE_t'] += FDE_t
        training_log['gen_v_loss'] += gen_v_loss
        training_log['dis_v_loss'] += dis_v_loss
        training_log['ADE_v'] += ADE_v
        training_log['FDE_v'] += FDE_v

        msg = f'Epoch {epoch+1}: '
        msg += f'time {datetime.timedelta(seconds=int(time.time()-start))} sec, '
        msg += f'gen_t_loss {mean(gen_t_loss):.4f}, '
        msg += f'dis_t_loss {mean(dis_t_loss):.4f}, '
        msg += f'ADE_t {mean(ADE_t):.4f}, '
        msg += f'FDE_t {mean(FDE_t):.4f}, '
        msg += f'gen_v_loss {mean(gen_v_loss):.4f}, '
        msg += f'dis_v_loss {mean(dis_v_loss):.4f}, '
        msg += f'ADE_v {mean(ADE_v):.4f}, '
        msg += f'FDE_v {mean(FDE_v):.4f}\n'
        print(msg)

        with open('train_progress' + name + '.txt', 'w') as json_file:
            json.dump(msg, json_file)

        torch.save(gen.state_dict(), './gen_' + name + '.pth')
        torch.save(dis.state_dict(), './dis_' + name + '.pth')

    plt.figure(figsize=(12.8, 19.2))
    plt.subplot(4, 3, 1)
    plt.plot(list(range(len(training_log['gen_t_loss']))), training_log['gen_t_loss'])
    plt.title('Generator training loss')
    plt.subplot(4, 3, 2)
    plt.plot(list(range(len(training_log['dis_t_loss']))), training_log['dis_t_loss'])
    plt.title('Discriminator training loss')
    plt.subplot(4, 3, 3)
    plt.plot(list(range(len(training_log['gen_t_loss']))), training_log['gen_t_loss'], list(range(len(training_log['dis_t_loss']))), training_log['dis_t_loss'])
    plt.title('Both of them')
    plt.subplot(4, 3, 4)
    plt.plot(list(range(len(training_log['gen_v_loss']))), training_log['gen_v_loss'])
    plt.title('Generator validation loss')
    plt.subplot(4, 3, 5)
    plt.plot(list(range(len(training_log['dis_v_loss']))), training_log['dis_v_loss'])
    plt.title('Discriminator validation loss')
    plt.subplot(4, 3, 6)
    plt.plot(list(range(len(training_log['gen_v_loss']))), training_log['gen_v_loss'], list(range(len(training_log['dis_v_loss']))), training_log['dis_v_loss'])
    plt.title('Both of them')
    plt.subplot(4, 3, 7)
    plt.plot(list(range(len(training_log['ADE_t']))), training_log['ADE_t'])
    plt.title('Average displacement error in training')
    plt.subplot(4, 3, 8)
    plt.plot(list(range(len(training_log['ADE_v']))), training_log['ADE_v'])
    plt.title('Average displacement error in validation')
    plt.subplot(4, 3, 9)
    plt.plot(list(range(len(training_log['ADE_t']))), training_log['ADE_t'], list(range(len(training_log['ADE_v']))), training_log['ADE_v'])
    plt.title('Both of them')
    plt.subplot(4, 3, 10)
    plt.plot(list(range(len(training_log['FDE_t']))), training_log['FDE_t'])
    plt.title('Final displacement error in training')
    plt.subplot(4, 3, 11)
    plt.plot(list(range(len(training_log['FDE_v']))), training_log['FDE_v'])
    plt.title('Final displacement error in validation')
    plt.subplot(4, 3, 12)
    plt.plot(list(range(len(training_log['FDE_t']))), training_log['FDE_t'], list(range(len(training_log['FDE_v']))), training_log['FDE_v'])
    plt.title('Both of them')
    #plt.show() #Save img instead
    plt.savefig('train_statistics_' + name + '.png')
    return training_log

def test_gan(gen, dis, test_loader, gen_opti, dis_opti, params, device):

    start = time.time()
    gen_loss, dis_loss, ADE_m, FDE_m = gan_epoch(gen, dis, test_loader, gen_opti, dis_opti, params, device, train_model=False)
    print('Starting testing')
    msg = 'Time {} sec, gen_loss {:.3f}, dis_loss {:.3f}, ADE {:.3f}, FDE {:.3f}'
    print(msg.format(datetime.timedelta(seconds=int(time.time()-start)), mean(gen_loss), mean(dis_loss), mean(ADE_m), mean(FDE_m)))
    return gen_loss, dis_loss, ADE_m, FDE_m

def mean(a):
    n = 0
    for e in a:
        n += e
    return n/len(a)