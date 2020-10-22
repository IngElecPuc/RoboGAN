import torch
from losses import generator_loss, discriminator_loss
from metrics import ADE, FDE
from matplotlib import pyplot as plt
import time
import datetime
import json

def trimm(batch, sequence_length, past_length, future_length, stride, trim_mode='absolute'):
    #This fuction will cut the entire sequence in chunks for the prediction
    #The Robot can only remember past_length steps of history for a prediction of size equal to future_length
    #The stride will serve to not overfitt the network to the training trajectories
    trimmed = {
        'noise' : [],
        'imgs' : [], 
        'past_traj' : [],
        'future_traj' : [],
        'target' : []
    }

    num_cuts = int((sequence_length - past_length - future_length) / (past_length - stride))
    jump = past_length - stride

    for i in range(num_cuts): #cuting chunks for prediction in a limited window
        #historical cuts
        trimmed['noise'].append(batch['noise'].narrow(1, i * jump, past_length))
        trimmed['imgs'].append(batch['imgs'].narrow(1, i * jump, past_length))
        trimmed['past_traj'].append(batch['trajectory'].narrow(1, i * jump, past_length))
        #future cuts
        trimmed['future_traj'].append(batch['trajectory'].narrow(1, i * jump + past_length, future_length))
        trimmed['target'].append(batch['target'].narrow(1, i * jump + past_length, future_length))

    if trim_mode == 'relative':
        #Translating each chunk to current robot's position
        for (past, futu) in zip(trimmed['past_traj'], trimmed['future_traj']):
            traj = past.permute(2, 1, 0).clone()
            xn = traj[0][-1] #last x coordinate of all the batch
            yn = traj[1][-1]
            traj[0] = traj[0] - xn #Moving to the origin
            traj[1] = traj[1] - yn 
            trimmed['past_traj'][i] = traj.permute(2, 1, 0)
            traj = futu.permute(2, 1, 0).clone()
            x0 = traj[0][0] #first x coordinate of all the batch
            y0 = traj[1][0]
            traj[0] = traj[0] - x0 #Moving to the origin
            traj[1] = traj[1] - y0 
            trimmed['future_traj'][i] = traj.permute(2, 1, 0)
            #We dont do this to the target because it it supposed to be calculated from current position from the simulator 
    
    return trimmed


def gan_epoch(gen, dis, loader, gen_opti, dis_opti, params, device, train_model=True):

    if (train_model):
        gen.train()
        dis.train()
    else:
        gen.eval()
        dis.eval()
        torch.no_grad()

    gen_mean_loss = 0
    dis_mean_loss = 0
    correct = 0

    for batch in loader: #Adjust window of the seq to this method
       
        trimmed = trimm(batch, 
                        params['seq_len'], 
                        params['history'], 
                        params['predict_seq'], 
                        int(params['history']/2), 
                        trim_mode='relative')

        steps = len(trimmed)

        for i in range(steps): 
            imgs = trimmed['imgs'][i].to(device)
            past_routes = trimmed['past_traj'][i].to(device)
            real_routes = trimmed['future_traj'][i].to(device)
            z = trimmed['noise'][i].to(device)
            objective = trimmed['target'][i].to(device)

            if imgs.shape[0] == 1: #Bug at some times
                continue

            gen.zero_grad()
            dis.zero_grad()
            fake_routes = gen(imgs, z, past_routes, objective)
            real_output = dis(imgs, real_routes, past_routes, objective)
            fake_output = dis(imgs, fake_routes, past_routes, objective)

            dis_loss = discriminator_loss(real_output, fake_output)
            dis_opti.zero_grad()
            if (train_model):
                dis_loss.backward(retain_graph=True)
                dis_opti.step()
            dis_mean_loss += dis_loss.item()/steps
            correct += real_output.eq(torch.ones_like(real_output)).sum().item()
            correct += fake_output.eq(torch.ones_like(fake_output)).sum().item()

            fake_output = dis(imgs, fake_routes, past_routes, objective) #Check if this step is really necessary
            gen_loss = generator_loss(fake_output)
            gen_opti.zero_grad()
            if (train_model):
                gen_loss.backward(retain_graph=True)
                gen_opti.step()
            gen_mean_loss += gen_loss.item()/steps

    gen_mean_loss /= len(loader.dataset)
    dis_mean_loss /= len(loader.dataset)
    dis_accuracy = correct/len(loader.dataset)

    return gen_mean_loss, dis_mean_loss, dis_accuracy

def train_gan(nepochs, gen, dis, train_loader, valid_loader, gen_opti, dis_opti, params, device, name):

    training_log = {}
    training_log['gen_t_loss'] = []
    training_log['dis_t_loss'] = []
    training_log['dis_t_acc'] = []
    training_log['gen_v_loss'] = []
    training_log['dis_v_loss'] = []
    training_log['dis_v_acc'] = []

    for epoch in range(nepochs):
    
        start = time.time()

        gen_t_loss, dis_t_loss, dis_t_acc = gan_epoch(gen, dis, train_loader, gen_opti, dis_opti, params, device, train_model=True)
        gen_v_loss, dis_v_loss, dis_v_acc = gan_epoch(gen, dis, valid_loader, gen_opti, dis_opti, params, device, train_model=False)
        training_log['gen_t_loss'].append(gen_t_loss)
        training_log['dis_t_loss'].append(dis_t_loss)
        training_log['dis_t_acc'].append(dis_t_acc)
        training_log['gen_v_loss'].append(gen_v_loss)
        training_log['dis_v_loss'].append(dis_v_loss)
        training_log['dis_v_acc'].append(dis_v_acc)

        msj = 'Epoch {:03d}: time {:.3f} sec, gen_t_loss {:.3f}, dis_t_loss {:.3f}, dis_t_acc {:.3f}, gen_v_loss {:.3f}, dis_v_loss {:.3f}, dis_v_acc {:.5f}'
        #print(msj.format(epoch+1, time.time()-start, gen_t_loss, dis_t_loss, dis_v_acc, gen_v_loss, dis_v_loss, dis_v_acc))
        
        with open('train_progress.txt', 'w') as json_file:
            json.dump(msj.format(epoch+1, 
                    datetime.timedelta(seconds=int(time.time()-start)),
                    gen_t_loss, 
                    dis_t_loss, 
                    dis_v_acc, 
                    gen_v_loss, 
                    dis_v_loss, 
                    dis_v_acc),
                    json_file)

        torch.save(gen.state_dict(), './gen.pth')
        torch.save(dis.state_dict(), './dis.pth')

    fig = plt.figure(figsize=(12.8, 14.4))
    plt.subplot(3, 3, 1)
    plt.plot(list(range(nepochs)), training_log['gen_t_loss'])
    plt.title('Generator training loss')
    plt.subplot(3, 3, 2)
    plt.plot(list(range(nepochs)), training_log['dis_t_loss'])
    plt.title('Discriminator training loss')
    plt.subplot(3, 3, 3)
    plt.plot(list(range(nepochs)), training_log['gen_t_loss'], list(range(nepochs)), training_log['dis_t_loss'])
    plt.title('Both of them')
    plt.subplot(3, 3, 4)
    plt.plot(list(range(nepochs)), training_log['gen_v_loss'])
    plt.title('Generator validation loss')
    plt.subplot(3, 3, 5)
    plt.plot(list(range(nepochs)), training_log['dis_v_loss'])
    plt.title('Discriminator validation loss')
    plt.subplot(3, 3, 6)
    plt.plot(list(range(nepochs)), training_log['gen_v_loss'], list(range(nepochs)), training_log['dis_v_loss'])
    plt.title('Both of them')
    plt.subplot(3, 3, 7)
    plt.plot(list(range(nepochs)), training_log['dis_t_acc'])
    plt.title('Discriminator training accuracy')
    plt.subplot(3, 3, 8)
    plt.plot(list(range(nepochs)), training_log['dis_v_acc'])
    plt.title('Discriminator validation accuracy')
    plt.subplot(3, 3, 9)
    plt.plot(list(range(nepochs)), training_log['dis_t_acc'], list(range(nepochs)), training_log['dis_v_acc'])
    plt.title('Both of them')
    #plt.show() #Save img instead
    plt.savefig('train_statistics_' + name + '.png')
    return training_log

def test_gan(gen, dis, test_loader, gen_opti, dis_opti, params, device):

    start = time.time()
    gen_loss, dis_loss, dis_acc = gan_epoch(gen, dis, test_loader, gen_opti, dis_opti, params, device, train_model=False)
    msj = 'Time {:.3f} sec, gen_loss {:.3f}, dis_loss {:.3f}, dis_acc {:.3f}'
    print(msj.format(datetime.timedelta(seconds=int(time.time()-start)), gen_loss, dis_loss, dis_acc))
    return gen_loss, dis_loss, dis_acc