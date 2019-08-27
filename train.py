import argparse
import torch
from torch import autograd
from sklearn.metrics import confusion_matrix
import numpy as np
from dataset import ToTensor, CSPDataset
from csp import matrix_assignment_consistency
from torch.utils.data import DataLoader, Dataset
from model import Generator, Discriminator, train_batch, weights_init, printgradnorm_forward, printgradnorm_backward
import json
from pathlib import Path
import time
import datetime
import gc
# create the parser
parser = argparse.ArgumentParser(description='Train the CSP GAN')

# Dataset
parser.add_argument('--size',       type=int,   default=10000,  help='Number of assignments of the dataset')
parser.add_argument('--k',          type=int,   default=2,    help='Arity of constraints')
parser.add_argument('--n',          type=int,   default=24,     help='Number of variables for the generated CSP')
parser.add_argument('--alpha',      type=float, default=1.0,    help=' alpha for the RB-model')
parser.add_argument('--r',          type=float, default=1.4,    help=' r for the RB-model')
parser.add_argument('--p',          type=float, default=0.5,    help=' p for the RB-model')

# seed
parser.add_argument('--seed',            type=int, default=30,    help=' Seed for the generation process')
parser.add_argument('--gen_lr',          type=float, default=0.0002,    help=' Generator\'s learning rate')
parser.add_argument('--discr_lr',        type=float, default=0.0001,    help=' Generator\'s learning rate')

parser.add_argument('--batch_size',          type=int, default=16,    help='Dimension of the batch')
parser.add_argument('--num_epochs',             type=int, default=5,    help='Number of epochs')

parser.add_argument('--save',             type=bool, default=False,    help='Save the generator and discriminator models')
parser.add_argument('--out_dir',          type=str, default='models/',    help='Folder where to save the model')

if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()
    
    #%% Check device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #%% Create dataset
    dataset = CSPDataset(size=args.size, k=args.k, n=args.n, alpha=args.alpha,\
         r=args.r, p=args.p, transform=ToTensor())

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # retrieving data
    # n = dataset.csp.n
    n = dataset.csp.n
    m = dataset.csp.m
    n_bad = dataset.csp.n_bad_assgn
    csp_shape = {'k':args.k, 'n':n, 'd':dataset.csp.d, 'm':m, 'n_bad':n_bad}
    gen_input_size = 128

    # define the architecture
    print("Initializing the model")
    generator = Generator(gen_input_size, csp_shape)
    # generator.to(device)

    discriminator = Discriminator(csp_shape['n'], 2)
    # discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # loss
    adversarial_loss = torch.nn.BCELoss()

    # optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.discr_lr)
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}

    # hooks to print gradients
    # discriminator.model.register_backward_hook(printgradnorm_backward)
    # discriminator.dense.register_backward_hook(printgradnorm_backward)
    # generator.conv.register_backward_hook(printgradnorm_backward)
    # generator.dense.register_backward_hook(printgradnorm_backward)



    print("Start training")
    gen_loss_history = []
    discr_loss_history = []
    D_x_history = []
    D_G_z1_history = []
    D_G_z2_history = []

    discr_top_grad =[]
    discr_bottom_grad=[]
    gen_top_grad = []
    gen_bottom_grad = []

    for epoch in range(args.num_epochs):

        # Iterate batches
        for i, batch_sample in enumerate(dataloader):

            # moving to device
            # batch = batch_sample.to(device)
            batch = batch_sample

            # Update network
            start = time.time()

            gen_loss, discr_loss, D_x, D_G_z1, D_G_z2, discr_top, discr_bottom, gen_top, gen_bottom = train_batch(generator, discriminator, batch, \
                csp_shape, adversarial_loss, optimizer)

            # train_batch(generator, discriminator, batch, csp_shape, adversarial_loss, optimizer)

            # print('***********TENSOR INFO***********')
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
            
            
            # discr_loss, D_x, D_G_z1 = train_batch(generator, discriminator, batch, \
            #     csp_shape, adversarial_loss, optimizer)

            # print(discr_top.shape)
            # print(torch.norm(gen_bottom))
            # saving metrics
            gen_loss_history.append(gen_loss)
            discr_loss_history.append(discr_loss)
            D_x_history.append(D_x)
            D_G_z1_history.append(D_G_z1)
            D_G_z2_history.append(D_G_z1)
            discr_top_grad.append(discr_top)
            discr_bottom_grad.append(discr_bottom)
            gen_top_grad.append(gen_top)
            gen_bottom_grad.append(gen_bottom)

            end = time.time()
            print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]"
            % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader), discr_loss, gen_loss, D_x, D_G_z1, D_G_z2))

    # count the number of wrong non-sat assignments
    
    #  Evaluation mode (e.g. disable dropout)
    print('Accuracy')
    generator.eval()
    with torch.no_grad(): # Disable gradient tracking

        # starting from random noise
        # mean = torch.zeros(1, 1, 128, 128)
        # variance = torch.ones(1, 1, 128, 128)
        # rnd_assgn = torch.normal(mean, variance)
        rnd_assgn = torch.randn((1, 1, 128, 128))

        csp, assgn = generator(rnd_assgn)
    
    print('csp', dataset.csp.matrix.shape)
    print('fake csp', csp.shape)
    errors = torch.sum((torch.from_numpy(dataset.csp.matrix).type(torch.float) - csp).pow(2)).item()

    csp = np.squeeze(csp.numpy())
    csp[csp > 0.5] = 1
    csp[csp <= 0.5] = 0 

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # print (csp.shape, dataset.csp.matrix.shape)
    # for row in range(csp.shape[0]):
    #     for col in range(csp.shape[0]):
    #         if csp[row, col]==dataset.csp.matrix[row, col] ==1:
    #             TP += 1
    #         if csp[row, col]!=dataset.csp.matrix[row, col] and csp[row, col]==1:
    #             FP +=1
    #         if csp[row, col]==dataset.csp.matrix[row, col] ==0:
    #             TN +=1
    #         if csp[row, col]!=dataset.csp.matrix[row, col] and csp[row, col]==0:
    #             FN +=1

    flatten_gen_data = dataset.csp.matrix.flatten()
    flatten_csp = csp.flatten()
    print(flatten_csp.shape, type(flatten_csp))
    cm = confusion_matrix(flatten_gen_data, flatten_csp)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN=cm[1][1]

    matrix_size = csp_shape['n']*csp_shape['d']
    accuracy = ((matrix_size**2) - errors)/matrix_size**2
    print(accuracy)
    print("TP %d, FP %d, TN %d, FN %d"% (TP, FP, TN, FN))


    # check if given samples are correctly recognized
    satisfaiability = matrix_assignment_consistency(torch.from_numpy(dataset.assignments).type(torch.int64), torch.from_numpy(csp), csp_shape['d'], csp_shape['n'])
    satisfaiability = satisfaiability.numpy()

    satisfaiability = np.squeeze(satisfaiability)
    assignment_sat = dataset.assignments[:,-1]
    assignment_sat = np.squeeze(assignment_sat)

    correctly_classified = (np.array(dataset.assignments[:,-1].astype(int)) == satisfaiability).sum()

    print('correctly classified: ', correctly_classified)

    #Save all needed parameters
    print("Saving parameters")
    # Create output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save network parameters
    date = datetime.datetime.now()
    date = date.strftime("%d-%m-%Y,%H-%M-%S")
    if args.save:
        gen_file_name = 'gen_params'+date+'.pth'
        discr_file_name = 'discr_params'+date+'.pth'
        torch.save(generator.state_dict(), out_dir / gen_file_name)
        torch.save(discriminator.state_dict(), out_dir / discr_file_name)


    # Save training parameters
    params_file_name = 'training_args'+date+'.json'
    with open(out_dir / params_file_name, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Save generated CSP
    csp_file_name = 'csp' + date + '.json'
    with open(out_dir / csp_file_name, 'w') as f:
        json.dump(dataset.csp.matrix.tolist(), f, indent=4)
    
    metrics = {'correctly classified': int(correctly_classified), 'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN), 'n_sat_assignments':dataset.n_sat_assignments, 'acc':accuracy, 'gen_loss':gen_loss_history, \
        'discr_loss':discr_loss_history, 'D_x':D_x_history, 'D_G_z1':D_G_z1_history, 'D_G_z2':D_G_z2_history, \
        'gen_top':gen_top_grad, 'gen_bottom':gen_bottom_grad, 'discr_top':discr_top_grad, 'discr_bottom':discr_bottom_grad}
    
    # Save metrics
    metric_file_name = 'metrics'+ date +'.json'
    with open(out_dir / metric_file_name, 'w') as f:
        json.dump(metrics, f, indent=4)
