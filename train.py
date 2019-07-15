import argparse
import torch
from dataset import ToTensor, CSPDataset
from torch.utils.data import DataLoader, Dataset
from model import Generator, Discriminator, train_batch, weights_init
import json
from pathlib import Path
import time
import datetime
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    generator.to(device)

    discriminator = Discriminator(csp_shape['n'], 2)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # loss
    adversarial_loss = torch.nn.BCELoss()

    # optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discr_lr, betas=(0.5, 0.999))
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}


    print("Start training")
    gen_loss_history = []
    discr_loss_history = []
    D_x_history = []
    D_G_z1_history = []
    D_G_z2_history = []
    for epoch in range(args.num_epochs):

        # Iterate batches
        for i, batch_sample in enumerate(dataloader):

            # moving to device
            # batch = batch_sample.to(device)
            batch = batch_sample

            # Update network
            start = time.time()
            gen_loss, discr_loss, D_x, D_G_z1, D_G_z2= train_batch(generator, discriminator, batch, \
                csp_shape, adversarial_loss, optimizer)

            # saving metrics
            gen_loss_history.append(gen_loss.item())
            discr_loss_history.append(discr_loss.item())
            D_x_history.append(D_x)
            D_G_z1_history.append(D_G_z1)
            D_G_z2_history.append(D_G_z1)

            end = time.time()
            print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]"
            % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader), discr_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

    # count the number of wrong non-sat assignments
    
    #  Evaluation mode (e.g. disable dropout)
    print('Accuracy')
    generator.eval()
    with torch.no_grad(): # Disable gradient tracking

        # starting from random noise
        mean = torch.zeros(1, 1, 128, 128)
        variance = torch.ones(1, 1, 128, 128)
        rnd_assgn = torch.normal(mean, variance)

        csp, assgn = generator(rnd_assgn)
    
    errors = torch.sum((torch.from_numpy(dataset.csp.matrix).type(torch.float) - csp).pow(2)).item()

    matrix_size = csp_shape['n']*csp_shape['d']
    accuracy = ((matrix_size**2) - errors)/matrix_size**2
    print(accuracy)
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
    
    metrics = {'n_sat_assignments':dataset.n_sat_assignments, 'acc':accuracy, 'gen_loss':gen_loss_history, 'discr_loss':discr_loss_history, 'D_x':D_x_history, 'D_G_z1':D_G_z1_history, 'D_G_z2':D_G_z2_history}
    
    # Save metrics
    metric_file_name = 'metrics'+ date +'.json'
    with open(out_dir / metric_file_name, 'w') as f:
        json.dump(metrics, f, indent=4)
