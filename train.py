import argparse
import torch
from dataset import ToTensor, CSPDataset
from torch.utils.data import DataLoader, Dataset
from model import Generator, Discriminator, train_batch
import json
# create the parser
parser = argparse.ArgumentParser(description='Train the CSP GAN')

# Dataset
parser.add_argument('--size',       type=int,   default=10000,  help='Number of assignments of the dataset')
parser.add_argument('--k',          type=int,   default=2,    help='Arity of constraints')
parser.add_argument('--n',          type=int,   default=20,     help='Number of variables for the generated CSP')
parser.add_argument('--alpha',      type=float, default=0.4,    help=' alpha for the RB-model')
parser.add_argument('--r',          type=float, default=1.4,    help=' r for the RB-model')
parser.add_argument('--p',          type=float, default=0.5,    help=' p for the RB-model')

# seed
parser.add_argument('--seed',               type=int, default=30,    help=' Seed for the generation process')

parser.add_argument('--batch_size',          type=int, default=32,    help='Dimension of the batch')
parser.add_argument('--num_epochs',             type=int, default=5,    help='Number of epochs')

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
    n = dataset.csp.n
    m = dataset.csp.m
    n_bad = dataset.csp.n_bad_assgn
    csp_shape = {'k':args.k, 'n':args.n, 'd':dataset.csp.d, 'm':m, 'n_bad':n_bad}
    gen_input_size = 128
    hidden_units = 256
    layers_num = 4

    # define the architecture
    print("###############")
    print("Initializing the model")
    generator = Generator(gen_input_size, hidden_units, layers_num,  csp_shape)
    generator.to(device)

    discriminator = Discriminator(dataset.csp.n, hidden_units, layers_num, 2)
    discriminator.to(device)

    # loss
    adversarial_loss = torch.nn.BCELoss()

    # optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), weight_decay=5e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), weight_decay=5e-4)
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}


    print("###############")
    print("Start training")
    for epoch in range(args.num_epochs):
        print('##################################')
        print('## EPOCH %d' % (epoch + 1))
        print('##################################')
        # Iterate batches
        for batch_sample in dataloader:

            # moving to device
            batch = batch_sample.to(device)

            # Update network
            batch_loss_gen, batch_loss_discr = train_batch(generator, discriminator, batch, \
                csp_shape, adversarial_loss, optimizer)
            print('\t Training loss (single batch):', batch_loss_gen, batch_loss_discr)

    #Save all needed parameters
    print("###############")
    print("Saving parameters")
    # Create output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save network parameters
    torch.save(generator.state_dict(), out_dir / 'gen_params.pth')
    torch.save(generator.state_dict(), out_dir / 'discr_params.pth')

    # Save training parameters
    with open(out_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Save generated CSP
    with open(out_dir / 'csp.json', 'w') as f:
        json.dump(dataset, f, indent=4)


