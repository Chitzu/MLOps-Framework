import json
import os
import torch
import torch.optim as optim
from data.data_manager import DataManager
from trainer import Trainer
from utils.data_logs import save_logs_about
from torch.utils.tensorboard import SummaryWriter
from networks.cnn import Net


def main():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config['device'])

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    logs_writer = SummaryWriter(os.path.join('runs', config['exp_name']))

    model = Net()
    model = model.to(config['device'])

    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'])

    data_manager = DataManager(config)
    train_loader, eval_dataloader , test_loader = data_manager.get_train_eval_test_dataloaders()

    trainer = Trainer(model, train_loader, eval_dataloader, criterion, optimizer, lr_scheduler, logs_writer, config)

    trainer.train()

    trainer.test_net(test_loader)


if __name__ == '__main__':
    main()
