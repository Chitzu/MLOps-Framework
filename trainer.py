import torch
import numpy as np
from networks.cnn import Net
from utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 lr_scheduler, logs_writer, config):
        self.config = config
        self.network = network
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logs_writer = logs_writer

        self.best_metric = 0.0

    def train_epoch(self, epoch):
        running_loss = []
        self.network = self.network.to(self.config['device'])

        for idx, (inputs, labels) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device'])
            labels = labels.to(self.config['device']).type(torch.LongTensor)
            
            self.optimizer.zero_grad()
            predictions = self.network(inputs)
            predictions = predictions.to(self.config['device'])
            labels = labels.to(self.config['device'])
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')

                self.logs_writer.add_scalar('Training Loss', running_loss, epoch * len(self.train_dataloader) + idx)
                running_loss = []

    def eval_net(self, epoch):
        stats_labels = []
        stats_predictions = []
        running_eval_loss = 0.0
        self.network.eval()

        total = 0
        correct = 0
        self.network = self.network.to(self.config['device'])

        for idx, (inputs, labels) in enumerate(self.eval_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).float()

            with torch.no_grad():
                predictions = self.network(inputs)
                predictions = torch.argmax(predictions, -1).float()

                total += labels.size(0)  
                correct += (predictions.to(self.config['device']) == labels.to(self.config['device'])).sum().item()
                                                                        
            eval_loss = self.criterion(predictions.to(self.config['device']), labels.to(self.config['device']))

            running_eval_loss += eval_loss.item()

            stats_predictions.append(predictions.detach().cpu().numpy())
            stats_labels.append(labels.detach().cpu().numpy())

        performance = correct / total
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        print(f'\nEvaluation loss on epoch {epoch} = {running_eval_loss}, Performance = {performance}\n')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'\nEvaluation loss on epoch {epoch} = {running_eval_loss}, Performance = {performance}\n')

        if self.best_metric < performance:
            self.best_metric = performance
            self.save_net_state(None, best=True)

        self.logs_writer.add_scalar('Validation Loss', running_eval_loss, (epoch + 1) * len(self.train_dataloader))

    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'],
                                                 self.config['exp_name'],
                                                 'latest_checkpoint.pth'),
                                                map_location=self.config['device'])
            
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i) + '\n')
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pth')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)

        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pth')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)

        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pth')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)

    def test_net(self, test_dataloader):
        running_test_loss = 0.0
        predictions_stats = []
        labels_stats = []

        checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'best_model.pth'),
                                map_location=self.config['device'])

        network = Net().to(self.config['device'])
        network.load_state_dict(checkpoint['model_weights'])
        network.eval()

        for idx, (inputs, labels) in enumerate(test_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).float()
            
            with torch.no_grad():
                predictions = self.network(inputs)
                predictions = torch.argmax(predictions, -1).float()
                
            loss = self.criterion(predictions.to(self.config['device']), labels.to(self.config['device']))

            predictions_stats.append(predictions.detach().cpu().numpy())
            labels_stats.append(labels.detach().cpu().numpy())

            running_test_loss += loss.item()

        running_test_loss = running_test_loss / len(test_dataloader)

        print(f'Test loss = {running_test_loss}\n')

        history = open(os.path.join(self.config['exp_path'], self.config['exp_name'], '__testStats__.txt'), "a")
        history.write(f'Test loss = {running_test_loss} ')
        history.close()



