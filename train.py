import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from functools import partial
from coconet import Net, save_heatmaps, harmonize_melody_and_save_midi, RNNNet
from hparams import I, T, P, MIN_MIDI_PITCH, MAX_MIDI_PITCH
from chords import get_chord_label
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./id2chord.txt') as f:
  id2chord = f.readlines()
id2chord = [c.strip() for c in id2chord]
chord2id = {}
for i, c in enumerate(id2chord):
  chord2id[c] = i

batch_size = 32
n_layers = 64
chord_emb_dim = 64
chord_emb_num = 656
chord_emb_num_pertrack = 8
hidden_size = 128
n_train_steps = 30000
N = batch_size

data = np.load('Jsb16thSeparated.npz', encoding='bytes', allow_pickle=True)

# data augmentation
train_tracks = []
for y in data['train']:
  for i in range(-3, 4):
    train_tracks.append(y + i)
len_train = len(train_tracks)

valid_tracks = []
for y in data['valid']:
  for i in range(-3, 4):
    valid_tracks.append(y + i)
len_valid = len(valid_tracks)

all_tracks = train_tracks + valid_tracks
# construct training data

train_tracks = []
for track in all_tracks[:len_train]:
  track = track.transpose()
  cut = 0
  while cut < track.shape[1] - T:
    if (track[:, cut:cut + T] > 0).all():
      train_tracks.append(track[:, cut:cut + T] - MIN_MIDI_PITCH)
    cut += T

valid_tracks = []
for track in all_tracks[len_train:len_train + len_valid]:
  track = track.transpose()
  cut = 0
  while cut < track.shape[1] - T:
    if (track[:, cut:cut + T] > 0).all():
      valid_tracks.append(track[:, cut:cut + T] - MIN_MIDI_PITCH)
    cut += T

train_tracks = np.array(train_tracks).astype(int)
valid_tracks = np.array(valid_tracks).astype(int)

# get test sample
# test_sample = data['test'][0].transpose()[:, :T]
# test_sample_melody = test_sample[0]

def collate_fn(batch):
    # Convert batch to PyTorch tensor
    batch = np.array(batch)
    batch = torch.tensor(batch).to(device)
    # Get the chord IDs for each part in the batch
    batch_chord_id = []
    for part in batch:
        c = get_chord_label(part.cpu().numpy())
        temp = [chord2id[cc] for cc in c]
        batch_chord_id.append(temp)
    batch_chord_id = torch.tensor(batch_chord_id).to(device)

    # Reshape the batch for input to the model
    batch = batch.view(N * I * T)
    x = torch.zeros((N * I * T, P)).to(device)
    r = torch.arange(N * I * T).to(device)
    x[r, batch] = 1
    x = x.view(N, I, T, P)

    # Randomly create the C tensor for each batch
    C = torch.randint(2, size=(N, I, T)).to(device)
    C2 = C.type(torch.FloatTensor).to(device)

    # Get the targets
    targets = batch.reshape(-1)

    return (x, C2, batch_chord_id, targets)

# Set up the DataLoader with multiprocessing
train_loader = DataLoader(train_tracks, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=4)
valid_loader = DataLoader(valid_tracks, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=4)

# Create a SummaryWriter to log training and validation metrics
log_dir = 'runs'
writer = SummaryWriter(log_dir)
if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  model = Net(n_layers, hidden_size, chord_emb_num, chord_emb_dim, chord_emb_num_pertrack).to(device)
#   model = RNNNet(n_layers, hidden_size, chord_emb_num, chord_emb_dim, chord_emb_num_pertrack).to(device)
  
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
  losses = []
  model.train()

  best_valid_loss = float('inf')
  best_model_state_dict = None
  n_epoches = 50
  for epoch in range(n_epoches):
    for i, (x, C2, batch_chord_id, targets) in enumerate(train_loader):
      out = model(x, C2, batch_chord_id)
      
      out = out.view(N * I * T, P)
      loss = loss_fn(out, targets)
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      print('Epoch {} Step {}: train_loss = {}'.format(epoch+1, i, loss.item()))
      writer.add_scalar('Loss/train_step', loss.item(), i)
      
      # adjust learning rate
      if i % 30 == 0 and i > 0:
        for g in optimizer.param_groups:
          g['lr'] *= .75
      
      # save the model every save_every steps
    torch.save(model.state_dict(), 'CNN_{}.pt'.format(epoch))
    # torch.save(model.state_dict(), 'RNN_{}.pt'.format(epoch))
    writer.add_scalar('Loss/train_epoch', loss.item(), epoch)
    
    #valid
    valid_loss = 0
    model.eval()
    with torch.no_grad():
      for x, C2, batch_chord_id, targets in valid_loader:
        x, C2, batch_chord_id, targets = x.to(device), C2.to(device), batch_chord_id.to(device), targets.to(device)
        out = model(x, C2, batch_chord_id)
        out = out.view(N * I * T, P)
        valid_loss += loss_fn(out, targets).item() * targets.shape[0]
      valid_loss /= len(valid_loader.dataset) # average the loss across validation set
    print('Epoch {}: Valid loss = {:.4f}'.format(epoch+1, valid_loss))
    writer.add_scalar('Loss/valid', valid_loss, epoch) # Log the validation loss

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      best_model_state_dict = model.state_dict()
    model.train() # set the model back to training mode

    # if i % 500 == 0:
    #   now_date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #   print(f'{now_date_time} | step: {i} | loss: {loss.item()}')
    #   D0 = np.ones((1, T))
    #   D1 = np.zeros((3, T))
    #   D = np.concatenate([D0, D1], axis=0).astype(int)
    #   y = np.random.randint(P, size=(I, T))
    #   y[0, :] = np.array(test_sample_melody - MIN_MIDI_PITCH)
    #   chord = get_chord_label(y[0, :])
    #   chord_id = [chord2id[cho] for cho in chord]
    #   save_heatmaps(model, y, D, chord_id, i, device)
    #   if i % 5000 == 0:
    #     harmonize_melody_and_save_midi(test_sample_melody, i, chord2id, model, device)
    #   model.train()

  torch.save(best_model_state_dict, 'CNN.pt')
#   torch.save(best_model_state_dict, 'RNN.pt')