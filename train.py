import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from coconet import Net, save_heatmaps, harmonize_melody_and_save_midi, RNNNet
from hparams import I, T, P, MIN_MIDI_PITCH, MAX_MIDI_PITCH
from chords import get_chord_label
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load training data
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

all_tracks = train_tracks + valid_tracks
train_tracks = np.array(train_tracks).astype(int)
valid_tracks = np.array(valid_tracks).astype(int)
all_tracks = np.array(all_tracks).astype(int)

# chord_set = set()
# for track in all_tracks:
#   fp = track[0, :]
#   chord = get_chord_label(fp)
#   for c in chord:
#     chord_set.add(c)
# chord_list = list(chord_set)
# with open('./id2chord.txt', 'w') as f:
#   for c in chord_list:
#     f.write(c + '\n')
# f.close()

# get test sample
test_sample = data['test'][0].transpose()[:, :T]
test_sample_melody = test_sample[0]

if __name__ == '__main__':
  batch_size = 2
  n_layers = 64
  chord_emb_dim = 64
  chord_emb_num = 614
  chord_emb_num_pertrack = 8
  hidden_size = 128
  n_train_steps = 80000
  save_every = n_train_steps // 10
  show_every = max(1, n_train_steps // 1000)
  softmax = F.softmax
  valid_loader = DataLoader(train_tracks, batch_size=batch_size, shuffle=False, drop_last=False)

  with open('./id2chord.txt') as f:
    id2chord = f.readlines()
  id2chord = [c.strip() for c in id2chord]
  chord2id = {}
  for i, c in enumerate(id2chord):
    chord2id[c] = i

  model = Net(n_layers, hidden_size, chord_emb_num, chord_emb_dim, chord_emb_num_pertrack).to(device)
#   model = RNNNet(n_layers, hidden_size, chord_emb_num, chord_emb_dim, chord_emb_num_pertrack).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
  losses = []

  model.train()
  N = batch_size

  for i in range(n_train_steps):

    # tensor of shape (N, I, T)
    C = np.random.randint(2, size=(N, I, T))

    # batch is an np array of shape (N, I, T), entries are integers in [0, P)
    indices = np.random.choice(train_tracks.shape[0], size=N)
    batch = train_tracks[indices]

    fpart = batch[:, 0, :]
    batch_chord_id = []
    for part in fpart:
      c = get_chord_label(part)
      temp = [chord2id[cc] for cc in c]
      batch_chord_id.append(temp)
    batch_chord_id = torch.tensor(batch_chord_id).to(device)

    # targets is of shape (N*I*T)
    targets = batch.reshape(-1)
    targets = torch.tensor(targets).to(device)

    # x is of shape (N, I, T, P)

    batch = batch.reshape(N * I * T)
    x = np.zeros((N * I * T, P))
    r = np.arange(N * I * T)
    x[r, batch] = 1
    x = x.reshape(N, I, T, P)
    x = torch.tensor(x).type(torch.FloatTensor).to(device)

    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    out = model(x, C2, batch_chord_id)
    out = out.view(N * I * T, P)
    loss = loss_fn(out, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print('{} step: train_loss = {}'.format(i, loss.item()))

    #valid
    # if i % 1 == 0:
    #   valid_losses = []
    #   model.eval()
    #   for batch in valid_loader:
    #     batch = batch.numpy()
    #     C = np.random.randint(2, size=(N, I, T))
    #     fpart = batch[:, 0, :]
    #     batch_chord_id = []
    #     for part in fpart:
    #       c = get_chord_label(part)
    #       temp = [chord2id[cc] for cc in c]
    #       batch_chord_id.append(temp)
    #     batch_chord_id = torch.tensor(batch_chord_id).to(device)

    #     # targets is of shape (N*I*T)
    #     targets = batch.reshape(-1)
    #     targets = torch.tensor(targets).to(device)

    #     # x is of shape (N, I, T, P)

    #     batch = batch.reshape(N * I * T)
    #     x = np.zeros((N * I * T, P))
    #     r = np.arange(N * I * T)
    #     x[r, batch] = 1
    #     x = x.reshape(N, I, T, P)
    #     x = torch.tensor(x).type(torch.FloatTensor).to(device)
    #     C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    #     out = model(x, C2, batch_chord_id)
    #     out = out.view(N * I * T, P)
    #     loss = loss_fn(out, targets)
    #     valid_losses.append(loss.item())
    #   model.train()
    #   print("="*80)
    #   print("valid_loss: {}".format(loss.item()))
    #   print("="*80)

    if i % 500 == 0:
      now_date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
      print(f'{now_date_time} | step: {i} | loss: {loss.item()}')
      D0 = np.ones((1, T))
      D1 = np.zeros((3, T))
      D = np.concatenate([D0, D1], axis=0).astype(int)
      y = np.random.randint(P, size=(I, T))
      y[0, :] = np.array(test_sample_melody - MIN_MIDI_PITCH)
      chord = get_chord_label(y[0, :])
      chord_id = [chord2id[cho] for cho in chord]
      save_heatmaps(model, y, D, chord_id, i, device)
      if i % 5000 == 0:
        harmonize_melody_and_save_midi(test_sample_melody, i, chord2id, model, device)
      model.train()

    # adjust learning rate
    if i % 5000 == 0 and i > 0:
      for g in optimizer.param_groups:
        g['lr'] *= .75

#   torch.save(model.state_dict(), 'RNN_pretrained.pt')
  torch.save(model.state_dict(), 'CNN_pretrained.pt')