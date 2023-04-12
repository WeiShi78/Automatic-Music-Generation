import numpy as np
import musicpy as mp

midi_to_note_dict = {
    # update all values to add 30 to the key
    21: 'A0',
    33: 'A1',
    45: 'A2',
    57: 'A3',
    69: 'A4',
    81: 'A5',
    93: 'A6',
    105: 'A7',
    22: 'A#0',
    34: 'A#1',
    46: 'A#2',
    58: 'A#3',
    70: 'A#4',
    82: 'A#5',
    94: 'A#6',
    106: 'A#7',
    23: 'B0',
    35: 'B1',
    47: 'B2',
    59: 'B3',
    71: 'B4',
    83: 'B5',
    95: 'B6',
    107: 'B7',
    24: 'C1',
    36: 'C2',
    48: 'C3',
    60: 'C4',
    72: 'C5',
    84: 'C6',
    96: 'C7',
    108: 'C8',
    25: 'C#1',
    37: 'C#2',
    49: 'C#3',
    61: 'C#4',
    73: 'C#5',
    85: 'C#6',
    97: 'C#7',
    26: 'D1',
    38: 'D2',
    50: 'D3',
    62: 'D4',
    74: 'D5',
    86: 'D6',
    98: 'D7',
    27: 'D#1',
    39: 'D#2',
    51: 'D#3',
    63: 'D#4',
    75: 'D#5',
    87: 'D#6',
    99: 'D#7',
    28: 'E1',
    40: 'E2',
    52: 'E3',
    64: 'E4',
    76: 'E5',
    88: 'E6',
    100: 'E7',
    29: 'F1',
    41: 'F2',
    53: 'F3',
    65: 'F4',
    77: 'F5',
    89: 'F6',
    101: 'F7',
    30: 'F#1',
    42: 'F#2',
    54: 'F#3',
    66: 'F#4',
    78: 'F#5',
    90: 'F#6',
    102: 'F#7',
    31: 'G1',
    43: 'G2',
    55: 'G3',
    67: 'G4',
    79: 'G5',
    91: 'G6',
    103: 'G7',
    32: 'G#1',
    44: 'G#2',
    56: 'G#3',
    68: 'G#4',
    80: 'G#5',
    92: 'G#6',
    104: 'G#7',
}

def get_chord_label(notes_midi):
    # Assume the input is seperated by 16th note 
    # and each bar has 4 beats
    # Then every bar should have 16 notes

    '''
    Inputs:
    notes: int array with shape (N,1)  N: number of notes
    
    Outputs:
    chord_labels: string array with shape (N // 16)
    '''
    num = notes_midi.shape[-1]
    chord_labels = []
    # calculate the chord for each bar
    for i in range(num // 16):
        # get the notes in this bar
        # if notes is 1d array, then notes[i*16:(i+1)*16]
        # if notes is above 1d array, then notes[:, i*16:(i+1)*16]
        if len(notes_midi.shape) == 1:
            notes = notes_midi[i*16:(i+1)*16]
        else:
            notes = notes_midi[:, i*16:(i+1)*16]
        # get the unique notes
        unique_notes = np.unique(notes)
        # get the chord
        note_names = midi_to_note(unique_notes)
        cur_chord = mp.alg.detect(note_names, same_note_special=True)
        cur_chord = cur_chord.strip("[").replace("/", " ").replace("]", "").split(" ")
        if cur_chord[0] == 'note':
          chord_labels.append(cur_chord[1])
        else:
          chord_labels.append(cur_chord[0])
    
    return np.array(chord_labels)

def midi_to_note(notes_midi):
    # Transfer midi number to note name
    '''
    Input:
    notes_midi

    Output:
    notes_name
    '''
    notes_name = []
    for note in notes_midi:
        cur = midi_to_note_dict.get(note+30)
        notes_name.append(cur)
    
    return notes_name