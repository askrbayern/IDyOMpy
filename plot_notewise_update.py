#!/usr/bin/env python3
import argparse
import os
import pickle
import matplotlib.pyplot as plt

def load_results(path):
    if os.path.isdir(path):
        # find first .pickle file
        for fname in os.listdir(path):
            if fname.endswith('.pickle'):
                return pickle.load(open(os.path.join(path, fname), 'rb'))
        raise FileNotFoundError(f"No .pickle file found in directory {path}")
    else:
        return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot note-wise update results')
    parser.add_argument('input', help='Path to plot_notewise_update .pickle file or its containing folder')
    parser.add_argument('--viewpoint', choices=None, help='Specific viewpoint to plot (e.g. pitch, length)')
    args = parser.parse_args()

    data = load_results(args.input)
    # data: { song_name: { viewpoint: [ {idx, IC, Entropy, KL}, ... ], ... }, ... }
    for song, views in data.items():
        for vp, notes in views.items():
            if args.viewpoint and args.viewpoint != vp:
                continue
            idx = [n['idx'] for n in notes]
            IC = [n['IC'] for n in notes]
            Ent = [n['Entropy'] for n in notes]
            KL = [n['KL'] for n in notes]
            plt.figure(figsize=(10, 5))
            plt.plot(idx, IC, label='IC')
            plt.plot(idx, Ent, label='Entropy')
            plt.plot(idx, KL, label='KL Divergence')
            plt.title(f'{song} - {vp}')
            plt.xlabel('Note index')
            plt.ylabel('Value')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show() 