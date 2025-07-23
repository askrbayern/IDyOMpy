#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_learning_events(log_file='out/learning_events.log'):
    """Analyze the learning events to validate incremental learning behavior"""
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found. Run the update function first.")
        return
    
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"Loaded {len(events)} learning events")
    
    # Group by reason
    by_reason = defaultdict(list)
    for event in events:
        by_reason[event['reason']].append(event)
    
    print(f"\nEvent breakdown:")
    for reason, evts in by_reason.items():
        print(f"  {reason}: {len(evts)} events")
    
    # Analyze KL > 0 events
    kl_events = by_reason['kl_nonzero']
    if kl_events:
        print(f"\n=== KL > 0 Events Analysis ===")
        kl_values = [e['kl_divergence'] for e in kl_events]
        print(f"KL divergence range: {min(kl_values):.6f} to {max(kl_values):.6f}")
        print(f"Mean KL: {np.mean(kl_values):.6f}")
        
        # Check distribution changes
        print(f"\nSample distribution changes:")
        for i, event in enumerate(kl_events[:5]):  # First 5 KL>0 events
            print(f"Event {event['event']} (order {event['order']}, note_idx {event['note_idx']}):")
            print(f"  Pre:  {event['pre_dist']}")
            print(f"  Post: {event['post_dist']}")
            print(f"  KL: {event['kl_divergence']:.6f}")
            print(f"  Alphabet size: {event['alphabet_size']}")
            print()
    
    # Analyze first training events  
    first_training = by_reason['first_training']
    if first_training:
        print(f"=== First Training Events ===")
        print(f"Orders that started training: {sorted(set(e['order'] for e in first_training))}")
        
        # When did each order start?
        order_starts = defaultdict(list)
        for event in first_training:
            order_starts[event['order']].append(event['note_idx'])
        
        print(f"Order start points:")
        for order in sorted(order_starts.keys()):
            start_points = sorted(order_starts[order])
            print(f"  Order {order}: notes {start_points}")
    
    # Check if learning is progressive
    print(f"\n=== Learning Progression ===")
    note_positions = [e['note_idx'] for e in events]
    sequence_lengths = [e['sequence_length'] for e in events]
    print(f"Note positions range: {min(note_positions)} to {max(note_positions)}")
    print(f"Sequence lengths range: {min(sequence_lengths)} to {max(sequence_lengths)}")
    
    # Plot learning progression
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if kl_events:
        kl_note_pos = [e['note_idx'] for e in kl_events]
        kl_values = [e['kl_divergence'] for e in kl_events]
        plt.scatter(kl_note_pos, kl_values, alpha=0.6)
        plt.xlabel('Note Position')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Over Time')
    
    plt.subplot(2, 2, 2)
    orders = [e['order'] for e in events]
    note_pos = [e['note_idx'] for e in events]
    plt.scatter(note_pos, orders, alpha=0.6)
    plt.xlabel('Note Position') 
    plt.ylabel('Model Order')
    plt.title('Which Orders Are Learning When')
    
    plt.subplot(2, 2, 3)
    alphabet_sizes = [e['alphabet_size'] for e in events]
    plt.scatter(note_pos, alphabet_sizes, alpha=0.6)
    plt.xlabel('Note Position')
    plt.ylabel('Alphabet Size')
    plt.title('Alphabet Growth')
    
    plt.subplot(2, 2, 4)
    if kl_events:
        kl_orders = [e['order'] for e in kl_events]
        kl_vals = [e['kl_divergence'] for e in kl_events]
        plt.scatter(kl_orders, kl_vals, alpha=0.6)
        plt.xlabel('Model Order')
        plt.ylabel('KL Divergence')
        plt.title('KL by Model Order')
    
    plt.tight_layout()
    plt.savefig('out/learning_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return events

if __name__ == "__main__":
    import os
    analyze_learning_events() 