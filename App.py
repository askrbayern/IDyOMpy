"""
Enter point of the program.
"""
from idyom import idyom
from idyom import data

from optparse import OptionParser
from shutil import copyfile
from shutil import rmtree
from glob import glob
from idyom.longTermModel import longTermModel
from tqdm import tqdm
import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import scipy.io as sio
import math
import random
import ast
import sys
import logging

# === DEBUG LOGGING CONFIGURATION START ===
# Creates a debug_logs folder and writes all DEBUG logs there
DEBUG_LOG_DIR = 'debug_logs'
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
DEBUG_LOG_FILE = os.path.join(DEBUG_LOG_DIR, 'kl_debug.log')
logging.basicConfig(
    level=logging.DEBUG,
    filename=DEBUG_LOG_FILE,
    filemode='w',
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
# === DEBUG LOGGING CONFIGURATION END ===

def _explain_kl_zero(pre_dist, post_dist, pre_is_deterministic, post_is_deterministic, distributions_identical):
    """Explain why KL divergence is zero"""
    if distributions_identical:
        if pre_is_deterministic and post_is_deterministic:
            return "Both distributions are deterministic (prob=1.0) and identical - no learning possible"
        elif pre_is_deterministic or post_is_deterministic:
            return "One distribution is deterministic, identical distributions"
        else:
            return "Distributions are numerically identical - possible precision issue"
    else:
        return "Distributions differ but KL=0 - unexpected case"

SERVER = False

if SERVER:
    plt.ioff()

def foo_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def vstr(L):
    ret = ""
    for elem in L:
        ret += str(elem) + "_"


    return ret[:-1]

def comparePitches(list1, list2, k=0.9):
    """
    Compare two list of pitches, with a criterion k
    """
    score = 0

    for i in range(min(len(list1), len(list2))):
        score += list1[i] == list2[i]

    if score > int(k*min(len(list1), len(list2))):
        return True
    else:
        return False

def checkDataSet(folder):
    """
    Function that check if the dataset is corrupted (contains duplicates).
    Does not delete automatically!
    """

    files = []
    for filename in glob(folder + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            files.append(filename)

    D = data.data(deleteDuplicates=False)
    D.addFiles(files)
    DATA = D.getData("pitch")

    delete = []
    delete_pitches = []

    for i in range(len(files)):
        for j in range(i, len(files)):
            if i != j and comparePitches(DATA[i], DATA[j]):

                print(files[i], "matches", files[j])


                # We recommand to delete the smallest one
                if len(DATA[i]) > len(DATA[j]):
                    for d in delete_pitches:
                        if comparePitches(d, DATA[i]):
                            delete.append(files[i])
                            delete_pitches.append(DATA[i])
                            break

                    delete.append(files[j])
                    delete_pitches.append(DATA[j])
                else:
                    for d in delete_pitches:
                        if comparePitches(d, DATA[j]):
                            delete.append(files[j])
                            delete_pitches.append(DATA[j])
                            break

                    delete.append(files[i])
                    delete_pitches.append(DATA[i])            

    if len(delete) > 0:
        print("We recommand you to delete the following files because they are duplicates:")
        print(list(set(delete)))
    else:
        print("We did not find any duplicates.")

def replaceinFile(file, tochange, out):
    s = open(file).read()
    s = s.replace(tochange, out)
    f = open(file, "w")
    f.write(s)
    f.close()

def cross_validation(folder, k_fold=10, maxOrder=20, quantization=24, time_representation=False, \
                                        zero_padding=True, long_term_only=False, short_term_only=False,\
                                        viewPoints=["pitch", "length"], genuine_entropies=False, use_original_PPM=False):
    """
    Cross-validate by training on on k-1 folds of the folder and evaluate on the remaining fold
    k_fold = -1 means leave-one-out 
    """
    np.random.seed(0)

    viewPoints_o = []
    for elem in viewPoints:
        if elem in data.AVAILABLE_VIEWPOINTS:
            viewPoints_o.append(elem)
        else:
            print("I don't know the viewpoint: " + str(elem))
            print("Please check and rerun the program.")
            quit()

    ICs = []
    Entropies = []

    files = []
    for filename in glob(folder + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            files.append(filename)

    np.random.shuffle(files)

    if int(k_fold) == -1:
        k_fold = len(files)

    if int(k_fold) > len(files):
        raise ValueError("Cannot process with k_fold greater than number of files. Please use -k options to specify a smaller k for cross validation.")

    k_fold = len(files) // int(k_fold)

    validationFiles = []
    Likelihoods = []

    for i in tqdm(range(math.ceil(len(files)/k_fold))):
        trainData = files[:i*k_fold] + files[(i+1)*k_fold:]
        evalData = files[i*k_fold:(i+1)*k_fold]

        L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, use_original_PPM=use_original_PPM)
        M = data.data(quantization=quantization)
        M.addFiles(trainData)

        L.train(M)

        for file in evalData:
            # ============= updated to handle KL return ===========
            IC, E, K = L.getSurprisefromFile(file, long_term_only=long_term_only, short_term_only=short_term_only, time_representation=time_representation, zero_padding=zero_padding, genuine_entropies=genuine_entropies)
            # ============= end of KL handling ====================
            ICs.append(IC)
            Entropies.append(E)
            filename = file[file.rfind("/")+1:file.rfind(".")]
            filename = filename.replace("-", "_")
            validationFiles.append(filename)
            # not collecting KL here!

    return ICs, Entropies, validationFiles


def Train(folder, quantization=24, maxOrder=20, time_representation=False, \
                zero_padding=True, long_term_only=False, short_term_only=False, \
                viewPoints=["pitch, length"], use_original_PPM=False):

    '''
    Train a model with the midi files contained in the passed folder.
    '''

    if folder[-1] == "/":
        folder = folder[:-1]

    viewPoints_o = []
    for elem in viewPoints:
        if elem in data.AVAILABLE_VIEWPOINTS:
            viewPoints_o.append(elem)
        else:
            print("I don't know the viewpoint: " + str(elem))
            print("Please check and rerun the program.")
            quit()

    ppm_name = "_originalPPM" if use_original_PPM else ""
    if os.path.isfile("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+vstr(viewPoints)+ str(ppm_name) + ".model"):
        print("There is already a model saved for these data, would you like to train again? (Y/N)\n")
        rep = input("")
        while rep not in ["y", "Y", "n", "N", "", "\n"]:
            rep = input("We did not understand, please type again (Y/N).")

        if rep.lower() == "y":
            pass
        else:
            return

    preComputeEntropies = not (long_term_only or short_term_only) # We only precompute if we need to combine short and long term models

    L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, use_original_PPM=use_original_PPM)
    M = data.data(quantization=quantization)
    M.parse(folder, augment=True)
    L.train(M)

    ppm_name = "_originalPPM" if use_original_PPM else ""
    L.save("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints)+ppm_name+ ".model")


def Train_by_piece_with_all_metrics(folder, nb_pieces=20, quantization=24, maxOrder=20, time_representation=False, \
                zero_padding=True, long_term_only=False, short_term_only=False, viewPoints=["pitch", "length"], \
                intialization="", use_original_PPM=False):
    """
    Modified version that saves IC, Entropy, and KL during evolution
    """
    
    name_temp_file = ".tmp_test_folder_" + folder[folder.rfind("/")+1:] + "_" + str(np.random.randint(100, 999))

    if folder[-1] == "/":
        folder = folder[:-1]

    viewPoints_o = []
    for elem in viewPoints:
        if elem in data.AVAILABLE_VIEWPOINTS:
            viewPoints_o.append(elem)
        else:
            print("I don't know the viewpoint: " + str(elem))
            print("Please check and rerun the program.")
            quit()

    L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, evolutive=True, use_original_PPM=use_original_PPM)

    files = glob(folder+'/**.mid', recursive=True) + glob(folder+'/**.midi', recursive=True)

    random.shuffle(files)
    train = files[:-nb_pieces]
    test = files[-nb_pieces:]

    if intialization != "":
        europe_files = files = glob(intialization, recursive=True)
        train = europe_files[:620] + train

    if os.path.exists(name_temp_file):
        if os.path.isdir(name_temp_file):
            rmtree(name_temp_file)
        else:
            os.remove(name_temp_file)
    os.mkdir(name_temp_file)
    
    for file in test: 
        copyfile(file, name_temp_file+file[file.rfind("/"):])

    try:
        note_counter = []
        # Create separate storage for each metric
        matrix_ic = np.zeros((len(train), nb_pieces))
        matrix_entropy = np.zeros((len(train), nb_pieces))
        matrix_kl = np.zeros((len(train), nb_pieces))
        
        # Store full sequences for last training step
        dico_ic = {}
        dico_entropy = {}
        dico_kl = {}
        
        print("___ Starting Training ___")
        k = 0
        for file in tqdm(train):
            try:
                M = data.data(quantization=quantization)
                M.parseFile(file)
                L.train(M, preComputeEntropies=False)

                # Get metrics (IC, Entropy, KL per-viewpoint)
                S, E, K_by_vp_all, files = L.getSurprisefromFolder(
                    name_temp_file,
                    time_representation=time_representation,
                    long_term_only=long_term_only,
                    short_term_only=short_term_only
                )
                
                note_counter.append(len(M.viewPointRepresentation["pitch"][0]))

                # Initialize per-viewpoint matrices on first iteration
                if k == 0:
                    matrix_kl_by_viewpoint = {
                        vp: np.zeros((len(train), nb_pieces)) for vp in viewPoints_o
                    }

                # Store all metrics
                for i in range(len(files)):
                    filename = files[i][files[i].rfind("/")+1:]
                    
                    # Store full sequences for last step
                    dico_ic[filename] = S[i]
                    dico_entropy[filename] = E[i]
                    # reconstruct total KL sequence by summing across viewpoints (for backward compat)
                    # and store per-viewpoint sequences separately
                    # total KL
                    if 'dico_kl' not in locals():
                        dico_kl = {}
                    if len(K_by_vp_all[i]) > 0:
                        # sum element-wise over viewpoints
                        K_total_seq = None
                        for vp_seq in K_by_vp_all[i].values():
                            arr = np.array(vp_seq)
                            K_total_seq = arr if K_total_seq is None else K_total_seq + arr
                        dico_kl[filename] = K_total_seq.tolist()
                    else:
                        dico_kl[filename] = []
                    # Also store per-viewpoint KL sequences for last step
                    # Use a nested dict: {piece: {vp: list}}
                    if 'dico_kl_by_viewpoint' not in locals():
                        dico_kl_by_viewpoint = {}
                    dico_kl_by_viewpoint[filename] = K_by_vp_all[i]
                    
                    # Store means in matrices
                    matrix_ic[k, i] = np.mean(S[i])
                    matrix_entropy[k, i] = np.mean(E[i])
                    # mean of total KL (from reconstructed K_total_seq)
                    matrix_kl[k, i] = np.mean(dico_kl[filename]) if len(dico_kl[filename]) > 0 else 0.0
                    # Per-viewpoint KL means
                    for vp in viewPoints_o:
                        if vp in K_by_vp_all[i]:
                            matrix_kl_by_viewpoint[vp][k, i] = np.mean(K_by_vp_all[i][vp])
                
                k += 1
            except (FileNotFoundError, RuntimeError, ValueError):
                print(file+ " skipped.")

        # Cumulative note counter
        for i in range(1, len(note_counter)):
            note_counter[i] += note_counter[i-1] 

        # Save all metrics
        saving = {
            'matrix_ic': matrix_ic,
            'matrix_entropy': matrix_entropy,
            'matrix_kl': matrix_kl,
            'note_counter': note_counter,
            'dico_ic': dico_ic,
            'dico_entropy': dico_entropy,
            'dico_kl': dico_kl,
            'dico_kl_by_viewpoint': dico_kl_by_viewpoint if 'dico_kl_by_viewpoint' in locals() else {},
            'matrix_kl_by_viewpoint': matrix_kl_by_viewpoint if 'matrix_kl_by_viewpoint' in locals() else {},
            'test_pieces': [f[f.rfind("/")+1:] for f in test]
        }

        if not os.path.exists("out/"+folder[folder.rfind("/"):]):
            os.makedirs("out/"+folder[folder.rfind("/"):])

        if not os.path.exists("out/"+folder[folder.rfind("/"):]+"/evolution/"):
            os.makedirs("out/"+folder[folder.rfind("/"):]+"/evolution/")

        output_path = "out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'_all_metrics.pickle'
        pickle.dump(saving, open(output_path, "wb"))
        print(f"Data saved at {output_path}")
        
        # Also save as .mat
        sio.savemat(output_path.replace('.pickle', '.mat'), saving)
        
        rmtree(name_temp_file)
        
    except Exception as e:
        rmtree(name_temp_file)
        raise e


def Train_by_piece(folder, nb_pieces=20, quantization=24, maxOrder=20, time_representation=False, \
                zero_padding=True, long_term_only=False, short_term_only=False, viewPoints=["pitch", "length"], \
                intialization="", use_original_PPM=False):

    '''
    Train iteratively on the passed folder (can be also initialized with western data)
    and generate the prediction error for each piece along the training. This allows
    to look at the dynamics of learning during training. 
    '''

    name_temp_file = ".tmp_test_folder_" + folder[folder.rfind("/")+1:] + "_" + str(np.random.randint(100, 999))

    if folder[-1] == "/":
        folder = folder[:-1]

    viewPoints_o = []
    for elem in viewPoints:
        if elem in data.AVAILABLE_VIEWPOINTS:
            viewPoints_o.append(elem)
        else:
            print("I don't know the viewpoint: " + str(elem))
            print("Please check and rerun the program.")
            quit()

    L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, evolutive=True, use_original_PPM=use_original_PPM)

    files = glob(folder+'/**.mid', recursive=True) + glob(folder+'/**.midi', recursive=True)

    random.shuffle(files)
    train = files[:-nb_pieces]
    test = files[-nb_pieces:]

    if intialization != "":
        europe_files = files = glob(intialization, recursive=True)
        train = europe_files[:200] + train

    if os.path.exists(name_temp_file):
        if os.path.isdir(name_temp_file):
            rmtree(name_temp_file)
        else:
            os.remove(name_temp_file)
    os.mkdir(name_temp_file)
    
    for file in test: 
        copyfile(file, name_temp_file+file[file.rfind("/"):])

    try:
        note_counter = []
        dicos = []
        matrix = np.zeros((len(train), nb_pieces))
        print("___ Starting Training ___")
        k = 0
        for file in tqdm(train):
            try:
                M = data.data(quantization=quantization)
                M.parseFile(file)
                L.train(M, preComputeEntropies=False)

                # ============= updated to handle KL return ===========
                S, E, K, files = L.getSurprisefromFolder(name_temp_file, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)
                # ============= end of KL handling ====================
                note_counter.append(len(M.viewPointRepresentation["pitch"][0]))

                dico = {}
                for i in range(len(files)):
                    dico[files[i][files[i].rfind("/")+1:]] = S[i]

                dicos.append(dico)
                tmp = []
                for s in S:
                    tmp.append(np.mean(s))

                matrix[k,:] = tmp
                k += 1
            except (FileNotFoundError, RuntimeError, ValueError):
                print(file+ " skipped.")

        for i in range(1, len(note_counter)):
            note_counter[i] += note_counter[i-1] 

        saving = {}
        saving['matrix'] = matrix
        saving['note_counter'] = note_counter
        saving['dico'] = dico

        if not os.path.exists("out/"+folder[folder.rfind("/"):]):
            os.makedirs("out/"+folder[folder.rfind("/"):])

        if not os.path.exists("out/"+folder[folder.rfind("/"):]+"/evolution/"):
            os.makedirs("out/"+folder[folder.rfind("/"):]+"/evolution/")

        pickle.dump(saving, open("out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.pickle', "wb" ) )
        print("Data saved at " +"out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.pickle')
        sio.savemat("out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.mat', saving)
        print("And at " +"out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.mat')
        rmtree(name_temp_file)
    except Exception as e:
        rmtree(name_temp_file)

def update(folder, initialization="", quantization=24, maxOrder=20, time_representation=False, zero_padding=True, long_term_only=False, short_term_only=False, viewPoints=["pitch", "length"], use_original_PPM=False):
    """
    Note-wise learning rate: Analyze each note, record distribution changes (KL divergence) before and after updates, and calculate IC and Entropy.
    """
    eps = 1e-12
    
    # Initialize IDyOM model with specified parameters
    L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints, evolutive=True, use_original_PPM=use_original_PPM)
    
    # Pre-training phase: load initialization data if provided
    if initialization:
        M0 = data.data(quantization=quantization)
        M0.parse(initialization, augment=True)
        L.train(M0)
    # If no initialization provided, skip pre-training and rely on incremental updates
    # Collect all MIDI files from the target folder recursively
    files = [f for f in glob(folder.rstrip('/')+'/**', recursive=True) if f.endswith(('.mid','.midi'))]
    results = {}  # Store results for each file and viewpoint
    file_order = []
    piece_starts = []
    cum_count = 0
    
    # Process each MIDI file with progress tracking
    for filepath in tqdm(files, desc="Note-wise update"):
        # Extract clean filename for results dictionary
        fname = os.path.splitext(os.path.basename(filepath))[0].replace('-', '_')
        # Record order and start offset
        file_order.append(fname)
        piece_starts.append(cum_count)
        
        # Parse current MIDI file
        Mfile = data.data(quantization=quantization)
        Mfile.parseFile(filepath)
        file_out = {}  # Store results for current file
        
        # Analyze each viewpoint (pitch, length, etc.) separately
        for vp_idx, vp in enumerate(viewPoints):
            seq = Mfile.viewPointRepresentation[vp][0]  # Get sequence for this viewpoint
            note_list = []  # Store metrics for each note in sequence
            
            # Process each note in the sequence (skip first note as it has no context)
            for i in range(1, len(seq)):
                # Always use up to maxOrder most recent notes as context
                start = max(0, i - maxOrder)
                ctx = seq[start:i]
                note = seq[i]  # Current note to predict
                
                # Pre-update per-order LTM distributions (slice ctx by order)
                pre_ltm_order_dist = []
                # order 0
                dist0 = L.LTM[vp_idx].modelOrder0.getPrediction()
                pre_ltm_order_dist.append(dist0)
                # orders 1..maxOrder
                for mc in L.LTM[vp_idx].models:
                    order_n = mc.order
                    ctx_n = ctx[-order_n:] if len(ctx) >= order_n else ctx
                    dist = mc.getPrediction(ctx_n)
                    # Fallback: if this order hasn't learned any symbols yet, create uniform distribution
                    # Use current note as minimal alphabet if even order-0 is empty
                    if not dist:
                        if dist0:
                            dist = {str(k): 1.0/len(dist0) for k in dist0.keys()}
                        else:
                            # Even order-0 is empty, use current note as alphabet
                            dist = {str(note): 1.0}
                    pre_ltm_order_dist.append(dist)
                
                # **FIX: 真正的增量训练 - 只用单个新观察更新模型**
                # 为每个order构建对应的转换并更新计数
                
                # Update order-0 model (just add the new note)
                modelOrder0 = L.LTM[vp_idx].modelOrder0
                new_note_str = str(note)
                if new_note_str not in modelOrder0.SUM:
                    modelOrder0.stateAlphabet.append(new_note_str)
                    modelOrder0.SUM[new_note_str] = 0
                modelOrder0.SUM[new_note_str] += 1
                modelOrder0.globalCounter += 1
                
                # Update higher-order models incrementally
                for mc in L.LTM[vp_idx].models:
                    order_n = mc.order
                    if len(ctx) >= order_n:  # Only update if we have enough context
                        # Extract the specific n-gram transition: ctx[-order:] -> note
                        state = ctx[-order_n:]
                        state_str = str(list(state))
                        target_str = str(note)
                        
                        # Initialize state if never seen
                        if state_str not in mc.observationsProbas:
                            mc.stateAlphabet.append(state_str)
                            mc.SUM[state_str] = 0
                            mc.observationsProbas[state_str] = {}
                        
                        # Add target to alphabet if new
                        if target_str not in mc.alphabet:
                            mc.alphabet.append(target_str)
                        
                        # Increment counts for this specific transition
                        if target_str not in mc.observationsProbas[state_str]:
                            mc.observationsProbas[state_str][target_str] = 1
                        else:
                            mc.observationsProbas[state_str][target_str] += 1
                        mc.SUM[state_str] += 1
                
                # Post-update per-order LTM distributions (slice ctx by order)
                post_ltm_order_dist = []
                # order 0
                post0 = L.LTM[vp_idx].modelOrder0.getPrediction()
                post_ltm_order_dist.append(post0)
                for mc in L.LTM[vp_idx].models:
                    order_n = mc.order
                    ctx_n = ctx[-order_n:] if len(ctx) >= order_n else ctx
                    post_dist = mc.getPrediction(ctx_n)
                    # Fallback: same as pre-update
                    if not post_dist:
                        if post0:
                            post_dist = {str(k): 1.0/len(post0) for k in post0.keys()}
                        else:
                            # Even order-0 is empty, use current note as alphabet
                            post_dist = {str(note): 1.0}
                    post_ltm_order_dist.append(post_dist)
                
                # Compute KL divergence and JS divergence per order
                KL_ltm_per_order = []
                JS_ltm_per_order = []
                for pre, post in zip(pre_ltm_order_dist, post_ltm_order_dist):
                    keys = set(pre.keys()) | set(post.keys())  # Dynamic union of all symbols
                    
                    # Handle special cases for KL divergence
                    if not pre and post:
                        # From empty to non-empty distribution: use cross-entropy
                        kl_val = -sum(post[k] * math.log(post[k] + eps, 2) for k in post.keys())
                    elif pre and not post:
                        # From non-empty to empty: undefined, use large value
                        kl_val = 100.0  # Large penalty
                    elif not pre and not post:
                        # Both empty: no change
                        kl_val = 0.0
                    else:
                        # Standard KL divergence calculation
                        kl_val = 0.0
                        for k in keys:
                            p_val = pre.get(k, eps)  # Use eps instead of 0 to avoid log(0)
                            q_val = post.get(k, eps)
                            if p_val > eps:  # Only include terms where p > 0
                                kl_val += p_val * math.log(p_val / q_val, 2)
                    
                    KL_ltm_per_order.append(kl_val)
                    
                    # JS divergence (symmetric)
                    if not pre and not post:
                        js_val = 0.0
                    else:
                        # Ensure both distributions are non-empty for JS calculation
                        pre_safe = pre if pre else {k: eps for k in post.keys()}
                        post_safe = post if post else {k: eps for k in pre.keys()}
                        keys_safe = set(pre_safe.keys()) | set(post_safe.keys())
                        
                        M = {k:(pre_safe.get(k, eps)+post_safe.get(k, eps))/2 for k in keys_safe}
                        KL_pre_M = sum(pre_safe.get(k, eps) * math.log((pre_safe.get(k, eps))/(M[k]), 2) for k in keys_safe)
                        KL_post_M = sum(post_safe.get(k, eps) * math.log((post_safe.get(k, eps))/(M[k]), 2) for k in keys_safe)
                        js_val = 0.5 * KL_pre_M + 0.5 * KL_post_M
                    
                    JS_ltm_per_order.append(js_val)
                
                # Probability merging LTM and STM (same as getSurprisefromFile)
                p_pre = L.getLikelihood(L.LTM[vp_idx], ctx, note, short_term_only=short_term_only, long_term_only=long_term_only)
                
                # Approximate entropy per viewpoint: LTM-only then STM-only then merged
                model = L.LTM[vp_idx]
                # Long-term entropy (approx across orders)
                e1 = model.getEntropy(ctx, genuine_entropies=False)
                e2 = None
                if not long_term_only:
                    # build STM for this viewpoint
                    STM = longTermModel(vp, maxOrder=maxOrder, STM=True, init=ctx, use_original_PPM=use_original_PPM)
                    STM.train([ctx], shortTerm=True)
                    # 先算一下 STM 的 P(ctx->note)，让它的 entropies[str(ctx)] 被填充
                    _ = STM.getLikelihood(ctx, note)
                    e2 = STM.getEntropy(ctx, genuine_entropies=False)
                
                # choose or merge entropies
                if long_term_only:
                    Ent_pre = e1
                elif short_term_only:
                    Ent_pre = e2
                elif e2 is not None and L.stm:
                    Ent_pre = L.mergeProbas([
                        e1, e2
                    ], [
                        model.getRelativeEntropy(ctx, genuine_entropies=False),
                        STM.getRelativeEntropy(ctx, genuine_entropies=False)
                    ])
                else:
                    Ent_pre = e1
                
                IC_pre = -math.log(p_pre + eps, 2)
                
                # Store metrics: IC, Entropy, per-order LTM KL, and per-order LTM JS
                note_list.append({
                    'idx': i,
                    'IC': IC_pre,
                    'Entropy': Ent_pre,
                    'KL_ltm_per_order': KL_ltm_per_order,
                    'JS_ltm_per_order': JS_ltm_per_order
                })
            
            file_out[vp] = note_list  # Store note list for this viewpoint
        
        # Update cumulative note count (all viewpoints have same count)
        note_count = len(file_out[viewPoints[0]])
        cum_count += note_count
        results[fname] = file_out  # Store file results
    
    # Attach meta info for plotting boundaries
    results['_meta_'] = {'file_order': file_order, 'piece_starts': piece_starts}
    # Save results to output directory
    base = os.path.basename(folder.rstrip('/'))  # Get folder name for output
    out_dir = os.path.join('out', base+'_notewise')
    os.makedirs(out_dir, exist_ok=True)  # Create output directory
    
    # Save in both MATLAB and pickle formats
    sio.savemat(os.path.join(out_dir, base+'_notewise.mat'), results)
    pickle.dump(results, open(os.path.join(out_dir, base+'_notewise.pickle'),'wb'))
    print(f"Note-wise learning rate saved to: {out_dir}")


def SurpriseOverFolder(folderTrain, folder, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
                                        zero_padding=True, long_term_only=False, short_term_only=False,\
                                        viewPoints=["pitch", "length"], genuine_entropies=False, use_original_PPM=False):

    '''
    Train a model (or load it if already saved) and evaluate it on the passed folder.
    Computed the surprise signal for each file in the folder
    '''

    L = idyom.idyom()

    if folderTrain[-1] == "/":
        folderTrain = folderTrain[:-1]

    if folder[-1] != "/":
        folder += "/"

    name_train = folderTrain[folderTrain[:-1].rfind("/")+1:] + "/"

    name = folder[folder[:-1].rfind("/")+1:]

    if not os.path.exists("out/"+name):
        os.makedirs("out/"+name)

    if not os.path.exists("out/"+name+"surprises/"):
        os.makedirs("out/"+name+"surprises/")

    if not os.path.exists("out/"+name+"surprises/"+name_train):
        os.makedirs("out/"+name+"surprises/"+name_train)

    if not os.path.exists("out/"+name+"surprises/"+name_train+"data/"):
        os.makedirs("out/"+name+"surprises/"+name_train+"data/")

    if not os.path.exists("out/"+name+"surprises/"+name_train+"figs/"):
        os.makedirs("out/"+name+"surprises/"+name_train+"figs/")

    ppm_name = "_originalPPM" if use_original_PPM else ""
    if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model"):
        print("We load saved model.")
        L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model")
    else:
        print("No saved model found, please train before.")
        print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model")
        quit()

    # ============= updated to handle KL return ===========
    S, _, _, files = L.getSurprisefromFolder(folder, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)
    # ============= end of KL handling ====================

    data = {}

    for i in range(len(S)):
        name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
        name_tmp = name_tmp.replace("-", "_")
        data[name_tmp] = np.array(S[i]).tolist()

    more_info = ""
    if long_term_only:
        more_info += "_longTermOnly"
    if short_term_only:
        more_info += "_shortTermOnly" 

    ppm_name = "_originalPPM" if use_original_PPM else ""
    more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name

    # ============= updated to handle KL return ===========
    S, E, K, files = L.getSurprisefromFolder(folder, time_representation=time_representation, \
                                        long_term_only=long_term_only, short_term_only=short_term_only, genuine_entropies=genuine_entropies)
    # ============= end of KL handling ====================

    data = {}

    for i in range(len(S)):
        name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
        name_tmp = name_tmp.replace("-", "_")
        # ============= updated to include KL =================
        data[name_tmp] = [np.array(S[i]).tolist(), np.array(E[i]).tolist(), np.array(K[i]).tolist()]
        # ============= end of KL inclusion ===================

    # ============= updated info description ==============
    data["info"] = "Each file corresponds to a song. For each song you have the Information Content as the first element, Relative Entropy as the second element, and KL divergence as the third element. They are all vectors over the time dimension."
    # ============= end of info update ====================

    sio.savemat("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
    pickle.dump(data, open("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

    print()
    print()
    print()
    print("Data (surprises/IC)have been succesfully saved in:","out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
    print("Including a .mat for matlab purpose and a .pickle for python purpose.")
    print()
    print()


def SilentNotesOverFolder(folderTrain, folder, threshold=0.3, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
                                        zero_padding=True, long_term_only=False, short_term_only=False, \
                                        viewPoints=["pitch", "length"], use_original_PPM=False):

    '''
    Function used in The music of silence. Part II: Cortical Predictions during Silent Musical Intervals (https://www.jneurosci.org/content/41/35/7449)
    It computes the probabily to have a note in each natural musical silences (using only the duration/rythm dimension). 
    '''

    L = idyom.idyom()

    if folderTrain[-1] == "/":
        folderTrain = folderTrain[:-1]

    if folder[-1] != "/":
        folder += "/"

    name_train = folderTrain[folderTrain[:-1].rfind("/")+1:] + "/"

    name = folder[folder[:-1].rfind("/")+1:]

    if not os.path.exists("out/"+name):
        os.makedirs("out/"+name)

    if not os.path.exists("out/"+name+"missing_notes/"):
        os.makedirs("out/"+name+"missing_notes/")

    if not os.path.exists("out/"+name+"missing_notes/"+name_train):
        os.makedirs("out/"+name+"missing_notes/"+name_train)

    if not os.path.exists("out/"+name+"missing_notes/"+name_train+"data/"):
        os.makedirs("out/"+name+"missing_notes/"+name_train+"data/")

    if not os.path.exists("out/"+name+"missing_notes/"+name_train+"figs/"):
        os.makedirs("out/"+name+"missing_notes/"+name_train+"figs/")


    if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+vstr(viewPoints)+ ".model"):
        print("We load saved model.")
        L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ".model")
    else:
        print("No saved model found, please train before.")
        print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ".model")
        quit()

    S, files = L.getDistributionsfromFolder(folder, threshold, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)

    data = {}

    for i in range(len(S)):
        name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
        name_tmp = name_tmp.replace("-", "_")
        data[name_tmp] = np.array(S[i]).tolist()

    more_info = ""
    if long_term_only:
        more_info += "_longTermOnly"
    if short_term_only:
        more_info += "_shortTermOnly" 

    ppm_name = "_originalPPM" if use_original_PPM else ""
    more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name


    sio.savemat("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
    pickle.dump(data, open("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

    print()
    print()
    print()
    print("Data (surprises/IC)have been succesfully saved in:","out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
    print("Including a .mat for matlab purpose and a .pickle for python purpose.")
    print()
    print()


if __name__ == "__main__":

    usage = "usage %prog [options]"
    parser = OptionParser(usage)

    # Create directory tree
    if not os.path.exists("out/"):
        os.makedirs("out/")

    parser.add_option("-a", "--test", type="int",
                      help="1 if you want to launch unittests",
                      dest="tests", default=0)

    parser.add_option("-t", "--train", type="string",
                  help="Train the model with the passed folder",
                  dest="train_folder", default=None)

    parser.add_option("-s", "--surprise", type="string",
                  help="Compute surprise over the passed folder. We use -t argument to train, if none are privided, we use the passed folder to cross-train.",
                  dest="trial_folder", default=None)

    parser.add_option("-n", "--silentNotes", type="string",
                  help="Compute silent notes probabilities over the passed folder. We use -t argument to train, if none are provided, we use the passed folder to cross-train.",
                  dest="trial_folder_silent", default=None)

    parser.add_option("-d", "--threshold_missing_notes", type="float",
                  help="Define the threshold for choosing the missing notes (0.2 by default)",
                  dest="threshold_missing_notes", default=0.2)

    parser.add_option("-z", "--zero_padding", type="int",
                  help="Specify if you want to use zero padding in the surprise output, enable time representation (default 0)",
                  dest="zero_padding", default=None)

    parser.add_option("-b", "--short_term", type="int",
                      help="Only use short term model (default 0)",
                      dest="short_term_only", default=0)

    parser.add_option("-c", "--cross_eval", type="string",
                      help="Compute likelihoods by pieces over the passed dataset using k-fold cross-eval.",
                      dest="cross_eval", default=None)

    parser.add_option("-l", "--long_term", type="int",
                      help="Only use long term model (default 0)",
                      dest="long_term_only", default=0)

    parser.add_option("-k", "--k_fold", type="int",
                      help="Specify the k-fold for all cross-eval, you can use -1 for leave-one-out (default 5).",
                      dest="k_fold", default=5)

    parser.add_option("-q", "--quantization", type="int",
                      help="Rythmic quantization to use (default 24).",
                      dest="quantization", default=24)

    parser.add_option("-v", "--viewPoints", type= "string", action='callback', callback=foo_callback,
                      help="Viewpoints to use: pitch, length, interval and velocity, separate them with comas, default pitch,length.",
                      dest="viewPoints", default=['pitch', 'length'])

    parser.add_option("-m", "--max_order", type="int",
                      help="Maximal order to use (default 20).",
                      dest="max_order", default=20)        

    parser.add_option("-g", "--genuine_entropies", type="int",
                      help="Use this parameter to NOT use the entropy approximation. It takes longer (5 times) to compute but generate the genuine entropies, not an approximation (default 0).",
                      dest="genuine_entropies", default=0)        

    parser.add_option("-r", "--check_dataset", type="string",
                      help="Check wether the passed folder contains duplicates.",
                      dest="folder_duplicates", default="")    

    parser.add_option("-e", "--evolution", type="string",
                      help="Train and evaluate over training on the passed folder (cross-val).",
                      dest="train_test_folder", default=None)    

    parser.add_option("-u", "--update", type="string",
                      help="Per-note evolution: train on each note and report IC and Entropy; can initialize with -i.",
                      dest="update_folder", default=None)

    parser.add_option("-i", "--init_evolution", type="string",
                      help="Folder to initialize the evolution on.",
                      dest="intialization", default="")    

    parser.add_option("-p", "--nb_pieces", type="int",
                      help="Number of pieces to evaluate on during evolution training.",
                      dest="nb_pieces", default=20)

    parser.add_option("-o", "--original_PPM", type="int",
                      help="Use the original Prediction by Partial Matching (PPM) mothod C",
                      dest="use_original_PPM", default=0)


    options, arguments = parser.parse_args()
    options.lisp = "" # Temporary

    if options.zero_padding is not None:
        time_representation = True
    else:
        time_representation = False

    if options.update_folder:
        print("Per-note evolution ...")
        update(options.update_folder, initialization=options.intialization, quantization=options.quantization, maxOrder=options.max_order, time_representation=time_representation, zero_padding=options.zero_padding==1, long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1, viewPoints=options.viewPoints, use_original_PPM=options.use_original_PPM==1)

    if options.train_test_folder is not None:
        print("Evolution Training ...")
        Train_by_piece_with_all_metrics(options.train_test_folder, nb_pieces=options.nb_pieces, quantization=options.quantization, maxOrder=options.max_order, \
                                    time_representation=time_representation, zero_padding=options.zero_padding==1, \
                                    long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
                                    viewPoints=options.viewPoints, intialization=options.intialization, use_original_PPM=options.use_original_PPM==1)        

    if options.train_folder is not None:

        isValid = type(options.viewPoints) is type(["", ""])
        if isValid:
            for elem in options.viewPoints:
                if type(elem) is not type(""):
                    isValid = False

        if not isValid: 
            print()
            print("The viewpoints you gave: " + str(options.viewPoints)+". Are not in the correct format. Please separate them with comas without spaces. Example: length,pitch.")
            quit()

        if options.viewPoints[-1] == "":
            print("Please do not use spaces between the viewpoint parameters.")
            quit()

        print("Viewpoints to use: "+ str(options.viewPoints))

        print("Training ...")
        Train(options.train_folder, quantization=options.quantization, maxOrder=options.max_order, \
                                    time_representation=time_representation, zero_padding=options.zero_padding==1, \
                                    long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
                                    viewPoints=options.viewPoints, use_original_PPM=options.use_original_PPM==1)

    if options.cross_eval is not None:
        print("Evaluation on", str(options.cross_eval), "...")
        evaluation(str(options.cross_eval), k_fold=options.k_fold, quantization=options.quantization, maxOrder=options.max_order, \
                                            time_representation=time_representation, zero_padding=options.zero_padding==1, \
                                            long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
                                            viewPoints=options.viewPoints, genuine_entropies=options.genuine_entropies==1, use_original_PPM=options.use_original_PPM==1)

    if options.trial_folder is not None:
        if options.train_folder is None:
            print("You did not provide a train folder, please provide one or use the --cross_eval option to cross-evaluate this folder.")
            quit()
        SurpriseOverFolder(options.train_folder, options.trial_folder, \
                            k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
                            time_representation=time_representation, zero_padding=options.zero_padding==1, \
                            long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
                            viewPoints=options.viewPoints, genuine_entropies=options.genuine_entropies==1, use_original_PPM=options.use_original_PPM==1)

    if options.trial_folder_silent is not None:
        if options.train_folder is None:
            print("You did not provide a train folder, please provide one or use the --cross_eval option to cross-evaluate this folder.")
            quit()


        SilentNotesOverFolder(options.train_folder, options.trial_folder_silent, threshold=options.threshold_missing_notes, \
                            k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
                            time_representation=time_representation, zero_padding=options.zero_padding==1, \
                            long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
                            viewPoints=options.viewPoints, use_original_PPM=options.use_original_PPM==1)

    if options.folder_duplicates != "":    
        checkDataSet(options.folder_duplicates)

    if options.lisp != "":    
        compareWithLISP(options.lisp)
    
    if options.tests == 1:
        loader = unittest.TestLoader()

        start_dir = "unittests/"
        suite = loader.discover(start_dir)

        runner = unittest.TextTestRunner()
        runner.run(suite)
