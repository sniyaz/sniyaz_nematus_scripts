import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

import pdb

def extract_bleu(bleu_file):
    bleu_obj = open(bleu_file, "r")
    line = bleu_obj.readline()
    parts = line.split()
    bleu = parts[2]
    if bleu[-1] == ",":
        bleu = bleu[:-1]
    bleu = float(bleu)
    return bleu

if __name__ == '__main__':
    
    model_dir = sys.argv[1]
    output_pic_path = sys.argv[2]
   
    main_model_path = os.path.join(model_dir, "model.npz")
    main_bleu_path = os.path.join(model_dir, "model.npz_bleu_scores")
    
    models = os.listdir(model_dir)
    models = [model_file for model_file in models if model_file[:10] == "model.iter" and model_file[-4:] == ".npz"]
    models.sort(key = lambda x: int(x[10:-4]))
    iters = []
    scores = []
    for cur_model in models:

	iters_run = cur_model[10:-4]
	iters_run = int(iters_run)
	iters.append(iters_run)
        
	os.system("cp " + os.path.join(model_dir, cur_model) + " " + main_model_path)
        os.system("./validate.sh")
        cur_bleu = extract_bleu(main_bleu_path)
        scores.append(cur_bleu)

    plt.plot(range(1, len(iters) + 1), scores, "b")
    plt.plot(range(1, len(iters) + 1), scores, "bo")
    plt.xlabel("Iterations: Steps of 30000")
    plt.ylabel("BLEU")
    plt.savefig(output_pic_path)

        
	

    
