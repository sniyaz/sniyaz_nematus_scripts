import sys
sys.path.append("../../nlp_segment")

from bpe import apply_merge_ops, delimit_corpus, get_vocabulary, apply_presegs, segment_vocab, recover_preseg_boundary 
from copy import deepcopy
from collections import Counter


if __name__ == '__main__':

    src = sys.argv[1]
    target = sys.argv[2]
    training_prefix = sys.argv[3]
    validation_prefix = sys.argv[4]

    src_presegs_file = sys.argv[5]
    target_presegs_file = sys.argv[6]

    #Load in presegs for each language
    src_presegs = open(src_presegs_file, "rb")
    src_presegs = pickle.load(src_presegs)
    target_presegs = open(target_presegs_file, "rb")
    target_presegs = pickle.load(target_presegs)

    #PHASE 1: THE TRAINING CORPORA
    
    #Apply the presegs to the training corpora
    src_training_file = os.path.join(training_prefix, ".tc." + src)
    src_training_obj = open(src_training_file, "r")
    src_training_vocab = get_vocabulary(src_training_obj)
    src_training_obj.close()
    src_training_preseg = apply_presegs(src_training_vocab, src_presegs)

    target_training_file = os.path.join(training_prefix, ".tc." + target)
    target_training_obj = open(target_training_file, "r")
    target_training_vocab = get_vocabulary(target_training_obj)
    target_training_obj.close()
    target_training_preseg = apply_presegs(target_training_vocab, target_presegs)

    #Merge the two vocabularies together.
    combined_training_vocab = Counter()
    for word in target_training_preseg:
        combined_training_vocab[word] +=  target_training_preseg[word]
    for word in src_training_preseg:
        combined_training_vocab[word] += src_training_preseg[word]

    #TODO: Avoid hard-coding
    num_iters = 89500
    #Train BPE
    merge_operations, training_segmentations = segment_vocab(combined_training_vocab, num_iters) 

    #Recover segmentations of training corpora and write then out
    final_src_train_seg = recover_preseg_boundary(src_training_vocab, src_presegs, training_segmentations)
    final_target_train_seg = recover_preseg_boundary(target_training_vocab, target_presegs, training_segmentations)

    src_train_output = os.path.join(training_prefix, ".bpe." + src)
    delimit_corpus(src_training_file, src_train_output, final_src_train_seg)  

    target_train_output = os.path.join(training_prefix, ".bpe." + target)
    delimit_corpus(target_training_file, target_train_output, final_target_train_seg)  

    #PHASE 2: THE VALIDATION CORPORA

    #Apply the presegs to the training corpora
    src_validation_file = os.path.join(validation_prefix, ".tc." + src)
    src_validation_obj = open(src_validation_file, "r")
    src_validation_vocab = get_vocabulary(src_validation_obj)
    src_validation_obj.close()
    src_validation_preseg = apply_presegs(src_validation_vocab, src_presegs)

    target_validation_file = os.path.join(validation_prefix, ".tc." + target)
    target_validation_obj = open(target_validation_file, "r")
    target_validation_vocab = get_vocabulary(target_validation_obj)
    target_validation_obj.close()
    target_validation_preseg = apply_presegs(target_validation_vocab, target_presegs)
   
    #Apply trained BPE operations to validation corpora
    src_val_inter_seg = apply_merge_ops(src_validation_preseg, merge_operations)
    target_val_inter_seg = apply_merge_ops(target_validation_preseg, merge_operations)

    #Recover final segmentations of validation corpora and write them out.
    final_src_val_seg = recover_preseg_boundary(src_validation_vocab, src_presegs, src_val_inter_seg)
    final_target_val_seg = recover_preseg_boundary(target_validation_vocab, target_presegs, target_val_inter_seg)

    src_val_output = os.path.join(validation_prefix, ".bpe." + src)
    delimit_corpus(src_validation_file, src_val_output, final_src_val_seg)  

    target_val_output = os.path.join(validation_prefix, ".bpe." + target)
    delimit_corpus(target_validation_file, target_val_output, final_target_val_seg)  






