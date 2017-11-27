import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate_pytorch import *
from tqdm import tqdm
import gzip
import pickle
from memory_profiler import profile
#import pdb

FIXED_PARAMETERS, config = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]

if not os.path.exists(FIXED_PARAMETERS["log_path"]):
    os.makedirs(FIXED_PARAMETERS["log_path"])
if not os.path.exists(config.tbpath):
    os.makedirs(config.tbpath)
    config.tbpath = FIXED_PARAMETERS["log_path"]

if config.test:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_test.log"
else:
    logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'DIIN')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################


if config.debug_model:
    # training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_matched, test_mismatched = [],[],[],[],[],[], [], []
    #@profile
    def load():
        test_matched = load_nli_data(FIXED_PARAMETERS["dev_snli"], shuffle = False)[:499]
        training_snli, dev_snli, test_snli = test_matched, test_matched,test_matched
        indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([test_matched])
        shared_content = load_mnli_shared_content()
        return training_snli, dev_snli, test_snli, indices_to_words, word_indices, char_indices, indices_to_chars, shared_content
    training_snli, dev_snli, test_snli, indices_to_words, word_indices, char_indices, indices_to_chars, shared_content = load()
else:

    logger.Log("Loading data SNLI")
    training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
    dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
    test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)
    shared_content = load_mnli_shared_content()
    """
    logger.Log("Loading data MNLI")
    training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
    dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
    dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

    test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], shuffle = False)
    test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], shuffle = False)

    shared_content = load_mnli_shared_content()
    """


    logger.Log("Loading embeddings")
    indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_snli, dev_snli, test_snli])

config.char_vocab_size = len(char_indices.keys())

#@profile
def embedding():
    embedding_dir = os.path.join(config.datapath, "embeddings")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)


    embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

    print("embedding path exist")
    print(os.path.exists(embedding_path))
    if os.path.exists(embedding_path):
        f = gzip.open(embedding_path, 'rb')
        loaded_embeddings = pickle.load(f)
        f.close()
    else:
        loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)
        f = gzip.open(embedding_path, 'wb')
        pickle.dump(loaded_embeddings, f)
        f.close()
    return loaded_embeddings
loaded_embeddings = embedding()


#@profile
def get_minibatch(dataset, start_index, end_index, training=False):
    indices = range(start_index, end_index)

    #genres = [dataset[i]['genre'] for i in indices]
    labels = [dataset[i]['label'] for i in indices]
    pairIDs = np.array([dataset[i]['pairID'] for i in indices])


    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)


    premise_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
    hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)
    premise_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=config.char_in_word_size)
    hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([dataset[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=config.char_in_word_size)

    premise_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence1_parse'][:] for i in indices], premise_pad_crop_pair)
    hypothesis_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence2_parse'][:] for i in indices], hypothesis_pad_crop_pair)

    premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(indices))], premise_pad_crop_pair, 1)
    hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(indices))], hypothesis_pad_crop_pair, 1)

    premise_exact_match = np.expand_dims(premise_exact_match, 2)
    hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)

    labels = torch.LongTensor(labels)

    minibatch_premise_vectors = torch.stack([torch.from_numpy(v) for v in premise_vectors]).squeeze().type('torch.LongTensor')
    minibatch_hypothesis_vectors = torch.stack([torch.from_numpy(v) for v in hypothesis_vectors]).squeeze().type('torch.LongTensor')

    minibatch_pre_pos = torch.stack([torch.from_numpy(v) for v in premise_pos_vectors]).squeeze().type('torch.LongTensor')
    minibatch_hyp_pos = torch.stack([torch.from_numpy(v) for v in hypothesis_pos_vectors]).squeeze().type('torch.LongTensor')

    premise_char_vectors = torch.stack([torch.from_numpy(v) for v in premise_char_vectors]).squeeze().type('torch.LongTensor')
    hypothesis_char_vectors = torch.stack([torch.from_numpy(v) for v in hypothesis_char_vectors]).squeeze().type('torch.LongTensor')
    premise_exact_match = torch.stack([torch.from_numpy(v) for v in premise_exact_match]).squeeze().type('torch.LongTensor')
    hypothesis_exact_match = torch.stack([torch.from_numpy(v) for v in hypothesis_exact_match]).squeeze().type('torch.LongTensor')

    return minibatch_premise_vectors, minibatch_hypothesis_vectors, labels, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match


#@profile
def train(model, loss_, optim, batch_size, config, train_snli, dev_snli):
    #sess_config = tf.ConfigProto()
    #sess_config.gpu_options.allow_growth=True   
    #self.sess = tf.Session(config=sess_config)
    #self.sess.run(self.init)

    display_epoch_freq = 1
    #display_step = config.display_step
    display_step = 1
    eval_step = config.eval_step
    save_step = config.eval_step
    embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
    dim = FIXED_PARAMETERS["hidden_embedding_dim"]
    emb_train = FIXED_PARAMETERS["emb_train"]
    keep_rate = FIXED_PARAMETERS["keep_rate"]
    sequence_length = FIXED_PARAMETERS["seq_length"] 
    alpha = FIXED_PARAMETERS["alpha"]
    config = config

    logger.Log("Building model from %s.py" %(model))
    model.train()
    #self.global_step = self.model.global_step

    # tf things: initialize variables and create placeholder for session
    logger.Log("Initializing variables")

    #self.init = tf.global_variables_initializer()
    #self.sess = None
    #self.saver = tf.train.Saver()

    step = 1
    epoch = 0
    best_dev_mat = 0.
    best_mtrain_acc = 0.
    last_train_acc = [.001, .001, .001, .001, .001]
    best_step = 0
    train_dev_set = False
    dont_print_unnecessary_info = False
    collect_failed_sample = False

    # Restore most recent checkpoint if it exists. 
    # Also restore values for best dev-set accuracy and best training-set accuracy


    ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
    if os.path.isfile(ckpt_file + ".meta"):
        if os.path.isfile(ckpt_file + "_best.meta"):
            #self.saver.restore(self.sess, (ckpt_file + "_best"))
            model.load_state_dict(torch.load(ckpt_file + "_best"))
            completed = False
            #dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(classify, dev_matched, batch_size, completed, model, loss_)
            #best_dev_mismat, dev_cost_mismat, _ = evaluate_classifier(classify, dev_mismatched, batch_size, completed, model, loss_)
            best_dev_snli, dev_cost_snli, confmx = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)
            #best_mtrain_acc, mtrain_cost, _ = evaluate_classifier(classify, train_mnli[0:5000], batch_size, completed, model, loss_)
            logger.Log("Confusion Matrix on dev-snli\n{}".format(confmx))
            if alpha != 0.:
                best_strain_acc, strain_cost, _  = evaluate_classifier(classify, train_snli[0:5000], batch_size, completed, model, loss_)
                logger.Log("Restored best SNLI-dev acc: %f\n Restored best SNLI train acc: %f" %(best_dev_snli, best_strain_acc))
            else:
                logger.Log("Restored best SNLI-dev acc: %f" %(best_dev_snli))
            if config.training_completely_on_snli:
                best_dev_mat = best_dev_snli
        else:
            model.load_state_dict(torch.load(ckpt_file))
        logger.Log("Model restored from file: %s" % ckpt_file)

    ### Training cycle
    logger.Log("Training...")

    while True:
        training_data = train_snli

        random.shuffle(training_data)
        avg_cost = 0.
        total_batch = int(len(training_data) / batch_size)
        
        # Boolean stating that training has not been completed, 
        completed = False 

        # Loop over all batches in epoch
        for i in range(total_batch):

            # Assemble a minibatch of the next B examples
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match  = get_minibatch(
                training_data, batch_size * i, batch_size * (i + 1), True)
            
            if config.cuda:
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
                minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match = minibatch_premise_vectors.cuda(), minibatch_hypothesis_vectors.cuda(), minibatch_labels.cuda(), \
                minibatch_pre_pos.cuda(), minibatch_hyp_pos.cuda(), premise_char_vectors.cuda(), hypothesis_char_vectors.cuda(), \
                premise_exact_match.cuda(), hypothesis_exact_match.cuda()           

            minibatch_premise_vectors = Variable(minibatch_premise_vectors)
            minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

            minibatch_pre_pos = Variable(minibatch_pre_pos)
            minibatch_hyp_pos = Variable(minibatch_hyp_pos)

            premise_char_vectors = Variable(premise_char_vectors)
            hypothesis_char_vectors = Variable(hypothesis_char_vectors)
            premise_exact_match = Variable(premise_exact_match)
            hypothesis_exact_match = Variable(hypothesis_exact_match)

            minibatch_labels = Variable(minibatch_labels)

            model.zero_grad()
            # Run the optimizer to take a gradient step, and also fetch the value of the 
            # cost function for logging

            model.dropout_rate_decay(step)
            output = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
                minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match)
            logger.Log("Finish forward")
            print("Finish forward{}".format(step))
            #pdb.set_trace()
            #print(model.parameters())
            lossy = loss_(output, minibatch_labels)

            diff_loss = F.mse_loss(model.self_attention_linear_p.weight.data, model.self_attention_linear_h.weight.data) * torch.numel(model.self_attention_linear_p.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p1.weight.data, model.fuse_gate_linear_h1.weight.data) * torch.numel(model.fuse_gate_linear_p1.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p2.weight.data, model.fuse_gate_linear_h2.weight.data) * torch.numel(model.fuse_gate_linear_p2.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p3.weight.data, model.fuse_gate_linear_h3.weight.data) * torch.numel(model.fuse_gate_linear_p3.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p4.weight.data, model.fuse_gate_linear_h4.weight.data) * torch.numel(model.fuse_gate_linear_p4.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p5.weight.data, model.fuse_gate_linear_h5.weight.data) * torch.numel(model.fuse_gate_linear_p5.weight.data) / 2.0 + \
                        F.mse_loss(model.fuse_gate_linear_p6.weight.data, model.fuse_gate_linear_h6.weight.data) * torch.numel(model.fuse_gate_linear_p6.weight.data) / 2.0

            diff_loss *= config.diff_penalty_loss_ratio

            lossy += diff_loss

            logger.Log("loss{}".format(lossy.data[0]))
            lossy.backward()
            logger.Log("Finish backward{}".format(step))
            print("Finish backward{}".format(step))
            torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), config.gradient_clip_value)
            optim.step()

            #print(step)
            if step % display_step == 0:
                logger.Log("Step: {} completed".format(step))
                print("Step: {} completed".format(step))
            if step % eval_step == 1:
                """
                if config.training_completely_on_snli and dont_print_unnecessary_info:
                    dev_acc_mat = dev_cost_mat = 1.0
                else:
                    logger.Log("start eval dev_matched:") 
                    dev_acc_mat, dev_cost_mat, confmx = evaluate_classifier(classify, dev_matched, batch_size, completed, model, loss_)
                    logger.Log("Confusion Matrix on dev-matched\n{}".format(confmx))
                """

                #if config.training_completely_on_snli:
                logger.Log("start eval dev_snli:") 
                print("start eval dev_snli:")
                dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)
                #dev_acc_mismat, dev_cost_mismat = 0,0
                #elif not dont_print_unnecessary_info or 100 * (1 - best_dev_mat / dev_acc_mat) > 0.04:
                    #dev_acc_mismat, dev_cost_mismat, _ = evaluate_classifier(classify, dev_mismatched, batch_size, completed, model, loss_)
                    #dev_acc_snli, dev_cost_snli, _ = evaluate_classifier(classify, dev_snli, batch_size, completed, model, loss_)

                #else:
                    #dev_acc_mismat, dev_cost_mismat, dev_acc_snli, dev_cost_snli = 0,0,0,0

                #if dont_print_unnecessary_info and config.training_completely_on_snli:
                    #print("mtrain_acc init")
                    #mtrain_acc, mtrain_cost = 0.0, 0.0
                #else:
                #mtrain_acc, mtrain_cost, _ = evaluate_classifier(classify, train_mnli[0:5000], batch_size, completed, model, loss_)
                strain_acc, strain_cost,_ = evaluate_classifier(classify, train_snli[0:5000], batch_size, completed, model, loss_)

                logger.Log("Step: %i\t Dev-SNLI acc: %f\t SNLI train acc: %f" %(step, dev_acc_snli, strain_acc))
                logger.Log("Step: %i\t Dev-SNLI cost: %f\t SNLI train cost: %f" %(step, dev_cost_snli, strain_cost))

                print("Finish eval")
                """
                if alpha != 0.:
                    if not dont_print_unnecessary_info or 100 * (1 - best_dev_mat / dev_acc_mat) > 0.04:
                        strain_acc, strain_cost,_ = evaluate_classifier(classify, train_snli[0:5000], batch_size, completed, model, loss_)
                    elif config.training_completely_on_snli:
                        strain_acc, strain_cost,_ = evaluate_classifier(classify, train_snli[0:5000], batch_size, completed, model, loss_)
                    else:
                        strain_acc, strain_cost = 0, 0
                    logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f\t SNLI train acc: %f" %(step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc, strain_acc))
                    logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f\t SNLI train cost: %f" %(step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost, strain_cost))
                else:
                    logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t MultiNLI train acc: %f" %(step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc))
                    logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t MultiNLI train cost: %f" %(step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost))
                """
            if step % save_step == 1:
                torch.save(model, ckpt_file)
                if config.training_completely_on_snli:
                    #print("mtrain acc cal")
                    dev_acc_mat = dev_acc_snli
                    #mtrain_acc = strain_acc
                best_test = 100 * (1 - best_dev_mat / dev_acc_mat)
                if best_test > 0.04:
                    torch.save(model, ckpt_file + "_best")
                    best_dev_mat = dev_acc_mat
                    #best_mtrain_acc = mtrain_acc
                    if alpha != 0.:
                        best_strain_acc = strain_acc
                    best_step = step
                    logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(best_dev_mat))

            if best_dev_mat > 0.872 and config.training_completely_on_snli:
                eval_step = 500
                save_step = 500
            
            if best_dev_mat > 0.878 and config.training_completely_on_snli:
                eval_step = 100
                save_step = 100
                dont_print_unnecessary_info = True 

            step += 1

            # Compute average loss
            avg_cost += lossy.data[0] / (total_batch * batch_size)
                            
        # Display some statistics about the epoch
        if epoch % display_epoch_freq == 0:
            logger.Log("Epoch: %i\t Avg. Cost: %f" %(epoch+1, avg_cost))
        print("Epoch: %i\t Avg. Cost: %f" %(epoch+1, avg_cost))
        epoch += 1 

        last_train_acc[(epoch % 5) - 1] = strain_acc

        # Early stopping
        early_stopping_step = 35000
        progress = 1000 * (sum(last_train_acc)/(5 * min(last_train_acc)) - 1) 

        
        if (progress < 0.1) or (step > best_step + early_stopping_step):
            logger.Log("Best dev accuracy: %s" %(best_dev_mat))
            logger.Log("SNLI Train accuracy: %s" %(best_strain_acc))
            if config.training_completely_on_snli:
                train_dev_set = True

                completed = True
                break
            else:
                completed = True
                break


def classify(examples, completed, batch_size, model, loss_):
    model.eval()
    # This classifies a list of examples
    if (test == True) or (completed == True):
        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        model.load_state_dict(torch.load(best_path))
        logger.Log("Model restored from file: %s" % best_path)

    total_batch = int(len(examples) / batch_size)
    pred_size = 3 
    logits = np.empty(pred_size)
    #total = 0
    genres = []
    costs = 0
    correct = 0
    for i in range(total_batch):
        #if i != total_batch:
        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match  = get_minibatch(
            examples, batch_size * i, batch_size * (i + 1))
        """
        else:
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
            minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match = get_minibatch(
                examples, batch_size * i, len(examples))
        """
            
        if config.cuda:
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match = minibatch_premise_vectors.cuda(), minibatch_hypothesis_vectors.cuda(), minibatch_labels.cuda(), \
            minibatch_pre_pos.cuda(), minibatch_hyp_pos.cuda(), premise_char_vectors.cuda(), hypothesis_char_vectors.cuda(), \
            premise_exact_match.cuda(), hypothesis_exact_match.cuda()           

        minibatch_premise_vectors = Variable(minibatch_premise_vectors)
        minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

        minibatch_pre_pos = Variable(minibatch_pre_pos)
        minibatch_hyp_pos = Variable(minibatch_hyp_pos)

        premise_char_vectors = Variable(premise_char_vectors)
        hypothesis_char_vectors = Variable(hypothesis_char_vectors)
        premise_exact_match = Variable(premise_exact_match)
        hypothesis_exact_match = Variable(hypothesis_exact_match)

        minibatch_labels = Variable(minibatch_labels)
        #logger.Log("Classsify - finish loading")
        #genres += minibatch_genres
        #logger.Log("genre".format(genres))
        logit = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match)
        #logger.Log("Classsify - finish forward")
        cost = loss_(logit, minibatch_labels).data[0]
        #logger.Log("Classsify - finish cost")
        costs += cost
        #predicted = torch.max(logit.data, 1)[1]
        #total += minibatch_labels.size(0)
        #return correct / float(total), costs, 0
        #print(predicted)
        #print(minibatch_labels.data)
        #correct += (predicted == minibatch_labels.data).sum()
        #logger.Log('correct'.format(correct))
        logits = np.vstack([logits, logit.data.numpy()])
    
    '''
    if test == True:
        logger.Log("Generating Classification error analysis script")
        correct_file = open(os.path.join(FIXED_PARAMETERS["log_path"], "correctly_classified_pairs.txt"), 'w')
        wrong_file = open(os.path.join(FIXED_PARAMETERS["log_path"], "wrongly_classified_pairs.txt"), 'w')

        pred = np.argmax(logits[1:], axis=1)
        LABEL = ["entailment", "neutral", "contradiction"]
        for i in range(pred.shape[0]):
            if pred[i] == examples[i]["label"]:
                fh = correct_file
            else:
                fh = wrong_file
            fh.write("S1: {}\n".format(examples[i]["sentence1"].encode('utf-8')))
            fh.write("S2: {}\n".format(examples[i]["sentence2"].encode('utf-8')))
            fh.write("Label:      {}\n".format(examples[i]['gold_label']))
            fh.write("Prediction: {}\n".format(LABEL[pred[i]]))
            fh.write("confidence: \nentailment: {}\nneutral: {}\ncontradiction: {}\n\n".format(logits[1+i, 0], logits[1+i,1], logits[1+i,2]))

        correct_file.close()
        wrong_file.close()
    '''
    return genres, np.argmax(logits[1:], axis=1), costs

def generate_predictions_with_id(path, examples, completed, batch_size, model, loss_):
    if (test == True) or (completed == True):
        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        model.load_state_dict(torch.load(best_path))
        logger.Log("Model restored from file: %s" % best_path)

    total_batch = int(len(examples) / batch_size)
    pred_size = 3
    logits = np.empty(pred_size)
    costs = 0
    IDs = np.empty(1)
    for i in range(total_batch + 1):
        if i != total_batch:
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
            hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
            hypothesis_NER_feature  = get_minibatch(
                examples, batch_size * i, batch_size * (i + 1))
        else:
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
            hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
            hypothesis_NER_feature  = get_minibatch(
                examples, batch_size * i, len(examples))

        if config.cuda:
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match = minibatch_premise_vectors.cuda(), minibatch_hypothesis_vectors.cuda(), minibatch_labels.cuda(), \
            minibatch_pre_pos.cuda(), minibatch_hyp_pos.cuda(), premise_char_vectors.cuda(), hypothesis_char_vectors.cuda(), \
            premise_exact_match.cuda(), hypothesis_exact_match.cuda()           

        minibatch_premise_vectors = Variable(minibatch_premise_vectors)
        minibatch_hypothesis_vectors = Variable(minibatch_hypothesis_vectors)

        minibatch_pre_pos = Variable(minibatch_pre_pos)
        minibatch_hyp_pos = Variable(minibatch_hyp_pos)

        premise_char_vectors = Variable(premise_char_vectors)
        hypothesis_char_vectors = Variable(hypothesis_char_vectors)
        premise_exact_match = Variable(premise_exact_match)
        hypothesis_exact_match = Variable(hypothesis_exact_match)

        minibatch_labels = Variable(minibatch_labels)

        logit = model(minibatch_premise_vectors, minibatch_hypothesis_vectors, \
            minibatch_pre_pos, minibatch_hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
            premise_exact_match, hypothesis_exact_match)
        IDs = np.concatenate([IDs, pairIDs])
        logits = np.vstack([logits, logit])
    IDs = IDs[1:]
    logits = np.argmax(logits[1:], axis=1)
    save_submission(path, IDs, logits)

batch_size = FIXED_PARAMETERS["batch_size"]
completed = False



model = MyModel(config, FIXED_PARAMETERS["seq_length"], emb_dim=FIXED_PARAMETERS["word_embedding_dim"],  hidden_dim=FIXED_PARAMETERS["hidden_embedding_dim"], embeddings=loaded_embeddings, emb_train=FIXED_PARAMETERS["emb_train"])

if config.cuda:
    model.cuda()

#optim = torch.optim.Adadelta(model.parameters(), lr = FIXED_PARAMETERS["learning_rate"])
optim = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr = FIXED_PARAMETERS["learning_rate"])
loss = nn.CrossEntropyLoss() 

test = params.train_or_test()
logger.Log('test'.format(test))

if config.preprocess_data_only:
    pass
elif test == False:
    train(model, loss, optim, batch_size, config, training_snli, dev_snli)
    completed = True
    logger.Log("Acc on SNLI test-set: %s" %(evaluate_classifier(classify, test_snli, FIXED_PARAMETERS["batch_size"], completed, model, loss)[0]))

    logger.Log("Generating SNLI dev pred")
    dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
    generate_predictions_with_id(dev_snli_path, dev_snli, completed, batch_size, model, loss)

    logger.Log("Generating SNLI test pred")
    test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
    generate_predictions_with_id(test_snli_path, test_snli, completed, batch_size, model, loss)
        
else:
    if config.training_completely_on_snli:
        logger.Log("Generating SNLI dev pred")
        dev_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_dev_{}.csv".format(modname))
        generate_predictions_with_id(dev_snli_path, dev_snli, completed, batch_size, model, loss)

        logger.Log("Generating SNLI test pred")
        test_snli_path = os.path.join(FIXED_PARAMETERS["log_path"], "snli_test_{}.csv".format(modname))
        generate_predictions_with_id(test_snli_path, test_snli, completed, batch_size, model, loss)
    