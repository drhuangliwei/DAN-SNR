# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:13:55 2018
Tensorflow Implementation of DAN-SNR: A Deep Attentive Network for Social-Aware Next Point-of-Interest Recommendation model in:
Liwei Huang, Yutao Ma, Yanbo Liu, and Keqing He. DAN-SNR: A Deep Attentive Network for Social-Aware Next Point-of-Interest Recommendation. arXiv.org, arXiv: 2004.12161 (2020).
@author: Liwei Huang (dr_huanglw@163.com)
"""

import tensorflow as tf
import numpy as np
import time
import collections,os
from modules import *
from sample_prepare import *

class Config(object):
    """config."""
    learning_rate = 0.00003
    keep_prob = 0.9     
    num_steps = 50     #length of shrot-term channel
    l_num_steps = 200   #length of long-term and social channel
    hidden_size = 256      #demension of hidden layers    
    max_max_epoch = 20   
    batch_size = 50    
    user_size = 10000    #number of users  
    location_size = 10000   #number of POIs
    negative_sample_num = 500  #number of negative samples for each sample
    init_scale = 1.0  
    train_dir = '/train'
    eval_dir = '/eval'   
    
class attentive_Model:
    
    def __init__(self, Config):
        
        self.num_steps=Config.num_steps 
        self.l_num_steps=Config.l_num_steps 
        self.hidden_size=Config.hidden_size
        self.user_size=Config.user_size
        self.location_size=Config.location_size
        self.negative_sample_num=Config.negative_sample_num
        self.train_dir=Config.train_dir
        self.global_steps = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.maximum(1e-10, tf.train.exponential_decay(Config.learning_rate,
                                                            self.global_steps,
                                                            decay_steps=200,
                                                            decay_rate=0.96,
                                                            staircase=True))
        #building model
        self.init_placeholder()
        self.build_model()
            
    def init_placeholder(self):
        
        #user IDs of two channels
        self.u=tf.placeholder(tf.int32, shape=(None,))
        self.l_u=tf.placeholder(tf.int32, shape=(None, self.l_num_steps))
        
        #POI IDs of two channels and target POI 
        self.s_x = tf.placeholder(tf.int32, shape=(None, self.num_steps)) 
        self.l_x = tf.placeholder(tf.int32, shape=(None, self.l_num_steps))
        self.y = tf.placeholder(tf.int32, shape=(None,))
        
        #user embeddings, which is pretrained on the location-based social network by node2vec
        self.user_vecs = tf.placeholder(tf.float32, shape=(self.user_size, self.hidden_size)) #表示所有用户的分布式表示
        #POI location embedding, which is pretrainedon the L2L graph by node2vec
        self.location_vecs = tf.placeholder(tf.float32, shape=(self.location_size, self.hidden_size)) #表示所有位置的分布式表示
        
        #Time embedding for each chechins in two channels and target POI 
        self.s_time_x = tf.placeholder(tf.int32, shape=(None, self.num_steps)) 
        self.l_time_x = tf.placeholder(tf.int32, shape=(None, self.l_num_steps))  
        self.time_y = tf.placeholder(tf.int32, shape=(None,)) 
        
        #location-based social network
        self.network = tf.placeholder(tf.int32, shape=(self.user_size, self.user_size)) 
        
        #negative samples for each sample
        self.negative_sample = tf.placeholder(tf.int32, shape=(None, self.negative_sample_num))
        
        #mask matrix
        self.short_mask=tf.placeholder(tf.int32, shape=(None, self.num_steps))
        self.long_mask=tf.placeholder(tf.int32, shape=(None, self.l_num_steps))

    def build_model(self):
        
        batch_size=tf.shape(self.u)[0]
       
        with tf.variable_scope("input_embedding"): 
          
            user_embedding=tf.nn.embedding_lookup(self.user_vecs,self.u) #[batch_size,hidden_size]
            user_embedding_x=tf.tile(tf.expand_dims(user_embedding,1),[1, self.num_steps, 1]) #[batch_size,num_steps,hidden_size]
            user_embedding_y=tf.expand_dims(user_embedding,1) #[batch_size,1,hidden_size]
            user_embedding_l=tf.nn.embedding_lookup(self.user_vecs,self.l_u) #[batch_size,l_num_steps,hidden_size]
            user_embedding_n=tf.tile(tf.expand_dims(user_embedding,1),[1, self.negative_sample_num, 1]) 
            user_embedding_a=tf.tile(tf.expand_dims(user_embedding,1),[1, self.location_size, 1])
            
            #嵌入POI
            poi_emb = tf.get_variable('poi_emb', dtype=tf.float32, shape=[self.location_size, self.hidden_size],initializer=tf.contrib.layers.xavier_initializer())
            poi_embedding_x = tf.nn.embedding_lookup(poi_emb,self.s_x) #[batch_size,num_steps,hidden_size]
            poi_embedding_y = tf.expand_dims(tf.nn.embedding_lookup(poi_emb,self.y),1) #[batch_size,1,hidden_size]
            poi_embedding_l = tf.nn.embedding_lookup(poi_emb,self.l_x) #[batch_size,l_num_steps,hidden_size]
            poi_embedding_n = tf.nn.embedding_lookup(poi_emb,self.negative_sample) 
            poi_embedding_a = tf.tile(tf.expand_dims(poi_emb,0),[batch_size, 1, 1])
            
            location_embedding_x=tf.nn.embedding_lookup(self.location_vecs,self.s_x) #[batch_size,num_steps,hidden_size]
            location_embedding_y=tf.expand_dims(tf.nn.embedding_lookup(self.location_vecs,self.y),1) #[batch_size,1,hidden_size]
            location_embedding_l=tf.nn.embedding_lookup(self.location_vecs,self.l_x) #[batch_size,l_num_steps,hidden_size]
            location_embedding_n=tf.nn.embedding_lookup(self.location_vecs,self.negative_sample)
            location_embedding_a = tf.tile(tf.expand_dims(self.location_vecs,0),[batch_size, 1, 1])
            
            time_emb = tf.get_variable('time_emb', dtype=tf.float32, shape=[20, self.hidden_size],initializer=tf.contrib.layers.xavier_initializer())
            time_embedding_x = tf.nn.embedding_lookup(time_emb,self.s_time_x) #[batch_size,num_steps,hidden_size]
            time_embedding_y = tf.expand_dims(tf.nn.embedding_lookup(time_emb,self.time_y),1) #[batch_size,1,hidden_size]
            time_embedding_l = tf.nn.embedding_lookup(time_emb,self.l_time_x) #[batch_size,l_num_steps,hidden_size]
            time_embedding_n = tf.tile(time_embedding_y, [1,self.negative_sample_num, 1])
            time_embedding_a = tf.tile(time_embedding_y, [1,self.location_size, 1])
            
            non_zeros=tf.stack([tf.range(batch_size),tf.reduce_sum(self.short_mask,-1)],axis=1)
            position_embedding=tf.cast(positional_encoding(batch_size,self.num_steps+1,num_units=self.hidden_size,zero_pad=False,scale=False),dtype=tf.float32)
            position_embedding_x=position_embedding[:,:-1,:]
            position_embedding_y=tf.expand_dims(tf.gather_nd(position_embedding,non_zeros),1)
            position_embedding_n = tf.tile(position_embedding_y, [1,self.negative_sample_num, 1])            
            position_embedding_a = tf.tile(position_embedding_y, [1,self.location_size, 1])

            W_U = tf.tile(tf.expand_dims(tf.get_variable("W_U", [self.hidden_size, self.hidden_size], dtype=tf.float32),0),[batch_size,1,1])  #用户输入的权值
            W_X = tf.tile(tf.expand_dims(tf.get_variable("W_X", [self.hidden_size, self.hidden_size], dtype=tf.float32),0),[batch_size,1,1])  #POI输入的权值
            W_l = tf.tile(tf.expand_dims(tf.get_variable("W_l", [self.hidden_size, self.hidden_size], dtype=tf.float32),0),[batch_size,1,1])  #位置输入的权值
            W_T = tf.tile(tf.expand_dims(tf.get_variable("W_T", [self.hidden_size, self.hidden_size], dtype=tf.float32),0),[batch_size,1,1])  #时间输入的权值
            W_p = tf.tile(tf.expand_dims(tf.get_variable("W_p", [self.hidden_size, self.hidden_size], dtype=tf.float32),0),[batch_size,1,1])  #序列输入的权值
                                                         
            x_embedding=tf.matmul(user_embedding_x ,W_U)+tf.matmul(poi_embedding_x ,W_X)+tf.matmul(location_embedding_x ,W_l)+tf.matmul(time_embedding_x ,W_T)+tf.matmul(position_embedding_x ,W_p) #[batch_size,num_steps,hidden_size]
            y_embedding=tf.matmul(user_embedding_y ,W_U)+tf.matmul(poi_embedding_y ,W_X)+tf.matmul(location_embedding_y ,W_l)+tf.matmul(time_embedding_y ,W_T)+tf.matmul(position_embedding_y ,W_p) #[batch_size,1,hidden_size]
            yl_embedding=tf.matmul(user_embedding_y ,W_U)+tf.matmul(poi_embedding_y ,W_X)+tf.matmul(location_embedding_y ,W_l)+tf.matmul(time_embedding_y ,W_T)
            l_embedding=tf.matmul(user_embedding_l ,W_U)+tf.matmul(poi_embedding_l ,W_X)+tf.matmul(location_embedding_l ,W_l)+tf.matmul(time_embedding_l ,W_T) #[batch_size,l_num_steps,hidden_size]
            n_embedding=tf.matmul(user_embedding_n ,W_U)+tf.matmul(poi_embedding_n ,W_X)+tf.matmul(location_embedding_n ,W_l)+tf.matmul(time_embedding_n ,W_T)+tf.matmul(position_embedding_n ,W_p)
            nl_embedding=tf.matmul(user_embedding_n ,W_U)+tf.matmul(poi_embedding_n ,W_X)+tf.matmul(location_embedding_n ,W_l)+tf.matmul(time_embedding_n ,W_T)
            a_embedding=tf.matmul(user_embedding_a ,W_U)+tf.matmul(poi_embedding_a ,W_X)+tf.matmul(location_embedding_a ,W_l)+tf.matmul(time_embedding_a ,W_T)+tf.matmul(position_embedding_a ,W_p)
            al_embedding=tf.matmul(user_embedding_a ,W_U)+tf.matmul(poi_embedding_a ,W_X)+tf.matmul(location_embedding_a ,W_l)+tf.matmul(time_embedding_a ,W_T)
            
            short_mask_t=tf.tile(tf.expand_dims(self.short_mask,-1),[1,1,self.hidden_size])
            long_mask_t=tf.tile(tf.expand_dims(self.long_mask,-1),[1,1,self.hidden_size])
            
            x_embedding=x_embedding*tf.cast(short_mask_t,dtype=tf.float32)
            l_embedding=l_embedding*tf.cast(long_mask_t,dtype=tf.float32)
            
            self.encoding_x=x_embedding
            self.encoding_l=l_embedding
            
            
        #modeling short-term influence
        with tf.variable_scope("short_attention"):
        
            for i in range(5):
                with tf.variable_scope("s_num_blocks_{}".format(i)):
            
                    self.encoding_x=mask_self_attention(self.encoding_x, self.encoding_x, num_units=self.hidden_size,num_heads=8, dropout_rate=Config.keep_prob,is_training=True,causality=False)
                   
                        # Feed Forward          
                    self.encoding_x = feedforward(self.encoding_x, num_units=[8*self.hidden_size, self.hidden_size]) #[batch_size,user_size*num_steps,hidden_size]
        
        #modeling long-term and social influence
        with tf.variable_scope("long_attention"):
            for i in range(5):
                with tf.variable_scope("l_num_blocks_{}".format(i)):
            
                    self.encoding_l=mask_self_attention(self.encoding_l, self.encoding_l, num_units=self.hidden_size,num_heads=8, dropout_rate=Config.keep_prob,is_training=True,causality=False)
                   
                        # Feed Forward          
                    self.encoding_l = feedforward(self.encoding_l, num_units=[8*self.hidden_size, self.hidden_size]) #[batch_size,user_size*num_steps,hidden_size]
                                                  
        
        with tf.variable_scope("Vanilla_attention"):   
                
            ns_attention=Van_attention(n_embedding, self.encoding_x, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=True,causality=False)
            nl_attention=Van_attention(nl_embedding, self.encoding_l, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=True,causality=False)
          
            ys_attention=Van_attention(y_embedding, self.encoding_x, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=True,causality=False)
            yl_attention=Van_attention(yl_embedding, self.encoding_l, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=True,causality=False)
                 
            ns_output=tf.reduce_sum(tf.multiply(ns_attention, n_embedding),-1)/(self.hidden_size**0.5) #[batch_size,negative_sample_num]  
            nl_output=tf.reduce_sum(tf.multiply(nl_attention,nl_embedding),-1)/(self.hidden_size**0.5) 
            n_prediction=ns_output+nl_output+tf.reduce_sum(tf.multiply(user_embedding_n,poi_embedding_n),-1)  #[batch_size,negative_sample_num]
                
            ys_output=tf.reduce_sum(tf.multiply(ys_attention, y_embedding),-1)/(self.hidden_size**0.5) #[batch_size,1]  
            yl_output=tf.reduce_sum(tf.multiply(yl_attention,yl_embedding),-1)/(self.hidden_size**0.5)
            y_prediction=ys_output+yl_output+tf.reduce_sum(tf.multiply(user_embedding_y,poi_embedding_y),-1)  #[batch_size,1]
                
            Pro_positive_sample_all=tf.tile(y_prediction, [1, self.negative_sample_num])#[batch_size,negative_sample_num]               
            ranking_loss=tf.Variable(0.0, trainable=False) 
            
            #bpr loss
            ranking_loss = tf.log(1.0+tf.exp(-tf.to_float(Pro_positive_sample_all-n_prediction))) #[batch_size,negative_sample_num]
               
            self.loss=tf.reduce_sum(ranking_loss)/tf.cast((batch_size*self.negative_sample_num),tf.float32)
                
            tf.summary.scalar('loss',self.loss)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op=optimizer.minimize(self.loss, global_step=self.global_steps)
            
            #prediction phase
            
            as_attention=Van_attention(a_embedding, self.encoding_x, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=False,causality=False)
            al_attention=Van_attention(al_embedding, self.encoding_l, num_units=self.hidden_size, dropout_rate=Config.keep_prob,is_training=False,causality=False)
                
            as_output=tf.reduce_sum(tf.multiply(as_attention, a_embedding),-1)/(self.hidden_size**0.5) 
            al_output=tf.reduce_sum(tf.multiply(al_attention,al_embedding),-1)/(self.hidden_size**0.5) 
            self.a_prediction=as_output+al_output+tf.reduce_sum(tf.multiply(user_embedding_a,poi_embedding_a),-1)
                
            self.recall_5, self.ndcg_at_5 = self._metric_at_k(5)
            self.recall_10, self.ndcg_at_10 = self._metric_at_k(10)
                
            tf.summary.scalar('recall_5',self.recall_5)
            tf.summary.scalar('recall_10',self.recall_10)
            tf.summary.scalar('ndcg_at_5',self.ndcg_at_5)
            tf.summary.scalar('ndcg_at_10',self.ndcg_at_10) 
            
            self.train_summary=tf.summary.merge_all()
    
    def _metric_at_k(self, k=20):
        prediction = self.a_prediction #(batch, n_items)
        prediction_transposed = tf.transpose(prediction)
        labels = self.y
        batch_size = tf.shape(self.u)[0]
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.location_size])
        ranks = tf.reduce_sum(tf.cast(prediction > tile_pred_values, dtype=tf.float32), -1) + 1 #(batch)
        ndcg = 1. / (tf.log(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k)
        recall = tf.reduce_sum(tf.cast(hit_at_k, dtype=tf.float32))/tf.cast(batch_size,tf.float32)
        ndcg_at_k = tf.reduce_sum(ndcg * tf.cast(hit_at_k, dtype=tf.float32))/tf.cast(batch_size,tf.float32)
        return recall, ndcg_at_k                                                                
        
    def save(self, sess, checkpoint_dir, step):
        saver = tf.train.Saver()
        model_name = "SWRec.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
               
        
def run_epoch(model,session,data,data_f,val_data, val_data_f, batch_size,location_Vector,user_embedding,network,summary_writer,checkpoint_dir,global_steps):
     
    fetches = [model.loss,model.train_op]
    highest_val_recall = -1.0
    patience = 20
    inc = 0
    early_stopping = False

    for step, (u,l_u,s_x,l_x,y,s_time_x,l_time_x,time_y,negative_sample,short_mask,long_mask) in enumerate(batch_iter_sample(data,data_f,batch_size)):
        
        time0 = time.time()
        
        feed_dict={}

        feed_dict[model.u]=u
        feed_dict[model.l_u]=l_u

        feed_dict[model.s_x]=s_x
        feed_dict[model.l_x]=l_x
        feed_dict[model.y]=y

        feed_dict[model.user_vecs]=user_embedding

        feed_dict[model.s_time_x]=s_time_x
        feed_dict[model.l_time_x]=l_time_x
        feed_dict[model.time_y]=time_y
        
        feed_dict[model.location_vecs]=location_Vector
      
        feed_dict[model.network]=network

        feed_dict[model.negative_sample]=negative_sample

        feed_dict[model.short_mask]=short_mask
        feed_dict[model.long_mask]=long_mask

        cost,_=session.run(fetches,feed_dict)
        
        if(global_steps%500==0):
            print("Valiate the model")
            total_recall_5,total_ndcg_at_5,total_recall_10,total_ndcg_at_10=evaluate(model,session,val_data, val_data_f,batch_size,location_Vector,user_embedding,network)
            if total_recall_10 >= highest_val_recall:
                model.save(session,checkpoint_dir,global_steps)
                highest_val_recall = total_recall_10
                inc = 0
                print("the test data total_recall_5 is %f,total_ndcg_at_5 is %f,total_recall_10 is %f,total_ndcg_at_10 is %f"%(total_recall_5,total_ndcg_at_5,total_recall_10,total_ndcg_at_10)) 
            else:
                inc += 1
            if inc >= patience:
                early_stopping = True
                break
        
        if(global_steps%10==0):
            print("the %i step, train cost is: %f, time:%f"%(global_steps,cost,time.time()-time0))
            result=session.run(model.train_summary,feed_dict)
            summary_writer.add_summary(result,global_steps)
            
        if(global_steps%1000==0):
            model.save(session,checkpoint_dir,global_steps)
        global_steps+=1
           
    return cost,global_steps,early_stopping
    
def evaluate(model,session,data,data_f,batch_size,location_Vector,user_embedding,network):
    
    total_num=0
    total_recall_5=0.0
    total_ndcg_at_5=0.0
    total_recall_10=0.0
    total_ndcg_at_10=0.0
    
    fetches = [model.recall_5,model.ndcg_at_5,model.recall_10,model.ndcg_at_10]

    for step, (u,l_u,s_x,l_x,y,s_time_x,l_time_x,time_y,short_mask,long_mask) in enumerate(batch_iter(data,data_f,batch_size)):
        
        total_num=total_num+1
        feed_dict={}

        feed_dict[model.u]=u
        feed_dict[model.l_u]=l_u

        feed_dict[model.s_x]=s_x
        feed_dict[model.l_x]=l_x
        feed_dict[model.y]=y

        feed_dict[model.user_vecs]=user_embedding

        feed_dict[model.s_time_x]=s_time_x
        feed_dict[model.l_time_x]=l_time_x
        feed_dict[model.time_y]=time_y
        
        feed_dict[model.location_vecs]=location_Vector
      
        feed_dict[model.network]=network

        feed_dict[model.short_mask]=short_mask
        feed_dict[model.long_mask]=long_mask

        c_recall_5,c_ndcg_at_5,c_recall_10,c_ndcg_at_10=session.run(fetches,feed_dict)
        total_recall_5=total_recall_5+c_recall_5
        total_ndcg_at_5=total_ndcg_at_5+c_ndcg_at_5
        total_recall_10=total_recall_10+c_recall_10
        total_ndcg_at_10=total_ndcg_at_10+c_ndcg_at_10
        
    total_recall_5=total_recall_5/total_num
    total_ndcg_at_5=total_ndcg_at_5/total_num
    total_recall_10=total_recall_10/total_num
    total_ndcg_at_10=total_ndcg_at_10/total_num
    
    return total_recall_5,total_ndcg_at_5,total_recall_10,total_ndcg_at_10

def count_params():
    total_parameters=0
    for variable in tf.trainable_variables():
        if variable.name=='poi_emb:0':
            continue
        if variable.name=='time_emb:0':
            continue
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
    
    
if __name__=='__main__':

    print("loading the dataset...")  
    config=Config()
    eval_config=Config()

    checkpoint_dir='/models'
       
    userlocation = np.loadtxt(open('data/userlocation.csv','rb'),delimiter=',',skiprows=0) 
    last_network=np.loadtxt(open('data/network.csv','rb'),delimiter=',',skiprows=0)
    poi_voc=collections.Counter(userlocation[1,:].tolist())
    user_voc=collections.Counter(userlocation[0,:].tolist())
    N=len(user_voc.keys())
    network=network_trans(last_network,N)
    locations=np.loadtxt(open('data/locations.csv','rb'),delimiter=',',skiprows=0)
    user_embedding=np.loadtxt('data/user_embedding.csv',delimiter=',',skiprows=0)
    location_embedding=np.loadtxt('data/location_embedding.csv',delimiter=',',skiprows=0)
    clusters=new_build_location_voc(locations)
    
    user_embedding=user_embedding.astype(np.float32)
    location_embedding=location_embedding.astype(np.float32)
    userlocation=userlocation.astype(np.float32)
    network=network.astype(np.float32)
    locations=locations.astype(np.float32)
    
    print("bulid the sample...")
    top_500=pop_n(userlocation,500)
    nofriend,addfriend=build_squence(userlocation, network,200)
    
    print("bulid the train and test set...")
    final_train_set,final_test_set,final_val_set,final_train_set_f,final_test_set_f,final_val_set_f=build_samples(nofriend,addfriend, 0.1, 0.1, locations, 500, clusters,top_500,50,200)    
    
    config.user_size=len(user_voc.keys())
    config.location_size=len(poi_voc.keys())
    config.batch_size=50
    eval_config.user_size=len(user_voc.keys())
    eval_config.location_size=len(poi_voc.keys())
    eval_config.batch_size=10
    
    print("begin the model training...")
    with tf.Graph().as_default(), tf.Session() as session:
        
        summary_writer = tf.summary.FileWriter('./tmp/logs',session.graph) 
        initializer = tf.random_normal_initializer()
        
        model = attentive_Model(config)  
        
        init_op = tf.global_variables_initializer()
        session.run(init_op)   
        print('Trainable params: {}'.format(count_params()))
        global_steps=1
        
        for i in range(config.max_max_epoch):
               
            print("the %d epoch training..."%(i+1))
            start_time=int(time.time())
            cost,global_steps,early_stopping=run_epoch(model,session,final_train_set,final_train_set_f,final_val_set,final_val_set_f,config.batch_size,location_embedding,user_embedding,network,summary_writer,checkpoint_dir,global_steps)
            
            print("Epoch: [%2d] time: %4.4f, cost: %.8f" \
                    % (i, time.time() - start_time, cost))
            
            if early_stopping:
                print('Early stop at epoch: {}, total training steps: {}'.format(i, global_steps))
                break
        
        print("begin the model testing...")
        model.load(session, checkpoint_dir)
        total_recall_5,total_ndcg_at_5,total_recall_10,total_ndcg_at_10=evaluate(model,session,final_test_set,final_test_set_f,eval_config.batch_size,location_embedding,user_embedding,network)
        print("the test data total_recall_5 is %f,total_ndcg_at_5 is %f,total_recall_10 is %f,total_ndcg_at_10 is %f"%(total_recall_5,total_ndcg_at_5,total_recall_10,total_ndcg_at_10))  
        print("program end!")
    
    

    
    
    
    
    
    
    
    
    
                


































