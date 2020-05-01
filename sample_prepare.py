# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:13:55 2018
Tensorflow Implementation of DAN-SNR: A Deep Attentive Network for Social-Aware Next Point-of-Interest Recommendation model in:
Liwei Huang, Yutao Ma, Yanbo Liu, and Keqing He. DAN-SNR: A Deep Attentive Network for Social-Aware Next Point-of-Interest Recommendation. arXiv.org, arXiv: 2004.12161 (2020).
@author: Liwei Huang (dr_huanglw@163.com)
"""

import numpy as np
import collections
import math

def build_squence(userlocation, network, longlen):
           
    user_voc=collections.Counter(userlocation[0,:].tolist())
   
    user_list=[]
    poi_list=[]
    time_list=[]
    latitude_list=[]
    longitude_list=[]
    
    all_user_list=[]
    all_poi_list=[]
    all_time_list=[]
    all_latitude_list=[]
    all_longitude_list=[]

    friend_list={}
    user_b={}

    for user in list(user_voc.keys()):
        
        checkin_user_redex=np.argwhere(userlocation[0,:]==user)  
        checkin_user_all=userlocation[:,checkin_user_redex[:,0]]
        
        sequence_user=[]
        sequence_poi=[]  
        sequence_time=[]
        sequence_latitude=[]
        sequence_longitude=[]
          
        sorted_time=np.sort(checkin_user_all[2,:]) 
        sorted_time_index=np.argsort(checkin_user_all[2,:])
        
        for i in range(len(checkin_user_redex)):
            
            if i==0: 
                sequence_user.append(checkin_user_all[0,sorted_time_index[i]])
                sequence_poi.append(checkin_user_all[1,sorted_time_index[i]])
                sequence_time.append(checkin_user_all[2,sorted_time_index[i]])
                sequence_latitude.append(checkin_user_all[3,sorted_time_index[i]])
                sequence_longitude.append(checkin_user_all[4,sorted_time_index[i]])
                   
            else:                
                if sorted_time[i]-sorted_time[i-1]>21600: #six hours
                    if len(sequence_poi)>4: 
                        sequence_user=list(map(int, sequence_user))
                        sequence_poi=list(map(int, sequence_poi))
                        sequence_time=list(map(int, sequence_time))
                        sequence_latitude=list(map(np.float32, sequence_latitude))
                        sequence_longitude=list(map(np.float32, sequence_longitude))
                        
                        user_list.append(sequence_user)
                        poi_list.append(sequence_poi)
                        time_list.append(sequence_time)
                        latitude_list.append(sequence_latitude)
                        longitude_list.append(sequence_longitude)
                        
                        sequence_user=[]
                        sequence_poi=[]  
                        sequence_time=[]
                        sequence_latitude=[]
                        sequence_longitude=[]
                        
                        sequence_user.append(checkin_user_all[0,sorted_time_index[i]])
                        sequence_poi.append(checkin_user_all[1,sorted_time_index[i]])
                        sequence_time.append(checkin_user_all[2,sorted_time_index[i]])
                        sequence_latitude.append(checkin_user_all[3,sorted_time_index[i]])
                        sequence_longitude.append(checkin_user_all[4,sorted_time_index[i]])

                else:
                    sequence_user.append(checkin_user_all[0,sorted_time_index[i]])
                    sequence_poi.append(checkin_user_all[1,sorted_time_index[i]])
                    sequence_time.append(checkin_user_all[2,sorted_time_index[i]])
                    sequence_latitude.append(checkin_user_all[3,sorted_time_index[i]])
                    sequence_longitude.append(checkin_user_all[4,sorted_time_index[i]])
    
        friend_list[int(user)]=[c[0] for c in list(np.argwhere(network[int(user),:]==1.0))]
        user_b[int(user)]=list(checkin_user_redex[:,0]) 
        
    #sample augment
    new_user=[]
    new_poi=[]
    new_time=[]
    new_latitude=[]
    new_longitude=[]
    
    for k in range(len(user_list)):
        for i in range(len(user_list[k])-3):
            new_user.append(user_list[k][0:i+4])
            new_poi.append(poi_list[k][0:i+4])
            new_time.append(time_list[k][0:i+4])
            new_latitude.append(latitude_list[k][0:i+4])
            new_longitude.append(longitude_list[k][0:i+4]) 
            
    print(len(new_user))
    for i in range(len(new_user)):
        user=new_user[i][0]
        
        flitered_u=[]
        for friend in friend_list[user]:
            flitered_u=flitered_u+user_b[friend]
        
        flitered=userlocation[:,flitered_u]
        sorted_time=np.sort(flitered[2,:]) 
        sorted_time_index=np.argsort(flitered[2,:])
        user_check1=flitered[:,sorted_time_index]
 
        end_time=new_time[i][-1] 
        selected2=np.argwhere(user_check1[2,:]<end_time)
        user_check2=user_check1[:,selected2[:,0]] 
        
        if user_check2.shape[1]<longlen:
            all_sequence_user=list(user_check2[0,:])
            all_sequence_poi=list(user_check2[1,:]) 
            all_sequence_time=list(user_check2[2,:])
            all_sequence_latitude=list(user_check2[3,:])
            all_sequence_longitude=list(user_check2[4,:])
        else:
            all_sequence_user=list(user_check2[0,user_check2.shape[1]-longlen:])
            all_sequence_poi=list(user_check2[1,user_check2.shape[1]-longlen:])  
            all_sequence_time=list(user_check2[2,user_check2.shape[1]-longlen:])
            all_sequence_latitude=list(user_check2[3,user_check2.shape[1]-longlen:])
            all_sequence_longitude=list(user_check2[4,user_check2.shape[1]-longlen:])
                
        all_sequence_user=list(map(int, all_sequence_user))
        all_sequence_poi=list(map(int, all_sequence_poi))
        all_sequence_time=list(map(int, all_sequence_time))
        all_sequence_latitude=list(map(np.float32, all_sequence_latitude))
        all_sequence_longitude=list(map(np.float32, all_sequence_longitude))
        
        
        all_user_list.append(all_sequence_user)
        all_poi_list.append(all_sequence_poi)
        all_time_list.append(time_encoding(all_sequence_time,end_time))
        all_latitude_list.append(all_sequence_latitude)
        all_longitude_list.append(all_sequence_longitude)  

    new_time1=[time_encoding(ti,ti[-1]) for ti in new_time]                   
                    
    nofriend=(new_user,new_poi,new_time1,new_latitude,new_longitude) 
    addfriend=(all_user_list,all_poi_list,all_time_list,all_latitude_list,all_longitude_list) 
              
    return nofriend,addfriend
    
def build_samples(checkin_list, addfriend, test_portion,val_portion,locations, num_sample, clusters,top_500, max_len, longlen):    
       
    user_list,poi_list,time_list,latitude_list,longitude_list=checkin_list
    user_list_f,poi_list_f,time_list_f,latitude_list_f,longitude_list_f=addfriend
      
    print("bulid train and test set")
    n_samples= len(user_list)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - test_portion-val_portion)))
    n_val = int(np.round(n_samples * (1. - test_portion)))
    
    test_set_user = [user_list[s] for s in sidx[n_val:]]
    test_set_poi = [poi_list[s] for s in sidx[n_val:]]
    test_set_time= [time_list[s] for s in sidx[n_val:]]
    test_set_latitude= [latitude_list[s] for s in sidx[n_val:]]
    test_set_longitude= [longitude_list[s] for s in sidx[n_val:]]
    
    train_set_user = [user_list[s] for s in sidx[:n_train]]
    train_set_poi = [poi_list[s] for s in sidx[:n_train]]
    train_set_time= [time_list[s] for s in sidx[:n_train]]
    train_set_latitude= [latitude_list[s] for s in sidx[:n_train]]
    train_set_longitude= [longitude_list[s] for s in sidx[:n_train]]
    
    val_set_user = [user_list[s] for s in sidx[n_train:n_val]]
    val_set_poi = [poi_list[s] for s in sidx[n_train:n_val]]
    val_set_time= [time_list[s] for s in sidx[n_train:n_val]]
    val_set_latitude= [latitude_list[s] for s in sidx[n_train:n_val]]
    val_set_longitude= [longitude_list[s] for s in sidx[n_train:n_val]]

    test_set_user_x =[x[0:-1] for x in test_set_user]
    test_set_poi_x = [x[0:-1] for x in test_set_poi]
    test_set_time_x = [x[0:-1] for x in test_set_time]
    test_set_latitude_x = [x[0:-1] for x in test_set_latitude]
    test_set_longitude_x = [x[0:-1] for x in test_set_longitude]
    
    train_set_user_x =[x[0:-1] for x in train_set_user]
    train_set_poi_x = [x[0:-1] for x in train_set_poi]
    train_set_time_x = [x[0:-1] for x in train_set_time]
    train_set_latitude_x = [x[0:-1] for x in train_set_latitude]
    train_set_longitude_x = [x[0:-1] for x in train_set_longitude]
    
    val_set_user_x =[x[0:-1] for x in val_set_user]
    val_set_poi_x = [x[0:-1] for x in val_set_poi]
    val_set_time_x = [x[0:-1] for x in val_set_time]
    val_set_latitude_x = [x[0:-1] for x in val_set_latitude]
    val_set_longitude_x = [x[0:-1] for x in val_set_longitude]
    
    test_set_user_y =[x[-1] for x in test_set_user]
    test_set_poi_y = [x[-1] for x in test_set_poi]
    test_set_time_y = [x[-1] for x in test_set_time]
    test_set_latitude_y = [x[-1] for x in test_set_latitude]
    test_set_longitude_y = [x[-1] for x in test_set_longitude]

    train_set_user_y =[x[-1] for x in train_set_user]
    train_set_poi_y = [x[-1] for x in train_set_poi]
    train_set_time_y = [x[-1] for x in train_set_time]
    train_set_latitude_y = [x[-1] for x in train_set_latitude]
    train_set_longitude_y = [x[-1] for x in train_set_longitude]
    
    val_set_user_y =[x[-1] for x in val_set_user]
    val_set_poi_y = [x[-1] for x in val_set_poi]
    val_set_time_y = [x[-1] for x in val_set_time]
    val_set_latitude_y = [x[-1] for x in val_set_latitude]
    val_set_longitude_y = [x[-1] for x in val_set_longitude]
       
    print("begin padding。。。")
    new_test_set_user = padding(test_set_user_x,test_set_user_y,max_len)
    new_test_set_poi = padding(test_set_poi_x,test_set_poi_y,max_len)
    new_test_set_time = padding(test_set_time_x,test_set_time_y,max_len)
    new_test_set_latitude = padding(test_set_latitude_x,test_set_latitude_y,max_len)
    new_test_set_longitude = padding(test_set_longitude_x,test_set_longitude_y,max_len)
    
    new_train_set_user = padding(train_set_user_x,train_set_user_y,max_len)
    new_train_set_poi = padding(train_set_poi_x,train_set_poi_y,max_len)
    new_train_set_time = padding(train_set_time_x,train_set_time_y,max_len)
    new_train_set_latitude = padding(train_set_latitude_x,train_set_latitude_y,max_len)
    new_train_set_longitude = padding(train_set_longitude_x,train_set_longitude_y,max_len)
    
    new_val_set_user = padding(val_set_user_x,val_set_user_y,max_len)
    new_val_set_poi = padding(val_set_poi_x,val_set_poi_y,max_len)
    new_val_set_time = padding(val_set_time_x,val_set_time_y,max_len)
    new_val_set_latitude = padding(val_set_latitude_x,val_set_latitude_y,max_len)
    new_val_set_longitude = padding(val_set_longitude_x,val_set_longitude_y,max_len)
    
    mask_train_x=generate_mask(train_set_user_x,max_len)
    mask_test_x=generate_mask(test_set_user_x,max_len)
    mask_val_x=generate_mask(val_set_user_x,max_len)
    
    negative_set_poi = padding_negative_sample(train_set_poi_y,locations,num_sample,clusters,top_500)
    
    final_train_set=(new_train_set_user,new_train_set_poi,new_train_set_time,new_train_set_latitude,new_train_set_longitude,mask_train_x,negative_set_poi)
    final_test_set=(new_test_set_user,new_test_set_poi,new_test_set_time,new_test_set_latitude,new_test_set_longitude,mask_test_x)
    final_val_set=(new_val_set_user,new_val_set_poi,new_val_set_time,new_val_set_latitude,new_val_set_longitude,mask_val_x)
    
    test_set_user_f = [user_list_f[s] for s in sidx[n_val:]]
    test_set_poi_f = [poi_list_f[s] for s in sidx[n_val:]]
    test_set_time_f= [time_list_f[s] for s in sidx[n_val:]]
    test_set_latitude_f= [latitude_list_f[s] for s in sidx[n_val:]]
    test_set_longitude_f= [longitude_list_f[s] for s in sidx[n_val:]]
    
    train_set_user_f = [user_list_f[s] for s in sidx[:n_train]]
    train_set_poi_f = [poi_list_f[s] for s in sidx[:n_train]]
    train_set_time_f= [time_list_f[s] for s in sidx[:n_train]]
    train_set_latitude_f= [latitude_list_f[s] for s in sidx[:n_train]]
    train_set_longitude_f= [longitude_list_f[s] for s in sidx[:n_train]]
    
    val_set_user_f = [user_list_f[s] for s in sidx[n_train:n_val]]
    val_set_poi_f = [poi_list_f[s] for s in sidx[n_train:n_val]]
    val_set_time_f= [time_list_f[s] for s in sidx[n_train:n_val]]
    val_set_latitude_f= [latitude_list_f[s] for s in sidx[n_train:n_val]]
    val_set_longitude_f= [longitude_list_f[s] for s in sidx[n_train:n_val]]

    print("begin padding friend。。。")
    new_test_set_user_f = padding_f(test_set_user_f,longlen)
    new_test_set_poi_f = padding_f(test_set_poi_f,longlen)
    new_test_set_time_f = padding_f(test_set_time_f,longlen)
    new_test_set_latitude_f = padding_f(test_set_latitude_f,longlen)
    new_test_set_longitude_f = padding_f(test_set_longitude_f,longlen)
    
    new_train_set_user_f = padding_f(train_set_user_f,longlen)
    new_train_set_poi_f = padding_f(train_set_poi_f,longlen)
    new_train_set_time_f = padding_f(train_set_time_f,longlen)
    new_train_set_latitude_f = padding_f(train_set_latitude_f,longlen)
    new_train_set_longitude_f = padding_f(train_set_longitude_f,longlen)
    
    new_val_set_user_f = padding_f(val_set_user_f,longlen)
    new_val_set_poi_f = padding_f(val_set_poi_f,longlen)
    new_val_set_time_f = padding_f(val_set_time_f,longlen)
    new_val_set_latitude_f = padding_f(val_set_latitude_f,longlen)
    new_val_set_longitude_f = padding_f(val_set_longitude_f,longlen)
    
    mask_train_x_f=generate_mask(train_set_user_f,longlen)
    mask_test_x_f=generate_mask(test_set_user_f,longlen)
    mask_val_x_f=generate_mask(val_set_user_f,longlen)
    
    final_train_set_f=(new_train_set_user_f,new_train_set_poi_f,new_train_set_time_f,new_train_set_latitude_f,new_train_set_longitude_f,mask_train_x_f)
    final_test_set_f=(new_test_set_user_f,new_test_set_poi_f,new_test_set_time_f,new_test_set_latitude_f,new_test_set_longitude_f,mask_test_x_f)
    final_val_set_f=(new_val_set_user_f,new_val_set_poi_f,new_val_set_time_f,new_val_set_latitude_f,new_val_set_longitude_f,mask_val_x_f)
    
    return final_train_set,final_test_set,final_val_set,final_train_set_f,final_test_set_f,final_val_set_f

def padding(x,y,max_len):

    new_x = np.zeros([len(x),max_len])
    new_y = np.zeros([len(y)])
    for i,(x_,y_) in enumerate(zip(x,y)):
        if len(x_)<=max_len:
            new_x[i,0:len(x_)]=x_
            new_y[i]=y_
        else:
            new_x[i]=x_[-max_len:]
            new_y[i]=y_
    new_set=new_x,new_y
    return new_set
    
def padding_f(x,max_len):

    new_x = np.zeros([len(x),max_len])
    for i,x_ in enumerate(x):
        if len(x_)<=max_len:
            new_x[i,0:len(x_)]=x_
        else:
            new_x[i]=x_[-max_len:]
    return new_x
    
def generate_mask(x,max_len):
    
    new_mask_x=np.zeros([len(x),max_len])
    for i,x_ in enumerate(x):     
        if len(x_)<=max_len:
            new_mask_x[i,:len(x_)]=1
        else: 
            new_mask_x[i,:]=1   
    return new_mask_x

def haversine(lonlat1, lonlat2):
    
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def generate_negative_sample(l,locations,num_sample,clusters,top_500): 
    
    city_id=int(locations[3,int(l)])  
    n_samples=len(clusters[city_id]) 
    ind=clusters[city_id].index(int(l))
    if n_samples>num_sample:
        name=list(range(n_samples))
        name.remove(ind)
        index=np.random.choice(name, num_sample, replace=False)
        lastindex=[clusters[city_id][i] for i in index]
    else:
        if n_samples>1:
            lastindex=list(set(clusters[city_id])-set([int(l)]))+top_500[0:num_sample-n_samples+1]
        else:
            if int(l) not in top_500:
                lastindex=top_500[0:num_sample]
            else:
                distance=[]
                lonlat1=(locations[1,int(l)],locations[2,int(l)])
                for i in range(locations.shape[1]):
                    if i!=int(l):
                        lonlat2=(locations[1,i],locations[2,i])
                        distance.append(haversine(lonlat1, lonlat2))
                max_index= math.argmin(distance)>=int(l) and math.argmin(distance)+1 or math.argmin(distance)
                lastindex=list(set(top_500[0:num_sample])-set([int(l)]))+[max_index]
    return lastindex
    
def padding_negative_sample(targets,locations,num_sample,clusters,top_500):
   
    suqence_num=len(targets)
    negative_sample=np.zeros([suqence_num,num_sample])
    for i in range(suqence_num):
        negative_sample[i,:]=np.mat(generate_negative_sample(targets[i],locations,num_sample,clusters,top_500)[0:num_sample])
    return negative_sample

def pop_n(userlocation,k): #选择最流行的top-k
    Locations_voc=collections.Counter([int (x) for x in userlocation[1,:].tolist()])
    sorted_Locations_voc=sorted(Locations_voc.items(), key=lambda d:d[1], reverse = True )
    return [a for i,(a,b) in enumerate(sorted_Locations_voc) if i<k]

def new_build_location_voc(locations): #构建用户和位置的字典
    
    clusters={}
    for i in range(locations.shape[1]):
        city_id = int(locations[3,i])
        if city_id in clusters.keys():
            clusters[city_id].append(int(locations[0,i]))
        else:
            clusters[city_id]=[int(locations[0,i])]
   
    return clusters
    
def time_encoding(inputs,init):
    gap=np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]) #20个等级
    inputs=[init-i for i in inputs]
    inputs=[i//600 for i in inputs]
    output=[np.sum(i>gap) for i in inputs]                             
    return output 
    
def network_trans(network,N):
    output=np.zeros([N,N])
    for i in range(N):
        output[i,i]=1
    for x in range(network.shape[0]):
        a=int(network[x,0])
        b=int(network[x,1])
        output[a,b]=1
        output[b,a]=1
    return output 
    
def batch_iter_sample(data,data_f,batch_size): 
    
    train_set_user,train_set_poi,train_set_time,train_set_latitude,train_set_longitude,mask_x,negative_set_poi=data  #导入原始训练数据
    new_train_set_user,new_train_set_poi,new_train_set_time,new_train_set_latitude,new_train_set_longitude,mask_train_x=data_f
    data_size=len(train_set_user[0])
    
    state=np.random.get_state()
    np.random.shuffle(train_set_user[0])
    
    for i in range(1,10):
        np.random.set_state(state)
        np.random.shuffle(data[int(i/2)][i%2])
    
    np.random.set_state(state)
    np.random.shuffle(negative_set_poi)
    
    np.random.set_state(state)
    np.random.shuffle(mask_x)

    for i in range(6):
        np.random.set_state(state)
        np.random.shuffle(data_f[i])
    
    print(data_size)
     
    num_batches_per_epoch=int(data_size/batch_size) #每一个epoch应该有多少批次，其实就是批次的数量
    
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        
        u=[user[0] for user in train_set_user[0][start_index:end_index]]
        l_u=new_train_set_user[start_index:end_index]
        
        s_x = train_set_poi[0][start_index:end_index]  
        l_x = new_train_set_poi[start_index:end_index] 
        y = train_set_poi[1][start_index:end_index]
        

        s_time_x = train_set_time[0][start_index:end_index]  
        l_time_x = new_train_set_time[start_index:end_index]
        time_y = train_set_time[1][start_index:end_index] 
     
        negative_sample = negative_set_poi[start_index:end_index]

        short_mask=mask_x[start_index:end_index]
        long_mask=mask_train_x[start_index:end_index]

        yield (u,l_u,s_x,l_x,y,s_time_x,l_time_x,time_y,negative_sample,short_mask,long_mask)  
        
def get_batches(pairs, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    result = []
    for idx in range(n_batches):
        x, y, t = [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
        result.append((np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32)))
    return result
               
def batch_iter(data,data_f,batch_size): 
    
    new_test_set_user,new_test_set_poi,new_test_set_time,new_test_set_latitude,new_test_set_longitude,mask_test_x=data  #导入原始训练数据
    new_test_set_user_f,new_test_set_poi_f,new_test_set_time_f,new_test_set_latitude_f,new_test_set_longitude_f,mask_test_x_f=data_f
    
    data_size=len(new_test_set_user[0])
     
    num_batches_per_epoch=int(data_size/batch_size) 
    
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        
        u=[user[0] for user in new_test_set_user[0][start_index:end_index]]
        l_u=new_test_set_user_f[start_index:end_index]
        
        s_x = new_test_set_poi[0][start_index:end_index]  
        l_x = new_test_set_poi_f[start_index:end_index]  
        y = new_test_set_poi[1][start_index:end_index]
           
        s_time_x = new_test_set_time[0][start_index:end_index] 
        l_time_x = new_test_set_time_f[start_index:end_index]
        time_y = new_test_set_time[1][start_index:end_index]  
        
        short_mask=mask_test_x[start_index:end_index]
        long_mask=mask_test_x_f[start_index:end_index]

        yield (u,l_u,s_x,l_x,y,s_time_x,l_time_x,time_y,short_mask,long_mask)  
           
  
