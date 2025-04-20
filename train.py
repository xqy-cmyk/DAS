import time
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_train_data_for_rnn,load_test_data_for_cnn, load_train_data_for_cnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn
from loss import  NaNMSELoss
from model import LSTMModel,CNN,ConvLSTMModel
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
#from cross_models.cross_former import Crossformer

def train(x,
          y,
          static,
          cluster_data,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True):
   
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split=cfg['valid_split']
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nf,nlat,nlon = x.shape  #x shape :nt,nf,nlat,nlon
	#Partition validation set and training set
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]       

    lossmse = torch.nn.MSELoss()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # mask see regions
    #Determine the land boundary
    if cfg['modelname'] in ['LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid, cluster_valid  = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, cluster_data, mask)
        x, y, static, dem = sea_mask_rnn(cfg, x, y, static, cluster_data, mask)
    elif cfg['modelname'] in ['CNN','ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['crossformer']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
#---------------------------------------k-means----------------------------------------------------
#    k = 3
#    print('dem are', dem.shape)
#    x_ = dem.reshape((dem.shape[1], 6))
#    model1 = MiniBatchKMeans(n_clusters=k)
#    model1.fit(x_)
#    print('the shape of x_ is:',x_.shape)
#    centers = model1.cluster_centers_
#    print('the centers is:',centers)
#    result = model1.predict(x_)
#    cluster_sample = []
#    cluster_num = []
#    for cluster_index in range(k):
#        mask = (result == cluster_index)
#        zero_positions = np.where(mask)
#        zero_positions = np.array(zero_positions)
#        row_indices = zero_positions[0]
#        cluster_sample.append(row_indices)
#        print(f"Cluster {cluster_index} - Number of zero_positions: {len(zero_positions[0])}")
#    cluster_num = np.array(cluster_num)
    
    
#--------------------------------------meanshift-------------------------------------------------
#    x_ = dem.reshape((dem.shape[1], 1)) 
#    model1 = MeanShift()
#    model1.fit(x_)
#    centers = model1.cluster_centers_ 
#    result = model1.predict(x_)   
#    cluster_sample = []  
#    small_cluster_samples = []  
#    large_cluster_samples = []     
#    for cluster_index in range(len(centers)):  
#        mask = (result == cluster_index)  
#        zero_positions = np.where(mask)  
#        row_indices = zero_positions[0]   
#        if len(row_indices) < 0:  
#            small_cluster_samples.extend(row_indices) 
#        else:  
#            large_cluster_samples.append(row_indices) 
#        print(f"Cluster {cluster_index} - Number of samples: {len(row_indices)}")
#    if len(small_cluster_samples) > 0:
#        cluster_sample.append(small_cluster_samples)  
#        print(f"Created new small cluster with {len(small_cluster_samples)} samples.")
#    cluster_sample.extend(large_cluster_samples)
#    print("Final Cluster sample numbers:", [len(samples) for samples in cluster_sample])
#    k = len(cluster_sample)
#    print("Total number of clusters (k):", k)
#    cluster_sample = np.array(cluster_sample, dtype=object)



    k = 3
    print('dem are', dem.shape)
    x_ = dem.reshape((dem.shape[1], 6))
    model1 = MiniBatchKMeans(n_clusters=k)
    model1.fit(x_)
    print('the shape of x_ is:',x_.shape)
    centers = model1.cluster_centers_
    print('the centers is:',centers)
    result = model1.predict(x_)
    cluster_sample = []

    
    
#    print('x shape isis:',x.shape)
#    print('y shape isis:', y.shape)
#    k = 3
#    
#    nan_mask = ~np.isnan(dem).any(axis=0)  
#    invalid_mask = nan_mask 
#    print('invalid mask shape:',invalid_mask.shape)
#    x[:, :, invalid_mask]  = np.nan
#    y[:, invalid_mask] = np.nan
#    x_ = dem[:, nan_mask].T 
#    #x_ = dem.reshape((dem.shape[1], 6))
#    model1 = MiniBatchKMeans(n_clusters=k)
#    model1.fit(x_)
#  
#    centers = model1.cluster_centers_
#    result = model1.predict(x_)
#
#    print(f"Valid samples after NaN removal: {np.sum(nan_mask)} / {len(nan_mask)}")
#    print('the shape of x is:', x.shape)
#    print('the shape of y is:', y.shape)
    index = np.arange(x_.shape[0])  
    print('The shape of index is:', index.shape)
  
  

#    
#    
#    x1_max = np.max(x_[result==0])
#    x1_min = np.min(x_[result == 0])
#    x2_max = np.max(x_[result==1])
#    x2_min = np.min(x_[result == 1])
#    x3_max = np.max(x_[result==2])
#    x3_min = np.min(x_[result == 2])
#    print('x1_max:', x1_max)
#    print('x1_min:', x1_min)
#    print('x2_max:', x2_max)
#    print('x2_min:', x2_min)
#    print('x3_max:', x3_max)
#    print('x3_min:', x3_min)
    # x4_max = np.max(x_[result==3])
    # x4_min = np.min(x_[result == 3])
    #DEM4 = cluster_data
    #DEM4[x1_max > DEM4 > x1_min] = 1
    #DEM4[x2_max > DEM4 > x2_min] = 2
    #DEM4[x3_max > DEM4 > x3_min] = 3
    # DEM4[x4_max > DEM4 > x4_min] = 4
    #dem_path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution'])
    #np.save(dem_path + '/DEM4.npy', DEM4)
    
    
    

    cluster_sample = []
    uncertainties = []
    for cluster_index in range(k):
        mask = (result == cluster_index)
        real_index = index[mask]
        real_index = np.array(real_index)
        cluster_sample.append(real_index)

        # cluster_distances = np.abs(x_[mask] - centers[cluster_index])
        cluster_distances = np.linalg.norm(x_[mask] - centers[cluster_index], axis=1)
        cluster_uncertainties = cluster_distances.flatten()  
        uncertainties.append(cluster_uncertainties)

        sorted_indices = np.argsort(-cluster_uncertainties) 
        sorted_samples = real_index[sorted_indices]
        print(f"Cluster {cluster_index} - Number of samples: {len(cluster_sample[cluster_index])}")
        print(f"Cluster {cluster_index} - Top 5 uncertainties: {cluster_uncertainties[sorted_indices[:5]]}")
        print(f"Cluster {cluster_index} - Top 5 sorted samples indices: {sorted_samples[:5]}")

    sorted_cluster_samples = [real_index[np.argsort(-uncertainties[i])] for i, real_index in enumerate(cluster_sample)]
    grouped_cluster_samples = []
    for cluster_index, samples in enumerate(sorted_cluster_samples):
        group_size = len(samples) // 3
        groups = [
            samples[:group_size], 
            samples[group_size:2 * group_size],  
            samples[2 * group_size:]  
        ]
        grouped_cluster_samples.append(groups)
        
        
        

    for num_ in range(cfg['num_repeat']):
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
            model = LSTMModel(cfg,lstmmodel_cfg).to(device)
        elif cfg['modelname'] in ['crossformer']:
            model = Crossformer(cfg["input_size"],
                                cfg['seq_len'],1, 14,2,10,256,512,4,\
                                3,0.2,False,device).to(device)
        elif cfg['modelname'] in ['CNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)


        optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])

        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):
 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _ ,cluster_sample = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y, grouped_cluster_samples, k)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        pred = torch.squeeze(pred,1)
                    elif cfg['modelname'] in ['crossformer']:
                         # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch)
                        pred = torch.squeeze(pred,1)
 #  train way for CNN model
                    elif cfg['modelname'] in ['CNN']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index)
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        x_batch = x_batch.squeeze(dim=1)
                        x_batch = x_batch.reshape(x_batch.shape[0],x_batch.shape[1]*x_batch.shape[2],x_batch.shape[3],x_batch.shape[4])
                        x_batch = torch.cat([x_batch, aux_batch], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['ConvLSTM']:
                        # generate batch data for Convolutional LSTM
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index) # same as Convolutional Neural Network
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1,1,1)
                        x_batch = x_batch.squeeze(dim=1)
                        pred = model(x_batch, aux_batch,cfg)
 # ------------------------------------------------------------------------------------------------------------------------------
                    loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(),lossmse)                              
                    optim.zero_grad()
                    loss.backward()
                    optim.step()                    
                    MSELoss += loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin)
                print(loss_str)
                # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    del x_batch, y_batch, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)                              
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for CNN model
                        elif cfg['modelname'] in ['CNN']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch.squeeze(1)
                                x_valid_batch = x_valid_batch.reshape(x_valid_batch.shape[0],x_valid_batch.shape[1]*x_valid_batch.shape[2],x_valid_batch.shape[3],x_valid_batch.shape[4])
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], axis=1)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for ConvLSTM model，same as CNN model
                        elif cfg['modelname'] in ['ConvLSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1,1,1)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch,cfg)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
             

                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss/(len(gt_list))
                        # get loss log
                        loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss, 
                                    t_end-t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best  :
                        best = MSELoss
                        wait = 0
                        #if epoch <=300:
                        #    torch.save(model,out_path+'/300/'+cfg['modelname']+'_para.pkl')
                        #if epoch >300 and epoch<=500:
                       #     torch.save(model,out_path+'/500/'+cfg['modelname']+'_para.pkl')                  
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    return
            return


