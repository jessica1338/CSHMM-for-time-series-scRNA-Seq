import numpy as np
import argparse
from cvxpy import *
#import random
import progressbar
from collections import defaultdict
from scipy.stats import spearmanr
import time
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans,SpectralClustering
from scipy.spatial import distance
from numpy import inf
import pickle
import multiprocessing as mp
import sys
def load_data_2(file_name,max_gene):
    print 'loading data......'
    lines=open(file_name).readlines()
    #print lines
    head=''
    cell_names=lines[0].replace('\n','').split('\t')[1:-1]
    cell_times=np.array(map(int,map(float,lines[1].replace('\n','').split('\t')[1:-1])))
    cell_labels=[]
    gene_exps=[]
    gene_names=[]
    for i,name in enumerate(cell_names):
        splits=name.split('_')
        cell_lab = splits[1]
        if 'Day' in cell_lab:
            cell_lab='NA'
        cell_labels.append(cell_lab)
    for line in lines[2:]:
        line=line.replace('\n','')
        splits=line.split('\t')[:-1]
        gene_names.append(splits[0])
        gene_exp=map(float,splits[1:])
        gene_exps.append(gene_exp)
    cell_exps=np.transpose(np.array(gene_exps))
    gene_names=np.array(gene_names)
    rm_col=np.all(cell_exps<0.1,axis=0)#remove all < 0.1 genes
    n_cell,n_gene = cell_exps.shape
    for j in range(n_gene):
        if np.count_nonzero(cell_exps[:,j])<n_cell/4: #remove the gene that express in less than 25% of cells
            rm_col[j]=True
    cell_exps=cell_exps[:,~rm_col]
    gene_names=gene_names[~rm_col]
    cell_exps=np.log2(cell_exps+1)
    n_cell,n_gene = cell_exps.shape
    if n_gene>max_gene: #select the top [max_gene] genes by variance
        cell_exps_var=np.var(cell_exps,axis=0)
        sort_index = np.argsort(-cell_exps_var)
        select_gene=sort_index[:max_gene]
        cell_exps=cell_exps[:,select_gene]
        gene_names=gene_names[select_gene]
    n_cell,n_gene = cell_exps.shape
    print n_cell, ' cell loaded with ',n_gene,' selected'
    print np.unique(np.array(cell_labels),return_counts=True)
    return cell_names,cell_times,cell_labels,cell_exps,gene_names
def load_data(file_name,max_gene):
    print 'loading data......'
    lines=open(file_name).readlines()
    head=lines[0].replace('\n','')
    cell_names=[]
    cell_day=[]
    cell_labels=[]
    cell_exps=[]
    gene_names=np.array(head.split('\t')[3:])
    for line in lines[1:]:
        line=line.replace('\n','')
        splits=line.split('\t')
        cell_name=splits[0]
        day=int(splits[1])
        label=splits[2]
        gene_exp=splits[3:]
        cell_names.append(cell_name)
        cell_day.append(day)
        cell_labels.append(label)
        cell_exps.append(map(float,gene_exp))
    cell_exps=np.array(cell_exps)
    n_cell,n_gene = cell_exps.shape
    if n_gene>max_gene: #select the top [max_gene] genes by variance
        cell_exps_var=np.var(cell_exps,axis=0)
        sort_index = np.argsort(-cell_exps_var)
        select_gene=sort_index[:max_gene]
        cell_exps=cell_exps[:,select_gene]
        gene_names=gene_names[select_gene]
    n_cell,n_gene = cell_exps.shape
    n_cell,n_gene = cell_exps.shape
    print n_cell, ' cell loaded with ',n_gene,' genes selected'
    return cell_names,cell_day,cell_labels,cell_exps,gene_names
def init_var_Jun(init_file,cell_names,cell_times,cell_exps,cell_labels):
    print 'initializing parameters and hidden variable with Juns model structure......'
    st_line=open(init_file).readlines()[0].replace('\n','').split('\t')
    c_line=open(init_file).readlines()[1].replace('\n','').split('\t')
    n_path=len(st_line)+1
    n_state=n_path+1
    n_cell,n_gene = cell_exps.shape
    path_info=[]
    adj_mat=np.zeros((n_state,n_state))
    adj_mat[0,1]=1
    for i in range(n_path):
        path_info.append(defaultdict(lambda:[]))
    path_info[0]['Sp_idx']=0
    path_info[0]['level']=0
    for i in range(n_path):
        path_info[i]['Sc_idx']=i+1
        path_info[i]['ID']=i
    for line in st_line:
        pa,pb = map(int,line.split(' '))
        path_info[pa]['child_path'].append(pb)
        path_info[pb]['parent_path']=pa
        path_info[pb]['Sp_idx']=path_info[pa]['Sc_idx']
        path_info[pb]['level']=path_info[pa]['level']+1
        adj_mat[path_info[pb]['Sp_idx'],pb+1]=1
    for i in range(n_state):
        adj_sum=np.sum(adj_mat[i])
        if adj_sum>0:
            adj_mat[i,:]/=adj_sum
    g_param=np.zeros((n_state,n_gene))
    sigma_param=np.ones(n_gene)
    K_param=np.random.sample((n_path,n_gene))*K_param_range
    A=adj_mat
    cell_path=np.zeros(n_cell,dtype=int)
    #cell_names=cell_names.tolist()
    print cell_names
    for line in c_line:
        cn,p=line.split(' ')
        p=int(p)
        if cn in cell_names:
            cell_path[cell_names.index(cn)]=p
    cell_time=np.random.sample((n_cell,))
    model={}
    model['g_param']=g_param
    model['sigma_param']=sigma_param
    model['K_param']=K_param
    model['trans_mat']=adj_mat
    model['path_info']=path_info
    hid_var={}
    hid_var['cell_time']=cell_time
    hid_var['cell_ori_time']=cell_times
    hid_var['cell_path']=cell_path
    hid_var['cell_labels']=np.array(cell_labels)
    optimize_w_nz(model,hid_var,cell_exps)
    path_trans_prob=compute_path_trans_log_prob(adj_mat,path_info)
    model['path_trans_prob']=path_trans_prob
    return model,hid_var
def save_model(file_name,model,hid_var):
    print 'saving model to file: ',file_name
    with open(file_name, 'wb') as handle:
        out_dict={}
        out_dict['model']=model
        out_dict['hid_var']=hid_var
        pickle.dump(out_dict, handle)
def load_model(file_name):
    print 'loading model from file: ',file_name
    with open(file_name, 'rb') as handle:
        out_dict = pickle.load(handle)
    return out_dict['model'],out_dict['hid_var']
def optimize_w_nz(model,hid_var,cell_exps):
    print 'M-step: optimizing w param......'
    path_info=model['path_info']
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    K_param=model['K_param']
    g_param=model['g_param']
    n_cell,n_gene=cell_exps.shape
    n_state=g_param.shape[0]
    n_path=n_state-1
    sigma_param=model['sigma_param']
    w_nz=np.ones((n_path,n_gene))  #non zero ratio for each gene in each path (wpj)
    if optimize_w:
        w_nz=np.zeros((n_path,n_gene))  #non zero ratio for each gene in each path (wpj)
        if progress_bar:
            bar = progressbar.ProgressBar(maxval=n_path*n_gene, \
                    widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' (', progressbar.ETA(), ') '] )
            bar.start()
        w_split=n_split
        path_gene_w_table=np.zeros((n_path,n_gene,w_split))
        for p in range(n_path):
            Sp_idx=path_info[p]['Sp_idx']
            Sc_idx=path_info[p]['Sc_idx']
            g_a=g_param[Sp_idx]
            g_b=g_param[Sc_idx]
            p_idx=(cell_path==p)
            cell_exps_p=cell_exps[p_idx]
            cell_time_p=cell_time[p_idx]
            for j in range(n_gene):
                x_js=cell_exps_p[:,j]
                mu_x_js=g_b[j]+(g_a[j]-g_b[j])*np.exp(-K_param[p,j]*cell_time_p)
                tmp=(x_js-mu_x_js)**2./(2.*sigma_param[j]**2.)
                prob2 = np.where(x_js!=0.,0.,drop_out_param)
                prob1= np.exp(-tmp)/(sigma_param[j])/np.sqrt(2.*np.pi)
                for ws in range(1,w_split+1):
                    w=1/float(w_split)*ws
                    mix_prob=w*prob1+(1-w)*prob2
                    sum_log_prob=np.sum(np.log(mix_prob))
                    path_gene_w_table[p,j,ws-1]=sum_log_prob
                max_ws=np.argmax(path_gene_w_table[p,j,:])+1
                max_w=1/float(w_split)*max_ws
                w_nz[p,j]=max_w
                #print 'max_w: ',max_w
                if progress_bar:
                    bar.update(p*n_gene+j+1)
        if progress_bar:
            bar.finish()
    model['w_nz']=w_nz
def compute_path_trans_log_prob(trans_mat,path_info):
    ret=[]
    for i,p in enumerate(path_info):
        mult = 1
        now=p
        while(True):
            Sp=now['Sp_idx']
            Sc=now['Sc_idx']
            if Sp==0:
                break
            mult*=trans_mat[Sp,Sc]
            now=path_info[now['parent_path']]
        ret.append(mult)
    with np.errstate(divide='ignore'):
        ret=np.log(np.array(ret))
    return ret
def calc_cell_exp_prob(p,t,model,x_i):
    path_info=model['path_info']
    path_trans_prob=model['path_trans_prob']
    g_param=model['g_param']
    sigma_param=model['sigma_param']
    K_param=model['K_param']
    w_nz=model['w_nz']
    Sp_idx=path_info[p]['Sp_idx']
    Sc_idx=path_info[p]['Sc_idx']
    g_a=g_param[Sp_idx]
    g_b=g_param[Sc_idx]
    mu_x_i=g_b+(g_a-g_b)*np.exp(-K_param[p]*t)
    tmp=(x_i-mu_x_i)**2./(2.*sigma_param**2.)+np.log((sigma_param*np.sqrt(2.*np.pi)) )
    prob2 = np.where(x_i!=0.,0.,drop_out_param)
    mix_prob=w_nz[p]*np.exp(-tmp)+(1-w_nz[p])*prob2
    log_mix_prob=np.log(mix_prob)
    ret=np.sum(log_mix_prob)+path_trans_prob[p]
    return ret
def log_likelihood(model,hid_var,cell_exps):
    ret=0.
    path_info=model['path_info']
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    g_param=model['g_param']
    K_param=model['K_param']
    n_state,n_gene = g_param.shape
    n_path=n_state-1
    n_cell=cell_exps.shape[0]
    for i in range(n_path):
        s_a=path_info[i]['Sp_idx']
        s_b=path_info[i]['Sc_idx']
        delta_g=g_param[s_a]-g_param[s_b]
        ret+=-lamb*np.sum(np.fabs(delta_g))
    for i in range(n_cell):
        p=cell_path[i]
        t=cell_time[i]
        x_i=cell_exps[i,:]
        ret+=calc_cell_exp_prob(p,t,model,x_i)
    return ret
def model_score(model,hid_var,cell_exps,method):
    print 'calculating ',method,' score......'
    n_cell=cell_exps.shape[0]
    g_param=model['g_param']
    n_state,n_gene = g_param.shape
    k=n_gene * n_state * 3 - n_gene # g_param: G*S, K_param: G*P = G*(S-1), sigma_param: G, w_nz: G*(S-1)
    if not optimize_w:
        k=n_gene * n_state * 2 # g_param: G*S, K_param: G*P = G*(S-1), sigma_param: G, w_nz: G*(S-1)
    ll2= 2*log_likelihood(model,hid_var,cell_exps) 
    BIC_score =  ll2 - np.log(n_cell)*k
    AIC_score =  ll2 - 2*k
    GIC2_score = ll2 - k**(1/3.)*k
    GIC3_score = ll2 - 2*np.log(k)*k
    GIC4_score = ll2 - 2*(np.log(k)+np.log(np.log(k)))*k
    GIC5_score = ll2 - np.log(np.log(n_cell))*np.log(k)*k
    GIC6_score = ll2 - np.log(n_cell)*np.log(k)*k
    if method=='BIC':
        return 'BIC= ', BIC_score
    if method=='AIC':
        return 'AIC= ', AIC_score
    if method=='ALL':
        return '(AIC,BIC,G2,G3,G4,G5,G6)=', (AIC_score,BIC_score,GIC2_score,GIC3_score,GIC4_score,GIC5_score,GIC6_score)

def optimize_transition_prob(model,hid_var):
    trans_mat=model['trans_mat']
    path_info=model['path_info']
    cell_path=hid_var['cell_path']
    new_trans_mat=np.zeros(trans_mat.shape)
    for i in range(cell_path.shape[0]):
        p=cell_path[i]
        Sp=path_info[p]['Sp_idx']
        Sc=path_info[p]['Sc_idx']
        new_trans_mat[Sp,Sc]+=1
    sum_vector=np.sum(new_trans_mat,axis=1)
    for i in range(new_trans_mat.shape[0]):
        if sum_vector[i]>0:
            new_trans_mat[i,:]/=sum_vector[i]
    model['trans_mat']=new_trans_mat
    path_trans_prob=compute_path_trans_log_prob(new_trans_mat,path_info)
    model['path_trans_prob']=path_trans_prob
    return
def assign_path_and_time(model,hid_var,cell_exps):
    print 'E-step: assigning new path and time for cell......'
    n_path=model['K_param'].shape[0]
    n_cell=cell_exps.shape[0]
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=n_cell, \
                widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' (', progressbar.ETA(), ') '] )
        bar.start()
    time_split=n_split
    path_time_table=np.zeros((n_path,time_split+1))
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    cell_ori_time=hid_var['cell_ori_time']
    if n_anchor:
        anchor=defaultdict(lambda:defaultdict(lambda:[]))
        for i in range(n_cell):
            p=cell_path[i]
            t=cell_time[i]
            prob=calc_cell_exp_prob(p,t,model,cell_exps[i,:])
            anchor[p]['cell_index'].append(i)
            anchor[p]['cell_prob'].append(prob)
        anchor_cell=np.array([-1])
        for p in range(n_path):
            cell_index=np.array(anchor[p]['cell_index'])
            cell_prob=np.array(anchor[p]['cell_prob'])
            anchor_p= cell_index[np.argsort(-cell_prob)[:n_anchor]]
            anchor_cell=np.union1d(anchor_cell,anchor_p)
        print 'anchor cell: ', anchor_cell
    for i in range(n_cell):
        if progress_bar:
            bar.update(i+1)
        if n_anchor and i in anchor_cell:
            continue
        for p in range(n_path):
            for t_sp in range(time_split+1):
                t=t_sp/float(time_split)
                path_time_table[p,t_sp]=calc_cell_exp_prob(p,t,model,cell_exps[i,:])
        max_time=np.argmax(path_time_table,axis=1) #max_time for every path
        max_prob=np.max(path_time_table,axis=1) #prob of every path with max_time 
        new_path = np.argmax(max_prob)
        ori_prob= np.exp(max_prob-np.max(max_prob))
        norm_prob=ori_prob/np.sum(ori_prob)
        valid_idx=np.array(range(n_path))
        sample_prob=norm_prob[valid_idx]/np.sum(norm_prob[valid_idx])
        sample = np.random.multinomial(1,sample_prob)
        for index,s in enumerate(sample):
            if s==1:
                sampled_path=valid_idx[index]
                break
        new_path=valid_idx[np.argmax(max_prob[valid_idx])]
        new_time=max_time[new_path]/float(time_split)
        cell_path[i]=new_path
        if assign_by_prob_sampling:
            cell_path[i]=sampled_path
        cell_time[i]=new_time
    hid_var['cell_time']=cell_time
    hid_var['cell_path']=cell_path
    if progress_bar:
        bar.finish()
    return    
def optimize_K_param(model,hid_var,cell_exps):
    print 'M-step: optimizing K param......'
    K_param=model['K_param']
    new_K_param=np.zeros(K_param.shape)
    w_nz=model['w_nz']
    n_path,n_gene=K_param.shape
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    path_info=model['path_info']
    g_param=model['g_param']
    sigma_param=model['sigma_param']
    k_split=n_split
    path_gene_k_table=np.zeros((n_path,n_gene,k_split))
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=n_path*n_gene, \
                widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' (', progressbar.ETA(), ') '] )
        bar.start()
    count=0
    for p in range(n_path):
        Sp_idx=path_info[p]['Sp_idx']
        Sc_idx=path_info[p]['Sc_idx']
        g_a=g_param[Sp_idx]
        g_b=g_param[Sc_idx]
        p_idx=(cell_path==p)
        cell_exps_p=cell_exps[p_idx]
        cell_time_p=cell_time[p_idx]
        for j in range(n_gene):
            x_js=cell_exps_p[:,j]
            for ks in range(1,k_split+1):
                k=K_param_range/float(k_split)*ks
                mu_x_js=g_b[j]+(g_a[j]-g_b[j])*np.exp(-k*cell_time_p)
                tmp=((x_js-mu_x_js)**2./(2.*sigma_param[j]**2.)+np.log((sigma_param[j]*np.sqrt(2.*np.pi)) ))
                prob2 = np.where(x_js!=0.,0.,drop_out_param)
                mix_prob=w_nz[p,j]*np.exp(-tmp)+(1-w_nz[p,j])*prob2
                sum_log_prob=np.sum(np.log(mix_prob))
                path_gene_k_table[p,j,ks-1]=sum_log_prob
            max_ks=np.argmax(path_gene_k_table[p,j,:])+1
            max_k=K_param_range/float(k_split)*max_ks
            K_param[p,j]=max_k
            count+=1
            if progress_bar:
                bar.update(count)
    if progress_bar:
        bar.finish()
def optimize_sigma_param(model,hid_var,cell_exps):
    print 'M-step: optimizing sigma param......'
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    path_info=model['path_info']
    g_param=model['g_param']
    n_cell,n_gene=cell_exps.shape
    new_sigma_param=np.zeros(n_gene)
    K_param=model['K_param']
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=n_gene, \
                widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' (', progressbar.ETA(), ') '] )
        bar.start()
    for i in range(n_cell):
        p=cell_path[i]
        Sp_idx=path_info[p]['Sp_idx']
        Sc_idx=path_info[p]['Sc_idx']
        g_a=g_param[Sp_idx]
        g_b=g_param[Sc_idx]
        t=cell_time[i]
        x_i=cell_exps[i]
        mu_x_i=g_b+(g_a-g_b)*np.exp(-K_param[p]*t)
        new_sigma_param+=(x_i-mu_x_i)**2
        if progress_bar:
            bar.update(i)
    new_sigma_param=(new_sigma_param/float(n_cell))**0.5
    new_sigma_param=np.where(new_sigma_param<1,1,new_sigma_param)
    model['sigma_param']=new_sigma_param    
    if progress_bar:
        bar.finish()

def optimize_g_param_cvx(model,hid_var,cell_exps):
    print 'M-step: optimizing g param with CVX......'
    path_info=model['path_info']
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    K_param=model['K_param']
    g_param=model['g_param']
    n_cell,n_gene=cell_exps.shape
    n_state=g_param.shape[0]
    sigma_param=model['sigma_param']
    A2=np.zeros((n_state-1,n_state))
    _,path_count=np.unique(hid_var['cell_path'],return_counts=True)
    path_nz_g=check_diff_gene(model)
    for index in range(n_state-1):
        path=path_info[index]
        Sp_idx=path['Sp_idx']
        Sc_idx=path['Sc_idx']
        A2[index,Sp_idx]=1
        A2[index,Sc_idx]=-1
        if lamb_data_mult=='N':
            A2[index,:]*=path_count[index] # multiply by N
        if lamb_data_mult=='sqrtN':
            A2[index,:]*=np.sqrt(path_count[index]) # multiply by sqrt(N)
        if lamb_data_mult=='logN':
            A2[index,:]*=np.log(path_count[index]) # multiply by log(N)
        if lamb_ratio_mult=='sqrtR':
            A2[index,:]*=np.sqrt(path_nz_g[index])
        if lamb_ratio_mult=='R':
            A2[index,:]*=path_nz_g[index]
    A2*=lamb
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=n_gene, \
                widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' (', progressbar.ETA(), ') '] )
        bar.start()
    for j in range(n_gene):
        A1=np.zeros((n_cell,n_state))
        Xjs=np.zeros(n_cell)
        sigma_j=sigma_param[j]
        for i in range(n_cell):
            p=cell_path[i]
            Sp_idx=path_info[p]['Sp_idx']
            Sc_idx=path_info[p]['Sc_idx']
            t=cell_time[i]
            x_ij=cell_exps[i,j]
            w_ij=np.exp(-K_param[p,j]*t)
            A1[i,Sp_idx]=w_ij
            A1[i,Sc_idx]=1-w_ij
            Xjs[i]=x_ij
        g_js = Variable(n_state)
        objective = Minimize(sum_squares(0.5*(A1*g_js-Xjs)/sigma_j)+pnorm(A2*g_js,1))
        constraints=[]
        prob = Problem(objective, constraints)
        result = prob.solve(solver=SCS)
        g_param[:,j]=g_js.value.flatten()
        if progress_bar:
            bar.update(j)
    if progress_bar:
        bar.finish()
def check_diff_gene(model):
    path_info=model['path_info']
    g_param=model['g_param']
    n_gene=g_param.shape[1]
    path_nz_g={}
    for index,path in enumerate(path_info):
        Sp_idx=path['Sp_idx']
        Sc_idx=path['Sc_idx']
        g_a=g_param[Sp_idx]
        g_b=g_param[Sc_idx]
        g_abs_diff=np.fabs(g_a-g_b)
        g_abs_diff_nz=np.where(g_abs_diff<1e-1,0,g_abs_diff)
        nz_count= len(g_abs_diff_nz[np.nonzero(g_abs_diff_nz)])
        nz_ratio = nz_count/float(n_gene)
        if verbose:
            print 'path: ', index, ' nz_ratio: ', nz_ratio
        path_nz_g[index]=nz_ratio
    return path_nz_g

def show_cell_time(hid_var):
    cell_path = hid_var['cell_path']
    cell_labels = hid_var['cell_labels']
    cell_time=hid_var['cell_time']
    cell_ori_time=np.array(hid_var['cell_ori_time'])
    paths=np.unique(cell_path)
    for p in paths:
        print '----------path: ',p,'-------------'
        print 'time\tlabel\t assigned_time'
        p_idx=(cell_path==p)
        cell_time_p=np.around(cell_time[p_idx],decimals=2)
        cell_labels_p=cell_labels[p_idx]
        cell_ori_time_p=cell_ori_time[p_idx]
        sort_idx=np.argsort(cell_time_p)
        for lab,ori_t,t in zip(cell_labels_p[sort_idx],cell_ori_time_p[sort_idx], cell_time_p[sort_idx]):
            print ori_t,'\t', lab, '     \t' , t
def compute_ARI_confuss_mat(hid_var,n_path):
    cell_path = hid_var['cell_path']
    cell_labels = hid_var['cell_labels']
    unique_paths=np.unique(cell_path)
    unique_labels={}
    cell_label_num=[]
    ARI_ans=[]
    ARI_pred=[]
    head=[]
    for i,lab in enumerate(cell_labels):
        if lab in unique_labels.keys():
            cell_label_num.append(unique_labels[lab])
        else:
            ID=len(unique_labels.keys())
            unique_labels[lab]=ID
            cell_label_num.append(unique_labels[lab])
            head.append(lab)
        if lab!='NA':
            ARI_ans.append(unique_labels[lab])
            ARI_pred.append(cell_path[i])
    confuss_mat=np.zeros((n_path,len(unique_labels)))
    for i,num in enumerate(cell_label_num):
        confuss_mat[cell_path[i],num]+=1
    print 'confussion matrix:'
    print head
    print confuss_mat
    ARI= adjusted_rand_score(ARI_ans, ARI_pred)
    print 'ARI: ',ARI
    return confuss_mat,ARI
def load_adj_mat(file_name):
    return np.load(file_name)

def path_distance(pa,pb,cell_exps,cell_path):
    pa_center = np.average(cell_exps[cell_path==pa],axis=0)
    pb_center = np.average(cell_exps[cell_path==pb],axis=0)
    return 1-spearmanr(pa_center,pb_center)[0]


def adjust_model_structure(model,hid_var,cell_exps):
    print 'adjusting model structure '
    path_info=model['path_info']
    cell_path=hid_var['cell_path']
    cell_time=hid_var['cell_time']
    n_path=len(path_info)
    valid_parent=sorted(list(set(np.unique(hid_var['cell_path']).tolist())))
    level_path=defaultdict(lambda:[])
    for i,p in enumerate(path_info):
        p['child_path']=[]
        if i not in valid_parent:
            print 'zero path: ', i
            continue
        level_path[p['level']].append(i)
    def getKey(item):
        return item['level']
    childs = sorted([x for x in path_info], key = getKey)
    for p in childs:
        ID=p['ID']
        if ID not in valid_parent or ID ==0:
            continue
        level= p['level']
        new_parent_list=level_path[level-1]
        if len(new_parent_list)==1:
            new_parent = new_parent_list[0]
            print str(new_parent) + ' -> '+str(ID)
        else:
            par_distance = []
            for par in new_parent_list:
                p['Sp_idx']=path_info[par]['Sc_idx']
                p_idx=(cell_path==ID)
                cell_exps_p=cell_exps[p_idx]
                cell_time_p=cell_time[p_idx]
                s=0
                for i in range(cell_exps_p.shape[0]):
                    s+=calc_cell_exp_prob(ID,cell_time_p[i],model,cell_exps_p[i,:])
                par_distance.append(s)
            #par_distance = [path_distance(ID,x,cell_exps,cell_path) for x in new_parent_list]
            new_parent = new_parent_list[np.argmax(par_distance)]
            print str(new_parent) + ' -> '+str(ID)
        p['parent_path']=new_parent
        new_p=path_info[new_parent]
        p['Sp_idx']=new_p['Sc_idx']
        new_p['child_path'].append(ID)
        p['level']=new_p['level']+1

def optimize_likelihood(cell_exps, model, hid_var, model_name,store_model=True):
    for out_it in range(1,n_iteration+1):
        prev_path=np.array(hid_var['cell_path'],copy=True)
        print 'training iteration: ', out_it
        
        sys.stdout.flush()
        
        print 'cell paths: ',np.unique(hid_var['cell_path'],return_counts=True)

        score = model_score(model,hid_var,cell_exps,method='ALL')
        print 'model score: ',score
        
        optimize_g_param_cvx(model,hid_var,cell_exps)
        print 'after M-step g_param full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
       
        sys.stdout.flush()
        #check_diff_gene(model)
        
        optimize_sigma_param(model,hid_var,cell_exps)
        print 'after M-step sigma_param full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        optimize_K_param(model,hid_var,cell_exps)
        print 'after M-step K_param full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        
        #assign_path_and_time(model,hid_var,cell_exps)
        assign_path_and_time(model,hid_var,cell_exps)
        print 'after E-step full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        adjust_model_structure(model,hid_var,cell_exps)
        
        sys.stdout.flush()
        optimize_w_nz(model,hid_var,cell_exps)
        print 'after setting w_nz full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        
        optimize_transition_prob(model,hid_var)
        #print 'opt trans mat full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        path_trans_prob=compute_path_trans_log_prob(model['trans_mat'],model['path_info'])
        model['path_trans_prob']=path_trans_prob
        print 'after setting trans_prob full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        n_path=len(model['path_info'])
        compute_ARI_confuss_mat(hid_var,n_path)
        if verbose:
            show_cell_time(hid_var)
        if np.array_equal(prev_path,hid_var['cell_path']):
            print 'path assignment the same as previous iteration, stop training.'
        #if out_it % 10 ==0:
        if store_model:
            save_model(model_name+'_it'+str(out_it)+'.pickle',model, hid_var)
            #model,hid_var=load_model(model_name)
           
        sys.stdout.flush()
    print 'maximum training iteration reached.'
    #print 'after M-step g_param full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
    
    #optimize_g_param_close(model,hid_var,cell_exps)
    #print 'after M-step g_param full log-likelihood (close): ',log_likelihood(model,hid_var,cell_exps)
def cv_split_idx(cell_day,n_fold=5):
    cell_day = np.array(cell_day)
    unique_day=np.unique(cell_day)
    n_cell=cell_day.shape[0]
    batch=n_cell/n_fold
    fold_idx=np.zeros(n_cell)
    cell_day_dict={}
    for ud in unique_day:
        ud_idx=np.where(cell_day==ud)[0]
        ud_count=ud_idx.shape[0]
        np.random.shuffle(ud_idx)
        batch=ud_count/n_fold
        for i in range(n_fold):
            fold_idx[ud_idx[batch*i:batch*(i+1)]]=i+1
        if batch*(i+1)<ud_count:
            fold_idx[ud_idx[batch*i:]]=i+1
    return np.array(fold_idx,dtype=int)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--data_file", help="specify the data file, if not specified then a default training data file will be used")
    parser.add_argument('-dt',"--data_file_testing", help="specify the testing data file and output best interation for testing, if not specifed then the model will not do testing.", default = None)
    parser.add_argument('-st',"--structure_file", help="specify the structure file, if not specified then a default structure file will be used")
    parser.add_argument('-seed',"--random_seed", help="specify the random seed, default is 0", type=int,default=0) 
    parser.add_argument('-ni',"--n_iteration", help="specify the number of training iteration, default is 10", type=int,default=10)
    parser.add_argument('-k',"--k_param_range", help="specify the range of K parameter, default is 5", type=int,default=5)
    parser.add_argument('-ns',"--n_split", help="specify the number of splits in learning K and assign cell time, default is 10", type=int,default=10)
    parser.add_argument('-na',"--n_anchor", help="specify the number of anchor cells to remain in each path during training, default is 0", type=int,default=0)
    parser.add_argument('-ng',"--n_gene", help="specify the maximum number of genes used in training, default is 1000", type=int,default=1000)
    parser.add_argument('-lamb',"--lamb", help="specify the regularizing parameter for L1 sparsity, default is 1", type=float,default=1)
    #parser.add_argument('-dop',"--drop_out_param", help="specify the drop-out parameter, default is 0.1", type=float,default=0.1, help=argparse.SUPPRESS)
    parser.add_argument('-dop',"--drop_out_param", type=float,default=0.1, help=argparse.SUPPRESS)
    parser.add_argument('-ps',"--assign_by_prob_sampling", help="specify the whether to use multinomial sampling in path assignment, default is 1", type=int,choices=[0,1],default=1)
    #parser.add_argument('-opt_w',"--optimize_w", help="specify the whether to optimize the w parameter in drop-out event, default is 0", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    parser.add_argument('-opt_w',"--optimize_w", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    #parser.add_argument('-ci',"--cluster_init", help="specify the whether to use k-means clustering as initialization of path assignment, default is 0", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    parser.add_argument('-ci',"--cluster_init", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    #parser.add_argument('-ldm',"--lamb_data_mult", help="specify the multiplier of lambda data parameter, default is logN",choices=['1','sqrtN','logN','N'],default='log(N)')
    #parser.add_argument('-lrm',"--lamb_ratio_mult", help="specify the multiplier of lambda ratio parameter, default is sqrtR",choices=['1','sqrtR','R'],default='sqrt(r)')
#     parser.add_argument('-ldm',"--lamb_data_mult", help="specify the multiplier of lambda data parameter, default is 1",choices=['1','sqrtN','logN','N'],default='1', help=argparse.SUPPRESS)
#     parser.add_argument('-lrm',"--lamb_ratio_mult", help="specify the multiplier of lambda ratio parameter, default is 1",choices=['1','sqrtR','R'],default='1', help=argparse.SUPPRESS)
#     parser.add_argument('-pc',"--path_constraint", help="specify the whether to apply path constraint in training, default is 0", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
#     parser.add_argument('-pg',"--progress_bar", help="specify the whether to show progress_bar in training, default is 1", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    parser.add_argument('-ldm',"--lamb_data_mult",choices=['1','sqrtN','logN','N'],default='1', help=argparse.SUPPRESS)
    parser.add_argument('-lrm',"--lamb_ratio_mult",choices=['1','sqrtR','R'],default='1', help=argparse.SUPPRESS)
    parser.add_argument('-pc',"--path_constraint", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    parser.add_argument('-pg',"--progress_bar", type=int,choices=[0,1],default=0, help=argparse.SUPPRESS)
    parser.add_argument('-mn',"--model_name", help="specify the model_name",default = None)
    parser.add_argument('-cv',"--cross_validation", help="specify whether to use 5-fold cross_validation, 0 means not, default is 0", type=int, choices=[0,1],default=0)
        
    
    args=parser.parse_args()
    print args
    data_file='data/treutlein2014'
    if args.data_file is not None:
        data_file=args.data_file
    splits=data_file.split('/')
    #structure_file=splits[0]+'/init_cluster_'+splits[1]+'.txt'
    if args.structure_file is not None:
        structure_file=args.structure_file
    else:
        structure_file=splits[0]+'/trained_cluster_'+splits[1]+'.txt'
    verbose=1
    np.random.seed(args.random_seed)
    n_iteration=args.n_iteration
    n_split=args.n_split
    K_param_range=args.k_param_range
    n_anchor=args.n_anchor
    lamb=args.lamb
    max_gene=args.n_gene
    drop_out_param=args.drop_out_param
    assign_by_prob_sampling=args.assign_by_prob_sampling 
    optimize_w=args.optimize_w
    path_constraint=args.path_constraint
    cluster_init=args.cluster_init
    lamb_data_mult=args.lamb_data_mult
    lamb_ratio_mult=args.lamb_ratio_mult
    progress_bar=args.progress_bar
    cv = args.cross_validation
    if args.model_name is not None:
        model_name=args.model_name
    else:
        model_name = 'model/model_'+splits[1]+'_ns_'+str(n_split)+'_lamb_'+str(lamb)+'_ng_'+str(max_gene)+'_cv_'+str(cv)


    verbose=0

    if args.data_file_testing is not None:
        verbose = 0
        cell_names_train,cell_day_train,cell_labels_train,cell_exps_train,gene_names=load_data(data_file,max_gene)
        cell_names_test,cell_day_test,cell_labels_test,cell_exps_test,gene_names=load_data(args.data_file_testing,max_gene)
        model,hid_var_train = init_var_Jun(structure_file,cell_names_train,cell_day_train,cell_exps_train,cell_labels_train)
        _,hid_var_test = init_var_Jun(structure_file,cell_names_test,cell_day_test,cell_exps_test,cell_labels_test)
        max_it=args.n_iteration
        #max_it=2
        max_test_ll=-float('inf')
        for it in range(1,max_it+1):
            n_iteration = 1
            assign_by_prob_sampling=args.assign_by_prob_sampling 
            optimize_likelihood(cell_exps_train, model, hid_var_train,model_name,store_model=False)
            assign_by_prob_sampling=False
            assign_path_and_time(model,hid_var_test,cell_exps_test)
            train_ll = log_likelihood(model,hid_var_train,cell_exps_train)
            test_ll = log_likelihood(model,hid_var_test,cell_exps_test)
            print 'iteration:\t ', it, '\t train_LL:\t ', np.around(train_ll,2), '\t test_ll: \t', np.around(test_ll,2)
            if test_ll > max_test_ll:
                max_test_ll = test_ll
                count = 0
                best_it = it
            else:
                count+=1
            if count>1:
                break
        print 'best_test_it: ', best_it, '\t max_test_ll: ', max_test_ll
        print best_it
        #best_its.append(best_it)
        #best_test_lls.append(max_test_ll)
        #print 'best_its: ', best_its
        #print 'best_test_lls: ', best_test_lls
        #print 'mean_best_its: ',np.mean(best_its)
        #print 'mean_best_test_lls: ',np.mean(best_test_lls)
        #print 'training all data with the best it:', int(np.rint(np.mean(best_its)))
        sys.exit(0)
    
    if args.cross_validation:
        cell_names,cell_day,cell_labels,cell_exps,gene_names=load_data(data_file,max_gene)
        n_fold = 5
        cv_idx = cv_split_idx(cell_day=cell_day,n_fold=n_fold)
        best_its=[]
        best_test_lls=[]
        verbose = 0
        for i in range(1,n_fold+1):
            print 'fold: ', i
            test_idx = cv_idx==i
            train_idx = cv_idx!=i
            cell_day_test = np.array(cell_day)[test_idx]
            cell_exps_test = np.array(cell_exps)[test_idx]
            cell_labels_test = np.array(cell_labels)[test_idx]
            cell_names_test = np.array(cell_names)[test_idx]
            cell_day_train = np.array(cell_day)[train_idx]
            cell_exps_train = np.array(cell_exps)[train_idx]
            cell_labels_train = np.array(cell_labels)[train_idx]
            cell_names_train = np.array(cell_names)[train_idx]
            
            model,hid_var_train = init_var_Jun(structure_file,cell_names_train.tolist(),cell_day_train,cell_exps_train,cell_labels_train)
            _,hid_var_test = init_var_Jun(structure_file,cell_names_test.tolist(),cell_day_test,cell_exps_test,cell_labels_test)
            
            #model,hid_var_train = init_var(adj_mat,cell_day_train,cell_exps_train,cell_labels_train)
            #_,hid_var_test = init_var(adj_mat,cell_day_test,cell_exps_test,cell_labels_test,testing=True)
            
            #train_ll = log_likelihood(model,hid_var_train,cell_exps_train)
            #test_ll = log_likelihood(model,hid_var_test,cell_exps_test)
            #print 'initial training full log-likelihood: ',train_ll
            #print 'initial testing full log-likelihood: ',test_ll
            #print 'iteration:\t ', 0, '\t train_LL:\t ', np.around(train_ll,2), '\t test_ll: \t', np.around(test_ll,2)
            #compute_ARI_confuss_mat(hid_var_train)
            n_iteration = 1
            max_it=args.n_iteration
            max_test_ll=-float('inf')
            count=0
            best_it = 0
            for it in range(1,max_it):
                assign_by_prob_sampling=args.assign_by_prob_sampling 
                optimize_likelihood(cell_exps_train, model, hid_var_train,model_name,store_model=False)
                assign_by_prob_sampling=False
                assign_path_and_time(model,hid_var_test,cell_exps_test)
                train_ll = log_likelihood(model,hid_var_train,cell_exps_train)
                test_ll = log_likelihood(model,hid_var_test,cell_exps_test)
                print 'iteration:\t ', it, '\t train_LL:\t ', np.around(train_ll,2), '\t test_ll: \t', np.around(test_ll,2)
                if test_ll > max_test_ll:
                    max_test_ll = test_ll
                    count = 0
                    best_it = it
                else:
                    count+=1
                if count>1:
                    break
            print 'best_test_it: ', best_it, '\t max_test_ll: ', max_test_ll
            best_its.append(best_it)
            best_test_lls.append(max_test_ll)
        print 'best_its: ', best_its
        print 'best_test_lls: ', best_test_lls
        print 'mean_best_its: ',np.mean(best_its)
        print 'mean_best_test_lls: ',np.mean(best_test_lls)
        print 'training all data with the best it:', int(np.rint(np.mean(best_its)))
        verbose = 1
        n_iteration= int(np.rint(np.mean(best_its)))
        #model,hid_var = init_var(adj_mat,cell_day,cell_exps,cell_labels)
        model,hid_var = init_var_Jun(structure_file,cell_names,cell_day,cell_exps,cell_labels)
        #print 'initial full log-likelihood: ',log_likelihood(model,hid_var,cell_exps)
        n_path=len(model['path_info'] )
        compute_ARI_confuss_mat(hid_var,n_path)
        optimize_likelihood(cell_exps, model, hid_var,model_name)
    else:
        cell_names,cell_day,cell_labels,cell_exps,gene_names=load_data(data_file,max_gene)
        n_cell=len(cell_names)
        model,hid_var = init_var_Jun(structure_file,cell_names,cell_day,cell_exps,cell_labels)
        n_path=len(model['path_info'] )
        compute_ARI_confuss_mat(hid_var,n_path)
        optimize_likelihood(cell_exps, model, hid_var,model_name)
