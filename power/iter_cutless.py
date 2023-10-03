# %%
from pathlib import Path
import pickle
import numpy as np
#import annette.hardware.ncs2.parser as ncs2_parser
#import annette.utils as utils
import matplotlib.pyplot as plt
from pathlib import Path
import powerutils
from powerutils import processing
import pandas as pd
import plotly.express as px
from scipy.signal import find_peaks, medfilt, convolve, peak_widths
from scipy.ndimage import convolve1d
import scipy
import yaml
from tqdm import tqdm
from pprint import pprint
import logging

#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

# load config
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
pprint(config)
hardware = "gap8"
network = "testnet"

hardware = "imx8"
#hardware = "ncs2"
network = "mobilenetv2-7-sim"


num = config['network'][network][hardware]['layers']
rate = config['hardware'][hardware]['rate'] # in kHz
duration = config['network'][network][hardware]['time'] # ms

tresh = config['network'][network][hardware]['thresh2']
extend = config['network'][network][hardware]['extend']
smoothed = config['network'][network][hardware]['smoothed']
ks = config['network'][network][hardware]['diverge_smoothing']
alignks = config['network'][network][hardware]['alignment_smoothing']

div_th = .1
max_th = div_th*4
#shift = 6000 #ncs2

shift = 300

time = [0]*num
result = [0]*num
cut = [0]*num
location = [0]*num
start = [0]*num
end = [0]*num


for x in tqdm(range(num)):
    net = f'{network}_{str(x)}'

    power_file= net+'_dec.npy'
    power_path = Path('data','edge_tpu_mobilenet_tf2')
    power_path = Path('data', f'{hardware}_{network}')
    
    
    #edgetpu\yolov3_tiny_part_max_pm
    #EDGE TPU MAX
    #net = f'yolov3_tiny_part_max_{x:02d}'
    #power_file= net+'.dat'
    #power_path = Path('data','edgetpu','yolov3_tiny_part_max_pm')
    #tresh = 1.3
    #mult = 50.

    #EDGE TPU STD
    #net = 'mobilenetv2-7-sim'+str(x)+'_edgetpu_std'
    #power_file= net+'.dat'
    #power_path = Path('data','edgetpu','mobilenet_tf2_part_std_pm')
    #tresh = 1.2
    #mult = 50.

    #TX2
    #power_path = Path('data','tx2_mobilenetv2_part_JC0')
    #net = 'tx2_mobilenet_partial_'+str(x)
    #power_file= net+'.dat'
    #tresh = 1.9

    if x < 10:
        v = True 
    else:
        v = False
    #result[x] = np.load(Path(power_path, power_file))*mult
    #ncs2 thresh 0.4 others 0.7
    result[x] = processing.extract_power_profile(power_file, power_path, duration, sample_rate = rate, start_thresh=0.3, end_thresh=0.3, peak_thresh=0.7, vis = v, multiplier=1.0, redefine_start=False, smooth=smoothed)
    start[x], end[x], time[x], cut[x] = processing.get_time(result[x], tresh, kernel_size=smoothed, vis=v) #12 for std edge freq

    #if x > 0:
    #    time[x] = min(time[x],time[x-1])
        #location[x] = locate_data(cut[0],cut[x-1],cut[x], vis=False)

    #print("Next Time: ",time[x])
    #duration = time[x]+extend
# %%
print(location)
print(time)
# plot cutouts

plt.plot(result[0])
plt.show()
# %%

time_new = [0]*num
i2 = 0

def smooth(data, ks=5):
    #data = processing.normalize(data)
    kernel = np.ones(ks)/ks
    return convolve1d(data, kernel, mode='reflect')

shifts = [0]*num

def compute_divergence(ks, div_th, max_th, r1, r2):
    """
    Compute divergence between two power profiles
    Args:
        ks (int): kernel size for smoothing
        div_th (float): divergence threshold
        max_th (float): maximum threshold
        r1 (np.array): power profile 1
        r2 (np.array): power profile 2
    Returns:
        divergence (np.array): divergence between r1 and r2
        div3 (np.array): divergence between r1 and r2, smoothed 3 times
        div2 (np.array): divergence between r1 and r2, smoothed 2 times
        div1 (np.array): divergence between r1 and r2, smoothed 1 time
    """
    
    def get_max(divergence, max_th):
        if np.max(divergence) > max_th:
            peaks = find_peaks(divergence, height=max_th*3)[0]
            if len(peaks) > 0:
                maximum = peaks[0]
            else:
                maximum = np.argmax(divergence)
            maximum = np.argmax(divergence)
        else:
            maximum = len(divergence)
        return maximum
    length = np.min([len(r1),len(r2)])
    divergence = np.abs(smooth(r1[:length],ks)-smooth(r2[:length],ks))
    divergence = smooth(divergence,ks)
    #divergece = np.log(divergece)[:-ks]
    divergence = divergence[:-ks]
    divergence2 = smooth(divergence,ks)
    divergence3 = smooth(divergence,ks)

    div3 = np.where(divergence3[:get_max(divergence3, max_th)] < div_th)[0]
    div2 = np.where(divergence2[:get_max(divergence2, max_th)] < div_th)[0]

    # find where divergance below threshold
    div = np.where(divergence[:get_max(divergence, max_th)] < div_th)[0]
    return divergence,div, div2, div3

for i in tqdm(range(0,num)):
    i1 = i
    # get index of first peak in signal r1 at 90 percent of max
    i1_first_peak = np.where(result[i1] > np.max(result[i1])*0.9)[0][0]
    i2_first_peak = np.where(result[i2] > np.max(result[i2])*0.9)[0][0]


    r1 = result[i1][i1_first_peak:]
    r2 = result[i2][i2_first_peak:-int(extend*rate)]
    dist, r1, r2 = processing.align(r1, r2, shift, vis=False, ks=alignks)
    dist -= i1_first_peak-i2_first_peak


    #plot correlation
    length = np.min([len(r1),len(r2)])
    divergence, div, d2, d3 = compute_divergence(ks, div_th, max_th, r1, r2)
    r1_norm = processing.normalize(r1[:length]); r2_norm = processing.normalize(r2[:length])
    divergence2, div2, _, _ = compute_divergence(ks, div_th, max_th, r1_norm, r2_norm)

    iters = 100
    divergs = [0]*iters
    divergs[0] = divergence
    #smoothen divergence 10 times iteratively
    for divs in range(iters-1):
        divergs[divs+1] = smooth(divergs[divs],100)
    #plot divergences
    for divs in range(iters-1):
        plt.plot(divergs[divs]-divergs[divs+1])
        divergs[divs] = divergs[divs]-divergs[divs+1]
    plt.show()

    #loop backwards through smoothed divergences and find the minima below 0
    minima = []
    if i == 0:
        current_min = [np.argmin(divergs[0])]
    else:
        for divs in range(iters-1,0,-1):
            minima = find_peaks(-divergs[divs], height=0)[0]
            print(minima)
            if divs == iters-1:
                #first iteration select position of global minimum
                current_min = [np.argmin(divergs[divs])]
            else:
                #select minima left of previous minima
                current_min = minima[minima < current_min]
                current_min = current_min[-1]

    print(current_min)

    div = current_min
    #div2 = current_min

        

    print(div)

    # find latest point where divergance below threshold
    # if no point found, take last point
    if len(div) == 0:
        div = len(divergence)
    else:
        div = div[-1]
    if len(div2) == 0:
        div2 = len(divergence2)
    else:
        div2 = div2[-1]
    
    # take the higher of the two
    #div = np.max([div,div2])
    div = np.max([div])



    # find time of latest point
    time_new[i1] = (div+dist)/rate


    if 0 <= i < 100:
        """
        plt.plot(result[i1])
        plt.plot(result[i2])
        plt.show()
        plt.plot(r1)
        plt.plot(r2)
        plt.show()
        """
        
        #plot smoothed signals
        plt.plot(smooth(r1[:length]))
        plt.plot(smooth(r2[:]))
        #plt.plot(smooth(r1_norm[:length]))
        #plt.plot(smooth(r2_norm[:]))
        plt.plot(divergence)
        plt.plot(divergence2)
        #plt.plot(divergece2)
        #vertical line at latest point
        plt.axvline(x=div, color='r', linestyle='--')
        plt.show()

# %%
print(time_new)

# %%

time_new[0] = time_new[1]
plt.plot(time_new)  	
plt.plot(shifts/np.max(shifts))
        
# remove outliers where time before and after are both higher or lower
for i in range(1,num-1):
    if time_new[i-1] < time_new[i] and time_new[i+1] < time_new[i]:
        time_new[i] = time_new[i+1]
    if time_new[i-1] > time_new[i] and time_new[i+1] > time_new[i]:
        time_new[i] = time_new[i+1]
plt.plot(time_new)  	
# make sure time is always increasing
for i in range(1,num):
    if time_new[i-1] < time_new[i]:
        time_new[i] = time_new[i-1]
plt.plot(time_new)


#%%
#np.flip(time)
np.flip(location)
time_s = time_new
# %%
print(time)
print(shifts)
# %%
print(time)
print(start)
time = [x  for x in time_s]
start[0] = 0
print(len(time))
#%%

#ncs2_results['measured'] *= duration/np.sum(ncs2_results['measured'])
#print(ncs2_results)

#all	3,33ms	all layers on coral
#63	3,29ms	excluding to prev.: q,smax,rs,conv2d
#61	3,25ms	excluding to prev.: mean,conv2d
#60	3,20ms	excluding to prev.: conv2d
#59	3,11ms	excluding to prev.: depthconv2d
#3	0,73ms	only q,conv2d,deconv2d,conv2d on coral


# Create DataFrame  
print(start[0])
start[0] = 0.4
print(time)
#concatenate lists

x= np.arange(len(result[0]))/(rate)
fig = plt.figure(figsize = (5.5,2.5))
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(x,result[0], label='Power')
for l, xc in enumerate(np.flip(time)):
    if l == 0:
        plt.axvline(x=xc,c='green',label='Network Start')
    elif l == 1:
        plt.axvline(x=xc,c='red',label='Layer Transition')
    else:
        plt.axvline(x=xc,c='red')
plt.axvline(x=0,c='green')
plt.legend()
plt.ylabel("Power [W]")
plt.xlabel("Time [ms]")
plt.xlim(-0.01,1.4)  # adjust the top leaving bottom unchanged
fig.savefig("edge_example.pdf", bbox_inches='tight')
cut[0] = result[0][:]

# %
#for i in [99,98,97,96,95,94,93,92,91,90,89,88,87,86,85]:
for i in [0]:
    plt.show()
    x= np.arange(len(result[i]))/(rate)
    #plt.rcParams["figure.figsize"] = (40,10)
    plt.plot(x,result[i])
    plt.show()
# result = power_utils.unite_latency_power_meas(latency_file, latency_path, power_file, power_path, sample_rate = rate)

x= np.arange(len(cut[0]))/rate
#plt.rcParams["figure.figsize"] = (40,10)
plt.plot(x,cut[0])
for xc in np.flip(time):
    plt.axvline(x=xc,c='red')
    print(xc)
plt.show()

# %% Plot:
x= np.arange(len(cut[0]))/rate
#plt.rcParams["figure.figsize"] = (40,10)
plt.plot(x,cut[0])
for xc in np.flip(time):
    #plt.axvline(x=xc,c='red')
    print(xc)
plt.show()

# %% Plot new
layer = [0]*len(cut)
reconst = None
l = [0]*len(cut)
#for i in [90,91,92,93,94,95,96,97,98,99]:

for i in range(len(cut)):
    x= np.arange(len(cut[i]))/rate
    #plt.rcParams["figure.figsize"] = (40,10)
    #plt.plot(x,cut[i])
    #plt.show()
    #for xc in np.flip(time):
        #plt.axvline(x=xc,c='red')
        #print(xc)
    if i+1 < len(cut):
        length = len(cut[i])-len(cut[i+1])
        if length > 0:
            layer[i] = cut[i][-length:]
            positions = len(cut[i])-length
            m = [10000]*positions
            for idx in range(positions):
                #exclude layer_lenght
                exclude = np.arange(idx,idx+length)
                curr = np.delete(cut[i],exclude) # remove window of layer_length
                m_curr = (np.square(cut[i+1] - curr)).mean() # mse for current execution to execution with one less layer
                if m_curr < m[idx]: # select if lower than current minimum
                    m[idx] = m_curr
                    layer[i] = cut[i][idx:idx+length]
                    s = idx
                    
            a = 1000
            for idx in np.arange(s-10,s+10):
                curr = cut[1][idx:idx+length]
                if len(layer[i]) == len(curr):
                    m_curr = (np.abs(layer[i] - curr)).mean() # mse for current execution to execution with one less layer
                    #print(m_curr)
                    if m_curr < a: # select if lower than current minimum
                        layer[i] = cut[1][idx:idx+length]
                        a = m_curr
            
            if reconst is None:
                reconst = layer[i]
            else:
                reconst = np.append(layer[i],reconst)
            
            #x= np.arange(len(layer[i]))/rate
            #plt.plot(m)
            #plt.show()
            #plt.plot(x,layer[i])
            #plt.show()

            l[i] = length

    elif i+1 == len(cut):
        layer[i] = cut[i][:]
        reconst = np.append(layer[i],reconst)
        l[i] = len(layer[i])
        #print(len(cut[i]))

print(len(reconst),len(cut[0]))
t = np.flip(l)/rate
t_cum = np.cumsum(t)
x= np.arange(len(reconst))/rate
plt.rcParams["figure.figsize"] = (40,10)
plt.plot(x,reconst)
for xc in t_cum:
    plt.axvline(x=xc,c='red')
    #print(xc)
plt.show()

x= np.arange(len(cut[0]))/rate
plt.rcParams["figure.figsize"] = (40,10)
plt.plot(x,cut[0])
for xc in np.flip(time):
    plt.axvline(x=xc,c='red')
    #print(xc)
plt.show()

# %%
original = cut[0]
reconst0 = reconst
m_curr2 = (np.square(cut[0][:len(reconst0)] - reconst0)).mean() # mse for current execution to execution with one less layer
# %%
#print(m_curr0)
#print(m_curr1)
print(m_curr2)




# %%
import json
with open('layers.json', 'r') as json_file:
    layers = json.load(json_file)
#print(time[0])
time1 = np.append(np.array(time)[1:],0)
#print(time1)
time2 = (time) - time1
print(time2)
time3 = time2[1]
time2[1] = time2[0]
time2[0] = time3
print(time2)
# reverse time2
time2 = np.flip(time2)
#data = {'name': ['3','59', '60', '61', '63','all'],\
#     'measured': [0.33, 2.38, 0.09, 0.05, 0.04, 0.04],\
#     'type': ['Conv','Conv','Conv','Conv','Conv','Conv']}  
data = {'name': [], 'measured': [], 'type': []}
for l in range(100):
    data['name'].append(str(l))
    data['measured'].append(time2[l])
    data['type'].append(layers[l]['type'])
data['name'].reverse()
data['measured'].reverse()
data['type'].reverse()
res = pd.DataFrame(data)  
print(data)
#print(res.tail(20))
# %%
print(res)

# %%
s = (int(start[0]*500))
dur = np.int(np.sum(res['measured'])*500)
print(s)
print(dur)
power = result[0][s:s+dur]
plt.plot(power)
plt.show()
print(res)

# %% reconstruction
power = reconst
print(res)
print(t)
# %%
import powerutils
annette = pd.read_csv(Path('data','edgetpu','mobile-ana','mobilenetv2_2.csv')).drop([0]).reset_index(drop=True)
res = powerutils.unite_latency_df_power_meas(res, power, sample_rate = 500, rate_div=1)
res2 = res

# %%
res2.tail(20)
# %%
res2[res2['measured']==0]

# %%
res

# %%
res['ops'] = annette['num_ops']
res['inputs'] = annette['num_inputs']
res['outputs'] = annette['num_outputs']
res['weights'] = annette['num_weights']
res['activations'] = annette['num_inputs']+annette['num_outputs']
res['data'] = res['activations']+res['weights']

v = 1
idle = 1.1

res['mean (V)'] = res['mean (V)']*v
res['dyn (Vs)'] = ((res['mean (V)']) -idle) * res['measured']
res['total (Vs)'] = (res['mean (V)']) * res['measured']
res['dyn (V)'] = (res['mean (V)']) - idle
res['mult (mJ)'] = res['mult (Vs)']*v
res['Mops'] = res['ops']/1000000
print(res.head(20))
print(res['dyn (Vs)'])
# %%
np.argmax(res['activations'])
# %% 
# 
import copy
print(res)
colors = np.where(res["type"]=='Conv','r','b').T
colors[res['type']=='DepthwiseConv']= 'y'
#colors[res["type"]=='<Extra>'] = 'g'
#colors[res["type"]=='Elwise'] = 'b'
colors[res["type"]=='Pool'] = 'black'
#colors[res["type"]=='reshape'] = 'red'
colors[res["type"]=='SoftMax'] = 'black'
colors[res["type"]=='FullyConnected'] = 'green'
#colors[res["type"]=='Relu6'] = 'black'
print(colors)
fig = plt.figure(figsize = (5,2.5))
ax = plt.axes()
plt.grid(color='grey', linestyle='-', linewidth=1)
#plt.scatter(result['measured'].to_numpy(),result['total (Vs)'].to_numpy(),marker='x',c=colors)
plt.ylabel("$E_{total}$ [mJ]")
plt.xlabel("Layer Runtime [ms]")
test = ax.scatter(res['measured'].to_numpy(),res['total (Vs)'].to_numpy(),marker='x',c=colors,label=colors)
test5 = ax.plot([0, 0.3], [0, 0.375])
test1 = copy.copy(test)
test1.set_color('red')
test2 = copy.copy(test)
test2.set_color('black')
test3 = copy.copy(test)
test3.set_color('blue')
test4 = copy.copy(test)
test4.set_color('y')
test5 = copy.copy(test)
test5.set_color('g')
plt.legend([test1, test2, test3, test4, test5],['Convolution', 'Pooling', 'Addition', 'DepthwiseConv','FullyConnected'])

plt.show()
fig.savefig("imx_total.pdf", bbox_inches='tight')

# %%
fig = plt.figure(figsize = (5,2.5))
ax = plt.axes()
plt.grid(color='grey', linestyle='-', linewidth=1)
test = ax.scatter(res['measured'].to_numpy(),res['dyn (Vs)'].to_numpy(),marker='x',c=colors,label=colors)
#test5 = ax.plot([0, 0.4], [0, 0.5])
plt.ylabel("$E_{dyn}$ [mJ]")
plt.xlabel("Layer Runtime [ms]")
plt.legend([test1, test2, test3, test4, test5],['Convolution', 'Pooling', 'Addition', 'DepthwiseConv','FullyConnected'])

plt.show()
fig.savefig("imx_dyn.pdf", bbox_inches='tight')

# %%
fig = plt.figure(figsize = (5,2.5))
plt.grid(color='grey', linestyle='-', linewidth=1)
test = plt.scatter(res['Mops'].to_numpy(),res['total (Vs)'].to_numpy(),marker='x',c=colors,label=colors)
plt.ylabel("$E_{total}$ [mJ]")
plt.xlabel("[Mops]")
plt.legend([test1, test2, test3, test4, test5],['Convolution', 'Pooling', 'Addition', 'DepthwiseConv','FullyConnected'])
plt.show()
fig.savefig("imx_Mops.pdf", bbox_inches='tight')
# %%

# %%
fig = plt.figure(figsize = (5,2.5))
plt.grid(color='grey', linestyle='-', linewidth=1)
test = plt.scatter(res['weights'].to_numpy(),res['total (Vs)'].to_numpy(),marker='x',c=colors,label=colors)
plt.ylabel("$E_{total}$ [mJ]")
plt.xlabel("[parameters]")
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend([test1, test2, test3, test4, test5],['Convolution', 'Pooling', 'Addition', 'DepthwiseConv','FullyConnected'])
plt.show()
fig.savefig("imx_weights.pdf", bbox_inches='tight')
# %%
fig = plt.figure(figsize = (5,2.5))
plt.grid(color='grey', linestyle='-', linewidth=1)
test = plt.scatter(res['activations'].to_numpy(),res['total (Vs)'].to_numpy(),marker='x',c=colors,label=colors)
plt.ylabel("$E_{total}$ [mJ]")
plt.xlabel("[activations]")
plt.xlim(-5e4,1.5e6)  # adjust the top leaving bottom unchanged
plt.legend([test1, test2, test3, test4, test5],['Convolution', 'Pooling', 'Addition', 'DepthwiseConv','FullyConnected'])
plt.show()
fig.savefig("imx_act.pdf", bbox_inches='tight')
#%% SAVE RESULTS FOR NICOLAS

import pickle
signal = reconst

data = {'transition_times': np.flip(time), 'signal': signal}

with open('imx_record.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

res.to_pickle("./imx_dataframe.pkl")
# %%
f = []
s = []
for fq in range(1,5001):
    x = 5000000/(fq*1000)
    if x - int(x) == 0.0:
        f.append(fq/10)
        s.append(int(x))
print(len(s))
print(s)
print(f)

total_result_dict = {}
result_dict = {}
total_energy_dict = {}
total_error_dict = {}
total_error_rel_dict = {}
mean_power_dict = {}
mean_power_rel_dict = {}
dropped_dict = {}
dropped_rel_dict = {}
mean_layer_error_rel_dict= {}
mean_layer_dyn_error_rel_dict= {}
summary_dict= {}
result = [0]*len(s)
total_result = [0]*len(s)
total_energy = [0]*len(s)
total_error = [0]*len(s)
total_error_rel = [0]*len(s)
mean_power = [0]*len(s)
mean_power_rel = [0]*len(s)
dropped = [0]*len(s)
dropped_rel = [0]*len(s)
mean_layer_error_rel= [0]*len(s)
mean_layer_dyn_error_rel= [0]*len(s)
import copy
for x in range(len(s)):
    print("Duration",duration)
    print("Rate_div",s[x])
    result[x] = copy.deepcopy(power_utils.unite_latency_df_power_meas(res, power, sample_rate = 500, rate_div=s[x]))
    result[x] = result[x][result[x]['measured'] != 0]
    total_result[x] = power
    print("Yo: ",result[x]['mean (V)'].isna().sum())

for x in range(len(s)):
    print("Yo: ",result[x]['mean (V)'].isna().sum())
# %%
print(result[0])

# %%
for x in reversed(range(len(s))):
        mean_power[x] = total_result[len(s)-1][::s[x]].mean()
        mean_power_rel[x] = abs(total_result[len(s)-1].mean() - total_result[len(s)-1][::s[x]].mean())/(total_result[len(s)-1].mean()-1.03)
        total_energy[x] = (total_result[len(s)-1][::s[x]].mean())*np.sum(result[x]['measured'])
        total_error[x] = abs(total_energy[len(s)-1]-total_energy[x])
        total_error_rel[x] = total_error[x]/total_energy[len(s)-1] *100
        mean_layer_error_rel[x] = (abs(result[len(s)-1]['mean (V)']-result[x]['mean (V)'].fillna(0))/(result[len(s)-1]['mean (V)'])).mean()*100
        mean_layer_dyn_error_rel[x] = (abs(result[len(s)-1]['mean (V)']-result[x]['mean (V)'].fillna(0))/(result[len(s)-1]['mean (V)']-1.03)).mean()*100
        mean_layer_error_rel[x] = (abs(result[len(s)-1]['mean (V)']-result[x]['mean (V)'])/(result[len(s)-1]['mean (V)'])).mean()*100
        mean_layer_dyn_error_rel[x] = (abs(result[len(s)-1]['mean (V)']-result[x]['mean (V)'])/(result[len(s)-1]['mean (V)']-1.03)).mean()*100
        dropped[x] = result[x]['mean (V)'].isna().sum()
        dropped_rel[x] = result[x]['mean (V)'].isna().sum()/len(result[x])*100
        print(s[x], f[x], mean_power_rel[x], mean_power[x], total_energy[x], total_error[x], total_error_rel[x],dropped_rel[x],dropped[x],len(result[x]))

total_result_dict[net] = total_result
result_dict[net] = result
total_energy_dict[net] = total_energy 
total_error_dict[net] = total_error 
total_error_rel_dict[net] = total_error_rel 
mean_power_dict[net] = mean_power 
mean_power_rel_dict[net] = mean_power_rel 
mean_layer_error_rel_dict[net] = mean_layer_error_rel 
mean_layer_dyn_error_rel_dict[net] = mean_layer_dyn_error_rel 
dropped_dict[net] = dropped 
dropped_rel_dict[net] = dropped_rel
summary_dict = {
    "total_result" : total_result_dict,
    "result" : result_dict,
    "total_energy" : total_energy_dict,
    "total_error" : total_error_dict,
    "total_error_rel" : total_error_rel_dict,
    "mean_power" : mean_power_dict,
    "mean_power_rel" : mean_power_rel_dict,
    "mean_layer_error_rel" : mean_layer_error_rel_dict,
    "mean_layer_dyn_error_rel" : mean_layer_dyn_error_rel_dict,
    "dropped" : dropped_dict,
    "dropped_rel" : dropped_rel_dict}
# %%
summary_df = pd.concat({k: pd.DataFrame(v) for k, v in summary_dict.items()}, axis=1)
print(summary_dict)
# %%
fig = plt.figure()
plt.rcParams["figure.figsize"] = (5.5,2.5)
fqs = [x*1000 for x in f]
plt.xscale('log')
plt.grid(color='grey', linestyle='-', linewidth=1)
plt.plot(fqs,summary_df['mean_layer_error_rel'].T.mean(), label='per layer $E_{total}$')
plt.plot(fqs,summary_df['mean_layer_dyn_error_rel'].T.mean(), label='per layer $E_{dyn}$')
plt.plot(fqs,summary_df['total_error_rel'].T.mean(), label='$E_{total}$')
plt.plot(fqs,summary_df['mean_power_rel'].T.mean()*100, label='$E_{dyn}$')
plt.plot(fqs,summary_df['dropped_rel'].T.mean(), label='dropped layers')

plt.ylabel("Average Error [%]")
plt.xlabel("Sampling Frequency [Hz]")
plt.ylim(0,50)  # adjust the top leaving bottom unchanged
plt.xlim(0,1000000)  # adjust the top leaving bottom unchanged

plt.legend()
plt.show()
fig.savefig("edge_frequency.pdf", bbox_inches='tight')
# %%
