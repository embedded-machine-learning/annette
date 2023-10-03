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
from scipy.ndimage import convolve1d, gaussian_filter1d
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
hardware = "edgetpu_std"
network = "mobilenetv2-7-sim"
#network = "cf_reid"


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

    power_file= net+'.npy'
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
    plt.plot(result[x])
    plt.show()

    #if x > 0:
    #    time[x] = min(time[x],time[x-1])
        #location[x] = locate_data(cut[0],cut[x-1],cut[x], vis=False)

    #print("Next Time: ",time[x])
    #duration = time[x]+extend
# %%
print(location)
print(time)
# plot cutouts

plt.plot(result[70], label = "result")
plt.plot(cut[70], label = "cut")
plt.show()
# %%

time_new = [0]*num
i2 = 0

def smooth(data, ks=5):
    #data = processing.normalize(data)
    #kernel = np.ones(ks)/ks
    #return convolve1d(data, kernel, mode='reflect')
    return gaussian_filter1d(data, ks, mode='reflect')

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
    divergence = smooth(r1[:length],ks)-smooth(r2[:length],ks)
    divergence = smooth(divergence,ks)
    #divergece = np.log(divergece)[:-ks]
    divergence = divergence[:-ks]

    # find where divergance below threshold
    div = np.where(divergence[:get_max(divergence, max_th)] < div_th)[0]
    return divergence,div

pre = 150

d_thr = 0.05
#d_thr = 0.02
sel_in = True
d_list = []
prev_div = 0
vis_div = True 

first_peak_list = []*num
for i in tqdm(range(0,num)):
#for i in tqdm(range(0,20)):

    if i > 0:
        top_i = i
    else:
        top_i = 1
            
    i1 = i
    r1 = 0
    r2 = 0
    length = 0

    r1_list = []
    r2_list = []
    div_list = []
    for i2 in range(0, top_i):
        # get index of first peak in signal r1 at 90 percent of max
        i1_first_peak = np.where(result[i1] > np.max(result[i1])*0.9)[0][0] - pre
        i2_first_peak = np.where(result[i2] > np.max(result[i2])*0.9)[0][0] - pre
        first_peak_list.append(i1_first_peak)



        r1 = result[i1][i1_first_peak:]
        r2 = result[i2][i2_first_peak:-int(extend*rate)]
        r1_list.append(r1)
        r2_list.append(r2)  
        dist, r1, r2 = processing.align(r1, r2, shift, vis=False, ks=alignks)
        dist -= i1_first_peak-i2_first_peak


        #plot correlation
        length = np.min([len(r1),len(r2)])
        divergence, div  = compute_divergence(ks, div_th, max_th, r1, r2)
        r1_norm = processing.normalize(r1[:length]); r2_norm = processing.normalize(r2[:length])
        divergence2, div2 = compute_divergence(ks, div_th, max_th, r1_norm, r2_norm)
        div_list.append(divergence)

    min_length = np.min([len(d) for d in div_list])
    div_list = [d[:min_length] for d in div_list]
    divergence = np.mean(div_list, axis=0)
    divergence = np.abs(divergence)

    d_list.append(divergence)

    min_length = np.min([len(d) for d in d_list])
    d_list = [d[:min_length] for d in d_list]

    print(len(d_list))

    from functools import reduce
    mom = 0.4
    d_ewma = reduce(lambda x,y : mom*x + (1-mom)*y, d_list)

    s_iter = 3
    s_ewma = [0]*s_iter
    for iter in range(s_iter):
        s_ewma[iter] = smooth(d_ewma,int(ks*(iter+1)*(iter+1)))
    
    # min value of s_ewma[-1]
    sub = np.min(s_ewma[-1])
    s_ewma = [s-sub for s in s_ewma]
        

    if i == 89:
        pass

    p_div = d_ewma

    if prev_div == 0:
        prev_div = len(d_ewma)
    print("prev_div: ",prev_div)

    lower_b = 0
    upper_b = prev_div
    def update_boundaries(lower_b, upper_b):
        print("lower_b?")
        in_lower = input("in_lower: ")
        if in_lower != "":
            lower_b = int(in_lower)
        print("upper_b?")
        in_upper = input("in_upper: ")
        if in_upper != "":
            upper_b = int(in_upper)
        return lower_b, upper_b
    if vis_div == True:
        plt.plot(r1_list[0][:length])
        plt.plot(r2_list[0][:length])
        plt.plot(d_list[-1])
        for iter in s_ewma:
            plt.plot(iter)
        plt.show()
    accept = ""    
    div = 0
    for rev in range(s_iter-1,-1,-1):
        prev_lower_b = lower_b
        prev_upper_b = upper_b
        div = np.where(s_ewma[rev][lower_b:upper_b+1] > d_thr)[0]
        div_double = np.where(s_ewma[rev][lower_b:upper_b+1] > d_thr)[0]
        div_below = np.where(s_ewma[rev][lower_b:upper_b+1] < d_thr)[0]
        if rev == 0:
            minima = find_peaks(-s_ewma[rev][lower_b:upper_b+1], height=-d_thr)[0]
            if len(minima) > 0:
                div = lower_b+minima[-1]
            else:
                if len(div_below) > 0:
                    div = lower_b+div_below[-1]
                else:
                    div = lower_b+np.argmin(s_ewma[rev][lower_b:upper_b+1])
        else:
            # if everywhere higher than threshold, take last point
            if len(div_below) == 0:
                div = upper_b-lower_b
                #print(f"no change: {div}")
            else:
                div = div_below[-1]
                if len(div_double) > 0 and rev == (s_iter-1):
                    #print(f"div_double: {div_double[0]}, lower_b: {lower_b}")
                    # find closes peak to div_double[0]
                    maxima = find_peaks(s_ewma[rev][lower_b:], height=d_thr)[0]
                    if len(maxima) > 0:
                        #sort by distance to div_double[0]
                        maxima = sorted(maxima, key=lambda x: np.abs(div_double[0]-x))
                        upper_b = np.min((upper_b,lower_b+maxima[0]))
                        div = lower_b+div_double[0]
                    #print(f"upper_b: {upper_b}")
                elif s_ewma[rev][div] >  s_ewma[rev-1][div]:
                    # if lower than previous, set new lower bound
                    #print(lower_b, div)
                    lower_b = lower_b+div
                    #print(f"lower_b: {lower_b}")
                else:
                    # if higher than previous, set new upper bound
                    upper_b = lower_b+div
                    #print(f"upper_b: {upper_b}")

        p_lower_b = np.min((prev_lower_b//100, div//100))*100
        p_upper_b = prev_upper_b
        if vis_div == True:
            print(f"lower_b: {lower_b}, upper_b: {upper_b}")
            plt.plot(r1_list[0][p_lower_b:p_upper_b], label="r1")
            plt.plot(r2_list[0][p_lower_b:p_upper_b], label="r2")
            plt.plot(d_list[-1][p_lower_b:p_upper_b], label="div")
            for iter in s_ewma:
                plt.plot(iter[p_lower_b:p_upper_b])
            plt.plot((div-p_lower_b), d_list[-1][div], "o")
            #get current ticks
            ticks = plt.xticks()[0]
            #get current tick labels
            labels = plt.xticks()[1]
            #build new set of tick labels, one for each tick
            new_labels = []
            for tick, label in zip(ticks, labels):
                new_labels.append(str(int(tick + p_lower_b)))
            #set new ticks
            plt.xticks(ticks, new_labels)
            plt.legend()
            plt.show()

        if sel_in == True:
            print("accept")
            accept = input("accept? ")
        if accept == "y" or accept == "yes" or accept == "Y" or accept == "Yes" or accept == "":
            pass
        else:
            lower_b, upper_b = update_boundaries(lower_b, upper_b)
            print(f"lower_b: {lower_b}, upper_b: {upper_b} div: {div}")

    print(f"Final div: {div}")
    p_lower_b = np.max((np.max((lower_b//100, div//100))*100-300, 0))
    p_upper_b = np.min((div+300, length))
    if vis_div == True or True:
        print(f"lower_b: {lower_b}, upper_b: {p_upper_b}")
        plt.plot(r1_list[0][p_lower_b:p_upper_b], label="r1")
        plt.plot(r2_list[0][p_lower_b:p_upper_b], label="r2")
        plt.plot(d_list[-1][p_lower_b:p_upper_b], label="div")
        for iter in s_ewma:
            plt.plot(iter[p_lower_b:p_upper_b])
        plt.plot((div-p_lower_b), d_list[-1][div], "o")
        #get current ticks
        ticks = plt.xticks()[0]
        #get current tick labels
        labels = plt.xticks()[1]
        #build new set of tick labels, one for each tick
        new_labels = []
        for tick, label in zip(ticks, labels):
            new_labels.append(str(int(tick + p_lower_b)))
        #set new ticks
        plt.xticks(ticks, new_labels)
        plt.legend()
        #make a grid
        plt.grid()
        # micro grid
        plt.minorticks_on()
        plt.show()
    print("DIV!")
    if sel_in == True:
        accept = input("final div? ")
    if accept == "y" or accept == "yes" or accept == "Y" or accept == "Yes" or accept == "":
        pass
    else:
        div = int(accept)
    # take the higher of the two
    #div = np.max([div,div2])
    div = np.max([div])



    prev_div = div
    # find time of latest point
    time_new[i1] = (div)/rate
    shifts[i] = dist


    if 0 <= i < 0:
        """
        plt.plot(result[i1])
        plt.plot(result[i2])
        plt.show()
        plt.plot(r1)
        plt.plot(r2)
        plt.show()
        """
        
        #plot smoothed signals
        plt.plot(r1_list[0][:length])
        plt.plot(r2_list[0][:length])
        for r in d_list[:-1]:
            plt.plot(r[:length])

        plt.axvline(x=div, color='r', linestyle='--')
        plt.show()
        #plt.plot(smooth(r1_norm[:length]))
        #plt.plot(smooth(r2_norm[:]))
        #plt.plot(smooth(r1_list[1][:length]))
        #plt.plot(smooth(r2_list[1][:]))


        #plt.plot(divergence)
        plt.plot(d_list[-1])
        for iter in s_ewma:
            plt.plot(iter)
        #plt.plot(divergence2)
        #plt.plot(divergece2)
        #vertical line at latest point
        plt.axvline(x=div, color='r', linestyle='--')
        plt.show()
        print(time_new)

time_stored = time_new


# %%

#time_new = time_stored
time_new.append(0)
print(time_new) 
diff = np.diff(np.flip(time_new))
print(diff)

# %%
import yaml

# load json
filename = Path("..","database", "benchmarks", "imx8", f'{network}_destruct.json')
with open(filename) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

# %%
time_new[0] = time_new[1]
plt.plot(time_new)  	
        
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

plt.plot([s/rate for s in shifts], label="shifts")
plt.plot([f/rate for f in first_peak_list], label='first_peak')
#legend
plt.legend()
plt.show()

# %%
# differences between time points
diff = np.diff(time_new)
print(diff)

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
begin = 0.40 # edgetpu mobilenetv2
#
# begin = 5.1 # ncs2  mobilenetv2
time = [x+begin  for x in time_s]
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


# %%
print(len(cut))

# get names and type of layers in data into dataframe
layer = [0]*len(cut)

df = pd.DataFrame(columns=['name', 'type', 'time'])
for i, l in zip(data['layers'], time):
    # append with concat
    df = pd.concat([df, pd.DataFrame([[i['name'], i['type'], l]], columns=['name', 'type', 'time'])], ignore_index=True)
print(df.tail(10))



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
