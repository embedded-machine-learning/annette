# %%
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, medfilt, convolve, peak_widths
from scipy.ndimage import convolve1d
from scipy.stats import mode
from pathlib import Path
from matplotlib import pyplot as plt
from powerutils import processing
from pprint import pprint
from tqdm import tqdm
import yaml

# load config
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
pprint(config)

hardware = "edgetpu_std"
network = "mobilenetv2-7-sim"
#network = "deeplabv3_mobilenet_v3_large-sim"
#network = "cf_reid"
niter = config['hardware'][hardware]['niter']

# Idut = Umeas / Rshunt
Rshunt = config['hardware'][hardware]['Rshunt']
Uref = config['hardware'][hardware]['Uref']
thresh = config['hardware'][hardware]['thresh']
rate = config['hardware'][hardware]['rate'] # in kHz
time = config['network'][network][hardware]['time'] # ms
layers = config['network'][network][hardware]['layers']

mean = config['hardware'][hardware]['mean'] 
median = config['hardware'][hardware]['median']

down = config['hardware'][hardware]['downsample']
pause = config['hardware'][hardware]['pause']
tau = config['hardware'][hardware]['tau']
max_peaks = config['hardware'][hardware]['max_peaks']
remove_last = config['hardware'][hardware]['remove_last']

pause = int(pause*rate/down)
width = int(0.5*rate/down)
min_dist = int(time*rate/down)
pre = 0
#%%
for layer in tqdm(range(0, layers), position=0, leave=True):
#for layer in tqdm(range(5, layers), position=0, leave=True):
    # load data
    #data_in = np.load(f'/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/database/benchmarks/imx8/destruct/{network}_destruct_{layer}.dat')# resistor = 0.1 ohms
    string = config['network'][network][hardware]['file'].format(hardware=hardware,network=network,layer=layer)
    data_in = np.load(string)# resistor = 0.1 ohms
    data = (Uref - data_in) * data_in / Rshunt

    data_cut = data[:]
    # downsample by 100
    data_down = data_cut[::down]

    # find peaks
    data_down_filt1 = convolve1d(data_down, np.ones(min_dist)/min_dist, mode='reflect')
    data_down_pause = convolve1d(data_down, np.ones(pause)/pause, mode='reflect')
    data_down_filt2 = convolve1d(data_down, np.ones(mean)/mean, mode='reflect')

    peaks, _ = find_peaks(data_down_filt1, height=thresh, distance=min_dist)

    #select peaks with highest width
    widths = peak_widths(data_down_filt1, peaks, rel_height=0.5)
    # just select the n broadest peaks
    peaks = peaks[np.argsort(widths[0])][::-1]
    peaks = peaks[:niter]
    #resort by index
    peaks = np.sort(peaks)
    print("Peaks",peaks)
    

    # just select the n highest peaks
    peaks = peaks[np.argsort(data_down_filt1[peaks])][::-1]
    peaks = peaks[:max_peaks]
    #resort by index
    peaks = np.sort(peaks)
    print("Peaks",peaks)

    # distance between peaks
    dist_peaks = np.diff(peaks)
    # find most common distance
    dist_peaks_mode = mode((dist_peaks/10).astype(int), keepdims=True)[0][0]*10
    # find peaks with distance to mode smaller than 10%
    p = np.abs(dist_peaks - dist_peaks_mode) < 100
    # get index of first true value of p
    argmax = np.argmax(p)
    #prepend one false value to p
    p = np.insert(p, 0, False)
    p[argmax] = True

    #peaks = peaks[np.abs(dist_peaks - dist_peaks_mode) < 100]
    peaks = peaks[p]
    if len(peaks) < 3:
        # add a peak at the end
        peaks = np.append(peaks, len(data_down_filt1)-1)
    print("Peaks",peaks)


    # find minimal point between the peaks on the signalpeaks
    minima = []
    for i in range(len(peaks)-1):
        minima.append(np.argmin(data_down_pause[peaks[i]:peaks[i+1]])+peaks[i])
    


    print("Minima",minima)

    plt.plot(data_down_pause)
    #plot minima
    for m in minima:
        plt.axvline(m, color='r')
    #plot peaks
    for p in peaks:
        plt.axvline(p, color='g')
    plt.plot(data_down_filt1)
    plt.show()
        

    # find edges
    threshold_crossings = np.diff(data_down_filt1 > thresh, prepend=False)
    plt.plot(threshold_crossings)
    plt.plot(data_down_filt2)
    #vertical lines for minima
    plt.show()

    edges = np.where(threshold_crossings == 1)[0]

    #print("Edges",len(edges))
    #print("Peaks",len(peaks))
    #print("Minima",minima)

    #check distance between edges otherwise select next edge
    #get edges lower than peak[1]
    starts = []
    ends = []

    for n in range(1,len(peaks)-1):
        lower = edges[edges < peaks[n]]
        lower = lower[lower >= minima[n-1]]
        #print("lower",lower)
        #print(minima[n-1])
        #print(peaks[n])
        #get edges higher than peak[1]
        higher = edges[edges > peaks[n]]
        higher = higher[higher <= minima[n]]
        #find closest edge to peak[1]
        if len(lower) == 0:
            start = minima[n-1]
        else:
            start = lower[np.abs(lower - minima[n-1]).argmin()]
        if len(higher) == 0:
            end = minima[n]
        else:
            end = higher[np.abs(higher - minima[n]).argmin()]
        starts.append(start)
        ends.append(end)

    print(starts,ends)
    # find minimal dist between starts and ends
    starts = np.array(starts)
    ends = np.array(ends)
    print("Starts",len(starts))
    print("Ends",len(ends))

    dists = (ends - starts)
    # get dists mode
    dists_mode = mode((dists).astype(int), keepdims=True)[0][0]
    # find dists with distance to mode smaller than 10%
    p = np.abs(dists - dists_mode) < dists_mode*0.1 
    print("p",p)
    print(dists)

    dist = np.min(ends - starts)


    #upsample 
    news = []
    for start, end in zip(starts, ends):
        start2 = (start-width)*down
        end2 = (start+dist+width)*down
        new = data_cut[start2:end2+(tau-1)]
        # sum all new in array
        news.append(new)
    
    #align
    for n in news:
        s, r1, r2 = processing.align(n,news[0],200)
        print(s.shape, r1.shape, r2.shape)

    
    #print("start accumulation")

    #accumulate
    new = np.zeros(len(news[0]))
    for n in news:
        new += n

    #normalize
    new = new / len(news)
    sel = np.argmax(dist)
    curr = 0
    for s, n in enumerate(news):
        best, dist = processing.eucledian_window(new, n)
        print(f'best:{dist[best]}')
        if dist[best] > curr:
            curr = dist[best]
            sel = s
            print("curr",curr)

    new = news[sel]

    pre = data_down_filt2[starts[sel]:ends[sel]]
    
    # median filter
    filt = medfilt(new, median)
    filt2 = convolve1d(filt, np.ones(mean)/mean, mode='constant')

    #generate ir of rc circuit

    ir2 = np.arange(0, 1000, 1)
    ir2 = np.exp(-ir2/tau)
    ir2 = ir2 / np.trapz(ir2)

    # deconvolve
    #print("start deconvolution")

    sub = filt2.min()
    filt3 = filt2-sub
    dec, rem = sp.signal.deconvolve(filt3, ir2)
    dec = dec + sub

    filt = sp.signal.medfilt(dec, 5)

    # plot
    plt.plot(filt2[20:-20], label='reflect')
    plt.show()

    Path.mkdir(Path(f'data/{hardware}_{network}'), exist_ok=True)

    # store data
    np.save(f'data/{hardware}_{network}/{network}_{layer}.npy', new[20:-tau])
    np.save(f'data/{hardware}_{network}/{network}_{layer}_dec.npy', filt[tau:-20])
# %%
