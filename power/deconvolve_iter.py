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
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
import logging

#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

# load config
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
pprint(config)

#hardware = "edgetpu_std"
hardware = "imx8"
hardware = "ncs2"
network = "mobilenetv2-7-sim"
#hardware = "gap8"
#network = "testnet"
#network = "deeplabv3_mobilenet_v3_large-sim"
#network = "cf_squeezenet"
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
width = config['hardware'][hardware]['width']
tau = config['hardware'][hardware]['tau']
max_peaks = config['hardware'][hardware]['max_peaks']
min_peaks = config['hardware'][hardware]['min_peaks']
remove_last = config['hardware'][hardware]['remove_last']

pause = np.max((int(pause*rate/down),1))
width = int(width*rate/down)
min_dist = int(time*rate/down)
pre = 0
alignment_shift = 600

select_one = True

logging.debug(f'{pause}')
#%%
vis = False
for layer in tqdm(range(0, layers), position=0, leave=True):
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


    ada_thresh = thresh
    peaks = []
    while len(peaks) < min_peaks:
        peaks, _ = find_peaks(data_down_filt1, height=ada_thresh, distance=min_dist)
        ada_thresh -= 0.05

    logging.debug(f"Initial Peaks {peaks}")
    logging.debug(f"Thresh {thresh}")
    #select peaks with highest width

    #widths = peak_widths(data_down_filt1, peaks, rel_height=0.5)
    # just select the n broadest peaks
    #peaks = peaks[np.argsort(widths[0])][::-1]
    peaks = peaks[:niter]
    #resort by index
    peaks = np.sort(peaks)
    logging.debug(f"Peaks {peaks}")
    
    if vis is True and len(peaks) > 0:
        plt.title("select thresh {thresh}")
        #horizontal line at thresh
        plt.hlines(thresh, 0, len(data_down_filt1))
        plt.plot(data_down_filt1)
        plt.plot(peaks, data_down_filt1[peaks], "x")
        plt.show()

    # just select the n highest peaks
    peaks = peaks[np.argsort(data_down_filt1[peaks])][::-1]
    peaks = peaks[:max_peaks]
    #resort by index
    peaks = np.sort(peaks)
    logging.debug(f"Peaks {peaks}")

    # distance between peaks
    dist_peaks = np.diff(peaks)
    # find most common distance
    dist_peaks_mode = mode((dist_peaks/10).astype(int), keepdims=True)[0][0]*10
    # find peaks with distance to mode smaller than 10%
    p = []
    thresh2 = 0.1
    while len(p) < min_peaks:
        p = np.abs(dist_peaks - dist_peaks_mode) < 1000*thresh2
        thresh2 += 0.05
    argmax = np.argmax(p)
    #prepend one false value to p
    p = np.insert(p, 0, False)
    p[argmax] = True

    if False:
        peaks = peaks[p]

    if len(peaks) < 3:
        # add a peak at the end
        peaks = np.append(peaks, len(data_down_filt1)-1)
    logging.debug(f"Peaks {peaks}")


    # find minimal point between the peaks on the signalpeaks
    minima = []
    for i in range(len(peaks)-1):
        minima.append(np.argmin(data_down_filt1[peaks[i]:peaks[i+1]])+peaks[i])
    


    logging.debug(f"Minima {minima}")

    if vis is True:
        plt.plot(data_down_filt1)
        #plot minima
        for m in minima:
            plt.plot(m, data_down_filt1[m], "x")
        #plot peaks
        for p in peaks:
            plt.plot(p, data_down_filt1[m], "x")
        plt.show()
        
    logging.debug(f"Minima {np.max(minima)}")

    # find edges
    threshold_crossings = np.diff(data_down_filt1 > (np.mean(data_down_filt1[peaks])+np.min(data_down_filt1[minima]))/2, prepend=False)
    if vis is True:
        plt.plot(threshold_crossings)
        plt.plot(data_down_filt2)
        #vertical lines for minima
        plt.show()

    edges = np.where(threshold_crossings == 1)[0]

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
        if (end - start) < min_dist*down*2:
            starts.append(start)
            ends.append(end)

    # find minimal dist between starts and ends
    starts = np.array(starts)
    ends = np.array(ends)
    logging.debug(f"Starts{len(starts)}")
    logging.debug(f"Ends{len(ends)}")

    dists = (ends - starts)
    # get dists mode
    dists_mode = mode((dists).astype(int), keepdims=True)[0][0]
    # find dists with distance to mode smaller than 10%
    p = np.abs(dists - dists_mode) < dists_mode*0.1 
    logging.debug(dists)

    dist = int(np.max(ends - starts))
    logging.debug(dist)

    #upsample 
    news = []
    for start, end in zip(starts, ends):
        curr_dist = end - start
        curr_width = int((curr_dist-dist)/2 + width)
        start2 = (start-curr_width)*down
        end2 = (start+dist+pause)*down
        end2 = int(end2+(tau-1))
        new = data_cut[start2:end2]
        # sum all new in array
        news.append(new)
    
    m = np.min([len(n) for n in news])
    news = [n[:m] for n in news]
    
    logging.debug(f'{len(news)}')
    #filter all cutouts with mean filter
    f = [convolve1d(n[::down], np.ones(int(dist/3))/int(dist/3),mode='reflect') for n in news]
    plot = np.vstack(f)
    if vis is True:
        plt.title("Filtered Cutouts")
        plt.plot(plot.T)
        plt.show()
    
    #remove signals that have highest activity at start or end
    #find max of each signal
    maxs = [np.max(n) for n in f]
    #select signals with max value higher than mean of maxs
    news2 = [n for n, m in zip(news, maxs) if m > np.percentile(maxs, 20)]
    first = [n[0] for n in f]
    last = [n[-1] for n in f]
    #select signals with first and last value lower than mean of first and last
    news2 = [n for n, f, l in zip(news2, first, last) if f < np.percentile(first, 50)]
    news2 = [n for n, f, l in zip(news2, first, last) if l < np.percentile(last, 50)]

    if len(news2) != 0:
        news = news2
    
    
    plot = np.vstack(news)
    
    if vis is True:
        plt.title("Cutouts")
        plt.plot(plot.T)
        plt.show()
    

    d = plot[:,::10]




    clf = IsolationForest(random_state=0).fit(d)
    # remove outliers
    news = [n for n, p in zip(news, clf.predict(d)) if p == 1]

    if vis is True:
        logging.debug(clf.predict(d))
        plot = np.vstack(news)
        plt.plot(plot.T)
        plt.show()
        

    #select the reference based on the median of the argmax 
    #find the argmax of each signal
    argmaxs = [np.argmax(n) for n in news]
    #find the median of the argmaxs
    argmax_median = np.median(argmaxs)
    #find the closest argmax to the median
    argmax_closest = np.abs(argmaxs - argmax_median).argmin()
    #select the reference
    reference = news[argmax_closest]

    #align all signals to the reference
    for iter, n in enumerate(news):
        s, r1, r2 = processing.align(n,reference,alignment_shift)
        logging.debug(f"{s}  {r1.shape}  {r2.shape}")
        news[iter] = r2
    
    
    #remove too short signals based on length outliers
    d = np.array([len(n) for n in news])
    clf = IsolationForest(random_state=0).fit(d.reshape(-1, 1))
    # remove outliers
    news = [n for n, p in zip(news, clf.predict(d.reshape(-1, 1))) if p == 1]
    logging.debug(f"Removed {len(d)-len(news)} outliers")
    if len(news) == 0:
        logging.debug("No signals left")
    else:
        # find minimal length
        d = np.min([len(n) for n in news])
        news = [n[:d] for n in news]
        logging.debug(f"Min length {d}")

    #plot = np.vstack(news)
    
    if vis is True:
        plt.title(f"Before final Isolation forrest")
        plt.plot(plot.T)
        plt.show()

    d = plot[:,::10]
    clf = IsolationForest(random_state=0).fit(d)

    #logging.debug(clf.predict(d))
    # remove outliers
    news = [n for n, p in zip(news, clf.predict(d)) if p == 1]


    """

    plot = np.vstack(news)
    d = plot[:,::10]
    clustering = AgglomerativeClustering().fit(d)


    labels = clustering.labels_
    # get label with most elements
    labels_mode = mode(labels, keepdims=True)[0][0]
    if vis is True:
        plot = np.vstack(news)
        plt.title(f"Before Agglomerative Clustering Mode {labels_mode}")
        plt.plot(plot.T)
        plt.show()
    # remove all labels not equal to mode
    news = [n for n, l in zip(news, labels) if l == labels_mode]

    if vis is True:
        plot = np.vstack(news)
        plt.title(f"After Agglomerative Clustering")
        plt.plot(plot.T)
        plt.show()
        
    #print("start accumulation")
    """

    #accumulate
    new = np.zeros(len(news[0]))
    for n in news:
        new += n
    new = new / len(news)

    #normalize
    if select_one is True:
        new = new / len(news)
        sel = np.argmax(dist)
        curr = 0
        for s, n in enumerate(news):
            best, dist = processing.eucledian_window(new[::10], n[::10])
            logging.debug(f'best:{dist[best]}')
            if dist[best] > curr:
                curr = dist[best]
                sel = s
                logging.debug(f"curr {curr}")

        new = news[sel]

    
    # median filter
    filt = medfilt(new, median)
    filt2 = convolve1d(filt, np.ones(mean)/mean, mode='constant')

    #generate ir of rc circuit

    ir2 = np.arange(0, tau*4, 1)
    ir2 = np.exp(-ir2/tau)
    ir2 = ir2 / np.trapz(ir2)

    # deconvolve
    #print("start deconvolution")

    sub = filt2.min()
    filt3 = sp.signal.medfilt(filt2, 15)
    filt3 = filt3-sub
    dec, rem = sp.signal.deconvolve(filt3, ir2)
    dec = dec + sub

    filt = sp.signal.medfilt(dec, 5)

    # plot
    if vis is True:
        plt.plot(news[0][20:-20], label='reflect')
        plt.plot(new[20:-20], label='reflect')
        plt.plot(filt2[20:-20], label='reflect')
        plt.show()
        plt.plot(dec[tau:-20], label='reflect')
        plt.show()

    Path.mkdir(Path(f'data/{hardware}_{network}'), exist_ok=True)

    # store data
    np.save(f'data/{hardware}_{network}/{network}_{layer}.npy', new[20:-tau])
    np.save(f'data/{hardware}_{network}/{network}_{layer}_dec.npy', filt[tau:-20])
# %%
