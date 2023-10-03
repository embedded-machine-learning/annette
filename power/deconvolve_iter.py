# %%
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, medfilt, convolve, peak_widths
from scipy.ndimage import convolve1d, gaussian_filter1d
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

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# load config
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
pprint(config)

#hardware = "edgetpu_std"
hardware = "imx8"
#hardware = "ncs2"
#network = "mobilenetv2-7-sim"
network = "yolov3"
#hardware = "gap8"
#network = "testnet"
#network = "deeplabv3_mobilenet_v3_large-sim"
#network = "cf_squeezenet"
#network = "cf_reid"
#network = "mobilenetv2"

#hardware = "xavier-fp32"
#network = "xavier_mobilenetv2"
niter = config['hardware'][hardware]['niter']

def try_from_network(config, key, alt = None):
    # check if key is in network
    if key in config['network'][network][hardware].keys():
        return config['network'][network][hardware][key]
    elif key in config['network'][network].keys():
        return config['network'][network][key]
    elif key in config['hardware'][hardware].keys():
        return config['hardware'][hardware][key]
    elif isinstance(alt, str):
        return try_from_network(config, alt) 
    elif alt is not None:
        return alt

# Idut = Umeas / Rshunt
Rshunt = try_from_network(config, 'Rshunt')
Uref = try_from_network(config, 'Uref')
thresh = try_from_network(config, 'thresh')
rate = try_from_network(config, 'rate') # in kHz
mean = try_from_network(config, 'mean')
median = try_from_network(config, 'median')
down = try_from_network(config, 'downsample')
pause = try_from_network(config, 'pause')
width = try_from_network(config, 'width')
tau = try_from_network(config, 'tau')
max_peaks = try_from_network(config, 'max_peaks')
min_peaks = try_from_network(config, 'min_peaks')
remove_last = try_from_network(config, 'remove_last')

layers = try_from_network(config, 'layers')
time = try_from_network(config, 'time') # ms
gop = try_from_network(config, 'gop')
mpar = try_from_network(config, 'mpar')

start_time = try_from_network(config, 'start', 0)
bench_start_samp= try_from_network(config, 'start_bench', 0)*down
exact_time = try_from_network(config, 'exact_time', 'time')

# imx8 tinyyolov3 npu
#start = 4.6 
#exact_time = 9.727

# imx8 tinyyolov3 cpu
#start = int(7*rate)
#exact_time = 758.5

exact_samples = int(exact_time*rate)
start_samples = int(start_time*rate)
pause = np.max((int(pause*rate/down),1))
width = int(width*rate/down)
min_dist = int(time*rate/down)
pre = 0
alignment_shift = 500
select_one =False 

print(bench_start_samp*10)

logging.debug(f'{pause}')
#%%
vis = False 
final_vis = True 
for layer in tqdm(range(0, layers), position=0, leave=True):
    # load data
    #data_in = np.load(f'/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/database/benchmarks/imx8/destruct/{network}_destruct_{layer}.dat')# resistor = 0.1 ohms
    string = config['network'][network][hardware]['file'].format(hardware=hardware,network=network,layer=layer)
    data_in = np.load(string)# resistor = 0.1 ohms
    data_in = data_in[bench_start_samp:]
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
        # remove peak height outliers
        if niter > 10:
            try:
                clf = IsolationForest(random_state=0).fit(data_down_filt1[peaks].reshape(-1, 1))
                peaks = peaks[clf.predict(data_down_filt1[peaks].reshape(-1, 1)) == 1]
                logging.debug(f"Peaks {peaks}")
                if vis is True and len(peaks) > min_peaks//2:
                    plt.title(f"select thresh {ada_thresh}")
                    #horizontal line at thresh
                    plt.hlines(ada_thresh, 0, len(data_down_filt1))
                    plt.plot(data_down_filt1)
                    plt.plot(peaks, data_down_filt1[peaks], "x")
                    plt.show()
            except:
                logging.debug(f"IsolationForest failed")
        ada_thresh -= 0.05

    logging.debug(f"Initial Peaks {peaks}")
    logging.debug(f"Thresh {thresh}")
    #select peaks with highest width

    #widths = peak_widths(data_down_filt1, peaks, rel_height=0.5)
    # just select the n broadest peaks
    #peaks = peaks[np.argsort(widths[0])][::-1]
    #peaks = peaks[:niter]
    #resort by index
    peaks = np.sort(peaks)
    logging.debug(f"Peaks {peaks}")
    
    if vis is True and len(peaks) > 0:
        plt.title(f"select thresh {thresh}")
        #horizontal line at thresh
        plt.hlines(thresh, 0, len(data_down_filt1))
        plt.plot(data_down_filt1)
        plt.plot(peaks, data_down_filt1[peaks], "x")
        plt.show()
    
    # select similar peaks with same distance
    # distance between peaks
    dist_peaks = np.diff(peaks)
    # find most common distance
    dist_peaks_mode = mode((dist_peaks/10).astype(int), keepdims=True)[0][0]*10
    logging.debug(f"Mode {dist_peaks_mode}")




    # just select the n highest peaks
    logging.debug(f"{np.argsort(data_down_filt1[peaks])}")
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
    thresh2 = 0.1
    p2 = 0
    while True:
        p = np.abs(dist_peaks - dist_peaks_mode) < 1000*thresh2
        # count true values
        p2 = np.sum(p)
        thresh2 += 0.05
        if p2 >= min((min_peaks, len(dist_peaks))):
            break
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
        minima.append(np.argmin(data_down_pause[peaks[i]:peaks[i+1]])+peaks[i])
    
    pre_minima = [0]
    post_minima = [0]
    for i in range(1,len(peaks)-1):
        #print(peaks[i],(peaks[i]+dist_peaks_mode))
        #print((peaks[i]-dist_peaks_mode),peaks[i])
        #post_minima.append(np.argmin(data_down_pause[peaks[i]:(peaks[i]+dist_peaks_mode)])+peaks[i])
        post_minima.append(np.argmin(data_down_pause[peaks[i]:(peaks[i]+min_dist+tau*4+dist_peaks_mode)])+peaks[i])
        #if peaks[i]-dist_peaks_mode < 0:
        if peaks[i]-min_dist < 0:
            pre_minima.append(np.argmin(data_down_pause[0:peaks[i]])+peaks[i])
        else:
            pre_minima.append(np.argmin(data_down_pause[(peaks[i]-min_dist):peaks[i]])+peaks[i]-min_dist)
            #pre_minima.append(np.argmin(data_down_pause[(peaks[i]-dist_peaks_mode):peaks[i]])+peaks[i]-dist_peaks_mode)

    


    logging.debug(f"Minima {minima}")
    logging.debug(f"Pre_Minima{pre_minima}")
    logging.debug(f"Post_minima{post_minima}")

    if vis is True:
        plt.plot(data_down_pause)
        #plot minima
        for m, m_pre, m_post in zip(minima, pre_minima, post_minima):
            plt.plot(m, data_down_filt1[m], "x")
            plt.plot(m_pre, data_down_filt1[m], "x")
            plt.plot(m_post, data_down_filt1[m], "x")
        #plot peaks
        for p in peaks:
            plt.plot(p, data_down_filt1[p], "x")
        plt.show()
        

    # find edges
    threshold_crossings = np.diff(data_down_filt1 > (np.mean(data_down_filt1[peaks])+np.min(data_down_filt1[minima]))/2, prepend=False)
    #threshold_crossings = np.diff(data_down_filt1 > (np.mean(data_down_pause[peaks])+np.min(data_down_pause[minima]))/2, prepend=False)
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
        lower = lower[lower >= pre_minima[n]]
        #print("lower",lower)
        #print(minima[n-1])
        #print(peaks[n])
        #get edges higher than peak[1]
        higher = edges[edges > peaks[n]]
        higher = higher[higher <= post_minima[n]]
        #find closest edge to peak[1]

        if len(lower) == 0:
            start = pre_minima[n]
        else:
            start = lower[np.abs(lower - pre_minima[n]).argmin()]
        if len(higher) == 0:
            end = post_minima[n]
        else:
            end = higher[np.abs(higher - post_minima[n]).argmin()]
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
        end2 = (start+dist+pause+width)*down
        end2 = int(end2+(tau-1))
        new = data_cut[start2:end2]
        # sum all new in array
        news.append(new)
    

    #remove all signals with length smaller than min_dist
    news = [n for n in news if len(n) > min_dist*down]

    #find mean of all signal lenghts
    m = int(np.min([len(n) for n in news]))
    # remove all signals that are shorter than mean
    news = [n for n in news if len(n) > m]
    # cut all signals to mean length so that they have the highest activity in the middle
    # get center m samples of each signal
    #news = [n[int((len(n)-m)/2):int((len(n)+m)/2)] for n in news]
    news = [n[:m] for n in news]
    
    logging.debug(f'{len(news)}')
    #filter all cutouts with mean filter
    f = [convolve1d(n[::down], np.ones(int(dist/10))/int(dist/10),mode='reflect') for n in news]
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
    news_temp = [n for n, p in zip(news, clf.predict(d)) if p == 1]
    if len(news_temp) != 0:
        logging.debug(f"Removed {len(news)-len(news_temp)} outliers")
        news = news_temp

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

    # if signal is long don't align
    if len(reference) < 1000000:
        #align all signals to the reference
        for iter, n in enumerate(news):
            s, r1, r2 = processing.align(n,reference,alignment_shift)
            logging.debug(f"{s}  {r1.shape}  {r2.shape}")
            news[iter] = r2
    else:
        logging.debug("Reference too long, skipping alignment")
        select_one = True
    
    
    #remove too short signals based on length outliers
    d = np.array([len(n) for n in news])
    clf = IsolationForest(random_state=0).fit(d.reshape(-1, 1))
    # remove outliers
    news_temp = [n for n, p in zip(news, clf.predict(d.reshape(-1, 1))) if p == 1]
    if len(news_temp) == 0:
        logging.debug("No signals left")
        logging.debug(f"{len(news)} outliers")
    else:
        news = news_temp
    # find minimal length
    logging.debug(f"Removed {len(d)-len(news_temp)} outliers")
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
    #news = [n for n, p in zip(news, clf.predict(d)) if p == 1]
    news_temp = [n for n, p in zip(news, clf.predict(d)) if p == 1]
    if len(news_temp) == 0:
        logging.debug("No signals left")
        logging.debug(f"{len(news)} outliers")
    else:
        news = news_temp


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

    select_one = False
    #normalize
    if select_one is True:
        new = news[1]
    elif select_one is True:
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
    filt2 = gaussian_filter1d(filt, 5, mode='reflect')

    #generate ir of rc circuit

    ir2 = np.arange(0, tau*4, 1)
    ir2 = np.exp(-ir2/tau)
    ir2 = ir2 / np.trapz(ir2)

    # deconvolve
    #print("start deconvolution")

    sub = filt2.min()
    filt2 = filt2 - sub
    #filt3 = sp.signal.medfilt(filt2, 15)
    #filt3 = filt3-sub
    if tau != 1:
        dec, rem = sp.signal.deconvolve(filt2, ir2)
        dec = dec + sub
    else:
        dec = filt2 + sub

    filt_dec = sp.signal.medfilt(dec, 5)

    # plot
    if (vis is True) or (final_vis is True):
        #plt.plot(dec[20:-20], label='deconved')
        plt.plot(news[0][20:-20], label='first')
        plt.plot(new[20:-tau], label='selected')
        plt.legend()
        plt.show()
        plt.plot(dec[tau:-20], label='rdeflect')
        plt.show()

    Path.mkdir(Path(f'data/{hardware}_{network}'), exist_ok=True)

    # store data
    np.save(f'data/{hardware}_{network}/{network}_{layer}.npy', new[20:-tau])
    np.save(f'data/{hardware}_{network}/{network}_{layer}_dec.npy', filt_dec[tau:-20])

#%%

#exact_time = 9.8
print(start_samples, exact_samples+start_samples)

exact_samples = exact_samples
fig = plt.figure(figsize = (5,2.5))
plt.plot(np.arange(exact_samples)/rate, new[start_samples:start_samples+exact_samples], label='Power')
plt.plot(np.arange(exact_samples)/rate, dec[start_samples:start_samples+exact_samples], label='Deconvolved Power')
# x axis in ms
plt.xlabel('Time [ms]')
# y axis in W
plt.ylabel('Power [W]')
# add grid
plt.grid()
# cut off plot lenght
plt.xlim(0, exact_samples/rate)
#plt.xlim(90, 100)
# vertical line at exact time
#plt.axvline(108.5, color='r', linestyle='--', label='Exact time')
#plt.axvline(104.95, color='r', linestyle='--', label='Exact time')
#plt.axvline(3.8, color='r', linestyle='--', label='Exact time')
# reduce size and save to pdf

plt.legend()
plt.show()
fig.savefig(f'graphs/{hardware}_{network}.pdf', bbox_inches='tight', pad_inches=0.01)

# compute energy of power trace by integrating over time

energy = np.trapz(new[start_samples:start_samples+exact_samples]/rate)
exact_energy = np.trapz(dec[start_samples:start_samples+exact_samples]/rate)

# print error in percent with two decimals
print(f"Error: {np.abs(energy-exact_energy)/exact_energy*100:.2f}%")

print(f"Energy: {energy} mJ")
print(f"Energy: {exact_energy} mJ")

# subtract base power constumption of entire power trace
# and compute energy of power trace by integrating over time

base_power = np.min(filt)
print(f"Base power: {base_power} W")


dyn_energy = np.trapz((new[start_samples:start_samples+exact_samples]-base_power)/rate)
exact_dyn_energy = np.trapz((dec[start_samples:start_samples+exact_samples]-base_power)/rate)

# print error in percent with two decimals
print(f"Error: {np.abs(dyn_energy-exact_dyn_energy)/exact_dyn_energy*100:.2f}%")

print(f"Energy: {dyn_energy} mJ")
print(f"Energy: {exact_dyn_energy} mJ")

print(f" & {hardware} & {(1000/float(exact_time)):.1f} & {exact_time:.1f} & {exact_energy/exact_time*1000:.0f} & {exact_energy:.2f}& {exact_energy-exact_dyn_energy:.2f} & {exact_dyn_energy:.2f}  & {exact_energy/gop:.2f} & {exact_energy/mpar:.2f}\\\\")


# %%
