# report result parser
# extracts layer execution times from profile files generated in DNNDK

import os
import pickle

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

import pandas

def add_measured_to_input(time_df, input_df, measured_df):
    """adds info of measuredment dataframe to time and layer exection dataframe

    Args:
        time_df: DataFrame with input layer names as columns, stores runtime
        input_df: DataFrame with input layes names as columns, stores execution status
        measured_df: DataFrame with measured layer names as rows, contains one measurement

    Returns: 
        time_df and input_df
    """
    #filter only available Layers
    
    loc = len(input_df)
    input_df = input_df.append(pandas.Series(), ignore_index=True)
    time_df = time_df.append(pandas.Series(), ignore_index=True)
    missing = {}
    for c in input_df.columns:
        if c in measured_df['LayerName'].values:
            row = measured_df.loc[measured_df['LayerName'] == c]
            input_df.at[loc, c] = 'EXECUTED'
            time_df.at[loc, c] = row['RunTime(ms)'].values[0]
            measured_df = measured_df.loc[measured_df['LayerName'] != c]
        else:
            missing[c]=0

# Additionally measured
    print(measured_df)
# Look for missing Layers

    print("\nMissing Layers:\n")
    for c in missing.keys():
        missing[c] = measured_df['LayerName'].str.contains(c+'_').sum() 

    cycle = len(missing)
    for c in range(cycle):
        v = min(missing, key=missing.get)
        print('layer: ',v,' found ', missing[v], ' time(s)')
        #time_df.at[loc.v] = measured_df[measured_df['LayerName'].str.contains(c+'_')]['RunTime(ms)'].sum() 
        if(missing[v]) == 0:
            input_df.at[loc, v] = 'REMOVED'
            del missing[v]
        else:
            #print(measured_df[measured_df['LayerName'].str.contains(v+'_')])
            time_sum = measured_df[measured_df['LayerName'].str.contains(v+'_')]['RunTime(ms)'].sum()
            input_df.at[loc, v] = 'EXECUTED'
            time_df.at[loc, v] = time_sum 
            print('Time_sum', time_sum)
            measured_df = measured_df[~measured_df['LayerName'].str.contains(v+'_')]
            del missing[v]

    return time_df, input_df

def extract_data_from_profile(infold, outfold, result, profile_list):
    """Reads file in a pandas dataframe and writes layer data into a pickle file

    Args:
        infold: folder where the reports are contained
        outfold: folder where the pickled data will be stored
        result: filename of the result of which the profile name derives from where the data will be extracted
        profile_list: list of all .prof files in infold directory

    Returns: none

    """

    print(result)
    # get unique identifier for the profile name from result name
    #profile_new = "_".join(result.split("result_")[1].split(".txt")[0].split("_")[:5])
    profile_new = result.replace("result_","").replace(".txt",".prof")

    # find the appropriate filename in the profile file list
    for prof in profile_list:
        if profile_new in prof:
            profile_new = prof

    print(profile_new)
    # open file and extract data
    try:
        profile_file = open(os.path.join(infold, profile_new), "r")
    except:
        return None
    data = profile_file.readlines()
    profile_file.close()
    #[print(dat) for dat in data]

    # declare variables for metrics from profile
    beg_timestamp, end_timestamp = 0, 0
    rem_stamp, interlayer_time_sum = 0, 0
    interlayer_time = 0
    interlayer_time_list = []
    layer_time_list= []

    for i, data_line in enumerate(data):
        if i == 0:
            # model name on every line at position 1
            #print(data_line.split("\t")[1])
            beg_timestamp = int(data_line.split("\t")[2])
            interlayer_time_list.append(0)
        else:
            # calculate the complete interlayer time from timestamps
            #print(i, int(data_line.split("\t")[2]) - rem_stamp)
            interlayer_time = int(data_line.split("\t")[2]) - rem_stamp
            interlayer_time_sum += interlayer_time
            interlayer_time_list.append(interlayer_time/1000) # add the interlayer time to the interlayer time list

        # remember the timestamp from previous line to calculate interlayer time
        begin = int(data_line.split("\t")[2])
        rem_stamp = int(data_line.split("\t")[3])
        print(begin,rem_stamp)
        layer_time_list.append((rem_stamp-begin)/1000) # add the interlayer time to the interlayer time list

    end_timestamp = int(data_line.split("\t")[3])
    overall_time = (end_timestamp - beg_timestamp)/1000
    interlayer_time_sum = interlayer_time_sum/1000
    print("overall time from timestamps: ", overall_time, "ms")
    print("interlayer time: ", interlayer_time_sum, "ms")
    #[print(intl) for intl in interlayer_time_list]

    # write the us data for the model execution time and interlayer time into the same pickle as the result data
    try:
        pickle_file_path = os.path.join(outfold, result.split("result_")[1].split(".txt")[0] + ".p")
    except:
        pickle_file_path = os.path.join(outfold, result.split(".txt")[0] + ".p")

    pickle_file = open(pickle_file_path, "rb")

    stored_data = pickle.load(pickle_file)["layer"]
    interlayer_time_list = pandas.DataFrame(interlayer_time_list)
    layer_time_list = pandas.DataFrame(layer_time_list)
    stored_data["InterlayerTime(ms)"] = interlayer_time_list
    stored_data["RunTime(ms)"] = layer_time_list
    pickle_file.close()

    # add the dictionary entry of the total runtime and total interlayer time
    pickle_file = open(pickle_file_path, "wb")
    pickle.dump({"layer": stored_data}, pickle_file)
    pickle_file.close()

    # add the dictionary entry of the total runtime and total interlayer time
    pickle_file = open(pickle_file_path, "ab+")
    pickle.dump({"total": ({"overall_time(ms)":overall_time}, {"interlayer_time_sum(ms)":interlayer_time_sum})}, pickle_file)
    pickle_file.close()
    
    
    return data


def extract_data_from_result(infold, outfold, result):
    """Data structure is as follows
    ID NodeName Workload(MOP) Mem(MB) RunTime(ms) Perf(GOPS) Utilization MB/S
    Args:
        infold: folder which contains the results from the dnndk bench
        outfold: folder where the extracted results in pickle format will be saved
        result: name of the result file

    Returns: pandas

    """
    data = []
    try:
        file = open(os.path.join(infold, result), "r")
    except:
        return None
    temp = open(os.path.join(outfold, "temp"), "w")

    for i, line in enumerate(file.readlines()):
        if i > 1:
            # Total in an element breaks the parsing, because everything after Total in unnecessary
            if "Total" in line:
                break

            temp_str = ""
            for j, elem in enumerate(line.split()):
                # change the name of the Utilization column
                if "Utilization" in elem:
                    elem = "Utilization(%)"
                # change NodeName to LayerName (as in the NCS2 bench reports)
                elif "NodeName" in elem:
                    elem = "LayerName"
                # % in the element indicates the utilization line, remove it
                elif "%" in str(elem):
                    elem = elem.split("%")[0]

                # do not add a delimiter in the beginning and the end
                if j == 0:
                    temp_str += str(elem)
                else:
                    temp_str += ";" + str(elem)

            # begin new line
            temp.write(temp_str + "\n")

    # close files
    file.close()
    temp.close()

    # open the generated file and read into a pandas dataframe
    temp = open(os.path.join(outfold, "temp"), "rb")
    data = pandas.read_csv(temp, sep=";")

    # delete the ID  column from data because of pandas redundancy
    del data["ID"]

    # close and delete temporary file
    temp.close()
    os.remove(os.path.join(outfold, "temp"))

    if outfold:
        # pickle the data to the NCS2 report extracted data if existing
        try:
            t = result.split("result_")[1].split(".txt")[0]
        except:
            t = result.split(".txt")[0]
        pickle_file = open(os.path.join(outfold,  t + ".p"), "wb")
        pickle.dump({"layer":data}, pickle_file)
        pickle_file.close()
    
    return data

def extract_data_from_folder(infold, outfold):
    """Extracts layer name and real time data from a folder of ncs2 reports

    Args:
        infold: folder containing the reports generated by benchmark_app.py
        outfold: folder where the extracted results will be saved

    Returns: none

    """

    # if the output data file does not exist, create it
    if not os.path.isdir(outfold):
        os.mkdir(outfold)

    print("Parsing data from " + infold + " following files found")

    # prof contains the model name and execution time stamps in us, as well as the layer execution time (end - beg)
    profile_list = [f for f in os.listdir(infold) if ".prof" in f]
    #[print(f) for f in profile_list]

    # result contains the model name, execution time in ms, node name (layer name?) and other parameters
    result = [f for f in os.listdir(infold) if "result" in f]
    #[print(f) for f in result]


    for i, res in enumerate(result):
        print(i + 1, ": parsing " + res)
        extract_data_from_result(infold, outfold, res)
        extract_data_from_profile(infold, outfold, res, profile_list)
        
def extract_data_from_file(infold, outfold, name):
    """Extracts layer name and real time data from a folder of ncs2 reports

    Args:
        infold: folder containing the reports generated by benchmark_app.py
        outfold: folder where the extracted results will be saved

    Returns: none

    """

    # if the output data file does not exist, create it
    if not os.path.isdir(outfold):
        os.mkdir(outfold)

    print("Parsing data from " + infold + " following files found")
    profile_list = [f for f in os.listdir(infold) if ".prof" in f]

    print("parsing " + name)
    data = extract_data_from_result(infold, outfold, name)
    extract_data_from_profile(infold, outfold, name, profile_list)

    return data
