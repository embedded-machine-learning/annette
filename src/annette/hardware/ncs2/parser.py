# report result parser
# extracts layer execution times from reports generated by the openvino
# benchmark_app.py

import os
import pickle
import pandas
import argparse
import json

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def add_measured_to_input(time_df, input_df, measured_df):
    """adds info of measurement dataframe to time and layer execution dataframe

    Args:
        time_df: DataFrame with input layer names as columns, stores runtime
        input_df: DataFrame with input layes names as columns, stores execution status
        measured_df: DataFrame with measured layer names as rows, contains one measurement

    Returns:
        time_df and input_df
    """
    # filter only available Layers

    loc = len(input_df)
    input_df = input_df.append(pandas.Series(), ignore_index=True)
    time_df = time_df.append(pandas.Series(), ignore_index=True)
    missing = {}
    for c in input_df.columns:
        if c in measured_df['LayerName'].values:
            row = measured_df.loc[measured_df['LayerName'] == c]
            input_df.at[loc, c] = row['ExecStatus'].values[0]
            time_df.at[loc, c] = row['RunTime(ms)'].values[0]
            measured_df = measured_df.loc[measured_df['LayerName'] != c]
        else:
            missing[c] = 0
    # Input Layer treated separately
        if c == 'x':
            row = measured_df.loc[measured_df['LayerName'] == '<Extra>']
            input_df.at[loc, c] = row['ExecStatus'].values[0]
            time_df.at[loc, c] = row['RunTime(ms)'].values[0]
            measured_df = measured_df.loc[measured_df['LayerName'] != '<Extra>']

    # Look for missing Layers
    print("\nMissing Layers:\n")
    print(measured_df)
    for c in missing.keys():
        missing[c] = measured_df['LayerName'].str.contains(c+'_').sum()

    cycle = len(missing)
    for c in range(cycle):
        v = min(missing, key=missing.get)
        print('layer: ', v, ' found ', missing[v], ' time(s)')
        #time_df.at[loc.v] = measured_df[measured_df['LayerName'].str.contains(c+'_')]['RunTime(ms)'].sum()
        if(missing[v]) == 0:
            input_df.at[loc, v] = 'REMOVED'
            del missing[v]
        else:
            # print(measured_df[measured_df['LayerName'].str.contains(v+'_')])
            time_sum = measured_df[measured_df['LayerName'].str.contains(
                v+'_')]['RunTime(ms)'].sum()
            input_df.at[loc, v] = 'EXECUTED'
            time_df.at[loc, v] = time_sum
            print('Time_sum', time_sum)
            measured_df = measured_df[~measured_df['LayerName'].str.contains(
                v+'_')]
            del missing[v]

    return time_df, input_df


def extract_data_from_ncs2_report(infold, outfold, report, format="pickle"):
    """Reads file in a pandas dataframe and writes layer data into a pickle file

    Args:
        infold: folder where the reports are contained
        outfold: folder where the pickled data will be stored
        report: filename of the report where the data will be extracted
        format: data format to save the data with - either pickle or json

    Returns: none
    """
    try:
        filename = os.path.join(infold, report)
        print(filename)
        data = pandas.read_csv(filename, sep=";")
        # rename the column names for better readability and understanding
        data.columns = ["LayerName", "ExecStatus", "LayerType",
                        "ExecType", "RunTime(ms)", "CpuTime(ms)"]

        # delete the last entry (total runtime)
        data.drop(data.tail(1).index, inplace=True)

        # change all "/" and "-" to "_" in LayerName
        for i, elem in enumerate(data["LayerName"]):
            if "/" in elem:
                elem = elem.replace("/", "_")
            elif "-" in elem:
                elem = elem.replace("-", "_")
            data["LayerName"][i] = elem
    except:
        print("File not found")
        data = None

    # construct the pickle file name from the report name (without sync/async mode in name)
    try:
        outfile = "_".join(report.split("benchmark_average_counters_report_")[
                           1].split("_")[:-1]).split(".csv")[0]
    except:
        outfile = report.split(".csv")[0]
        print(report)

    if format == "pickle":
        # open a new file and write extracted and modified data using pickle
        with open(os.path.join(outfold, outfile + ".p"), "wb") as out_f:
            pickle.dump(data, out_f)
    elif format == "json":
        with open(os.path.join(outfold, outfile + ".json"), "wb") as out_f:
            json.dump(data, out_f)
    else:
        print("Invalid format:", format, " passed!")

    return data


def extract_data_from_folder(infold, outfold):
    """Extracts layer name and real time data from a folder of ncs2 reports

    Args:
        infold: folder containing the reports generated by benchmark_app.py
        outfold: folder where the extracted results will be saved

    Returns: none

    """

    # if the output data directory does not exist, create it
    if not os.path.isdir(FLAGS.outfold):
        os.mkdir(FLAGS.outfold)

    # avg_count holds the csv filenames of the benchmark_average_counters containing layer data
    avg_count = [f for f in os.listdir(
        infold) if "benchmark_average_counters_report" in f]

    #print("Parsing data from " + infold + ", following models found:")
    #[print(f.split("benchmark_average_counters_report_")[1].split("_sync.csv")[0]) for f in avg_count]

    # go over files and extract data
    for i, report in enumerate(avg_count):
        print(i + 1, ": parsing " + report)
        extract_data_from_ncs2_report(infold, outfold, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-if", '--infold', default='./report',
                        help='Folder containing reports', required=True)
    parser.add_argument("-of", '--outfold', default='report_sync_extracted',
                        help='folder which will contain the output pickle files', required=True)
    args = parser.parse_args()

    extract_data_from_folder(args.infold, args.outfold)