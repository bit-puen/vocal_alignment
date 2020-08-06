import os
import sys
import time
import numpy as np
import librosa
import warnings
import soundfile as sf
import progressbar
import concurrent.futures

from shutil import copyfile
from fastdtw import fastdtw
from argparse import ArgumentParser
from scipy.io.wavfile import write


warnings.filterwarnings('ignore')

# CONFIG_SEQUENCE_TIME_PAIR = {'primary_alignment': 9.5, 
#                              'secondary_alignment': [[9.5, 35.7], [35.7, 70.0]], 
#                              'tertiary_alignment_1st': [[9.5, 17.8], [17.8, 26.7], [26.7, 35.7]], 
#                              'tertiary_alignment_2nd': [[35.7, 40.2], [40.2, 44.5], [44.5, 48.8], [48.8, 57.1], 
#                                                         [57.1, 59.3], [59.3, 61.5], [61.5, 63.8], [63.8, 70.0]]}
# CONFIG_SEQUENCE_TIME_PAIR = {'primary_alignment': 1.0, 
#                              'secondary_alignment': [[1.0, 29.3], [29.3, 63.5]], 
#                              'tertiary_alignment_1st': [[1.0, 11.4], [11.4, 20.6], [20.6, 29.3]], 
#                              'tertiary_alignment_2nd': [[29.3, 33.7], [33.7, 38.1], [38.1, 42.4], [42.4, 50.5], 
#                                                         [50.5, 52.7], [52.7, 54.9], [54.9, 57.7], [57.7, 63.5]]}
# CONFIG_SEQUENCE_TIME_PAIR = {'primary_alignment': 1.0, 
#                              'secondary_alignment': [[1.0, 27.0], [27.0, 61.0]],
#                              'tertiary_alignment_1st': [[1.0, 9.0], [9.0, 18.2], [18.2, 27.0]], 
#                              'tertiary_alignment_2nd': [[27.0, 31.3], [31.3, 35.7], [35.7, 40.0], [40.0, 48.2], 
#                                                         [48.2, 50.4], [50.4, 52.6], [52.6, 55.2], [55.2, 61.0]]}
CONFIG_SEQUENCE_TIME_PAIR = {'primary_alignment': 4.0, 
                             'secondary_alignment': [[4.0, 27.55], [27.55, 59.0]], 
                             'tertiary_alignment_1st': [[4.0, 11.7], [11.7, 19.8], [19.8, 27.55]], 
                             'tertiary_alignment_2nd': [[27.55, 31.5], [31.5, 35.5], [35.5, 39.4], [39.4, 47.0], 
                                                        [47.0, 49.4], [49.4, 51.5], [51.5, 54.0], [54.0, 59.0]]}
CONFIG_THRESHOLD_TERTIARY_1ST = [150, 150, 140]
CONFIG_THRESHOLD_TERTIARY_2ND = [65 , 70, 70, 145, 
                                 40, 27, 30, 60]
                                                        

''' ################################ Utils function ################################ '''


def zscore_normalization(audio_time_series):
    """
    Compute the z score of each value in the sample, relative to the sample mean and standard deviation.
    :param audio_time_series: An array of audio time series
    :return: The audio time series that normalize
    """
    return (audio_time_series - np.mean(audio_time_series)) / np.std(audio_time_series)


def make_them_equal(array_a, array_b, side="back"):
    """
    Make 2 array have the same dimensions by adding zeros.
    :param array_a: An array of audio time series 1
    :param array_b: An array of audio time series 2
    :param side: The side that needs to be added to the center
    :return: array_a and array_b of the same dimension
    """
    len_diff = array_a.shape[0] - array_b.shape[0]
    dummy_ats = np.zeros((abs(len_diff), ))
    if len_diff > 0:
        if side == "back":
            array_b = np.concatenate((array_b, dummy_ats))
        elif side == "front":
            array_b = np.concatenate((dummy_ats, array_b))
        else:
            raise Exception("[ ERROR ] Side error, please select front or back.")
    elif len_diff < 0:
        if side == "back":
            array_a = np.concatenate((array_a, dummy_ats))
        elif side == "front":
            array_a = np.concatenate((dummy_ats, array_a))
        else:
            raise Exception("[ ERROR ] Side error, please select front or back.")
    else:
        pass
    return array_a, array_b


def composite_audio(list_audio_time_series):
    """
    Composite all audios in the list
    :param list_audio_time_series: A list of audio time series
    :return: An audio time series
    """
    if len(list_audio_time_series) <= 1:
        raise Exception('[ ERROR ] Unable to compose these audios.')

    result_ats = list_audio_time_series[0]
    for ats in list_audio_time_series[1:]:
        result_ats, ats = make_them_equal(result_ats, ats)
        result_ats = result_ats + ats
    return result_ats


def get_audio_time_series(directory, file_name_extension, sampling_rate, anchor_file_name=None, logger=True):
    """
    pass
    :param directory:
    :param file_name_extension:
    :param sampling_rate:
    :param anchor_file_name:
    :param logger:
    :return:
    """
    data_path = os.path.abspath(directory)
    bar = None
    count = 0
    anchor = None
    list_query = []
    dummy = np.zeros((int(sampling_rate * CONFIG_SEQUENCE_TIME_PAIR['tertiary_alignment_2nd'][-1][1]), ))

    if '.' not in file_name_extension:
        file_name_extension = '.' + file_name_extension

    list_data_file_name = os.listdir(data_path)
    list_data_file_name = [f for f in list_data_file_name if f.endswith(file_name_extension)]

    if logger:
        bar = progressbar.ProgressBar(maxval=len(list_data_file_name),
                                      widgets=['Reading file...', ' ', progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()

    if file_name_extension == '.npy':
        if anchor_file_name is not None:
            list_data_file_name.remove(anchor_file_name)
            anchor = np.load(os.path.join(data_path, anchor_file_name))
            if logger:
                count += 1
                bar.update(count)
        for data_file_name in list_data_file_name:
            query = np.load(os.path.join(data_path, data_file_name))
            list_query.append(query)
            if logger:
                count += 1
                bar.update(count)
    elif file_name_extension == '.mp3' or file_name_extension == '.wav':
        if anchor_file_name is not None:
            list_data_file_name.remove(anchor_file_name)
            anchor, _ = librosa.load(os.path.join(data_path, anchor_file_name), sr=sampling_rate)
            # anchor = zscore_normalization(anchor)
            anchor = librosa.util.normalize(anchor)
            if logger:
                count += 1
                bar.update(count)
        for data_file_name in list_data_file_name:
            query, _ = librosa.load(os.path.join(data_path, data_file_name), sr=sampling_rate)
            # query = zscore_normalization(query)
            query = librosa.util.normalize(query)
            if np.isnan(query).all():
                list_data_file_name.remove(data_file_name)
            else:
                query, _ = make_them_equal(query, dummy)
                list_query.append(query)
            if logger:
                count += 1
                bar.update(count)
    if logger:
        bar.finish()
    return anchor, list_query, list_data_file_name


def argsort(distance_matrix, dim=2):
    """
    pass
    :param distance_matrix:
    :param dim:
    :return:
    """
    if dim == 1:
        return np.argsort(distance_matrix)
    elif dim == 2:
        index_pair = []
        div = distance_matrix.shape[0]
        sor = np.argsort(distance_matrix.flatten())
        for s in sor:
            y = int(s / div)
            x = s % div
            index_pair.append((y, x))
        return np.asanyarray(index_pair)


def get_unique_index(argsort_matrix):
    """
    pass
    :param argsort_matrix:
    :return:
    """
    if type(argsort_matrix[0]) is tuple:
        unique_index = []
        for index_pair in argsort_matrix:
            for index in index_pair:
                if index not in unique_index:
                    unique_index.append(index)
    else:
        unique_index = argsort_matrix
    return unique_index


def calculate_xcase(data_length, start_index):
    """
    pass
    :param data_length:
    :param start_index:
    :return:
    """
    return int(0.5 * data_length * (data_length - 1 + (2 * start_index)))


def split_data(n_data, n_worker):
    """
    pass
    :param n_data:
    :param n_worker:
    :return:
    """
    n_case = int(n_data * (n_data - 1) / 2)
    n_split = n_case / n_worker

    xcase_old = 0
    start_index = 0
    split_points = []

    i = 0
    while i < n_data:
        xcase = calculate_xcase(i-start_index, start_index)
        if xcase >= n_split:
            if i < 0:
                raise Exception("[ ERROR ] Something went wrong, please try to decrease n_worker.")
            if abs(xcase-n_split) >= abs(xcase_old-n_split):
                i -= 1
            split_points.append(i)
            start_index = i
        xcase_old = xcase
        i += 1

    return split_points


def duplicate_raw_chorus_file(list_file_name, data_path, raw_chorus_path):
    data_path = os.path.abspath(data_path)
    raw_chorus_path = os.path.abspath(raw_chorus_path)
    
    for file_name in list_file_name:
        copyfile(os.path.join(data_path, file_name), os.path.join(raw_chorus_path, file_name))


''' ############################## Alignment function ############################## '''


def process_onset(onset_strength):
    """
    pass
    :param onset_strength:
    :return:
    """
    # normalise the values (zscore)
    onset_strength = zscore_normalization(onset_strength)
    # take any values > 2 standard deviations
    onset_strength = np.where(onset_strength > 2, 1.0, 0.0)
    # add an 'decay' to the values such that we can do a more 'fuzzy' match
    # forward pass
    for i in range(1, len(onset_strength)):
        onset_strength[i] = max(onset_strength[i], onset_strength[i-1] * 0.9)
    # backwards pass
    for i in range(len(onset_strength)-2, 0, -1):
        onset_strength[i] = max(onset_strength[i], onset_strength[i+1] * 0.9)
    
    return onset_strength


def measure_error(src_1, src_2, offset):
    """
    function to measure two waveforms with one offset by a certain amount
    :param src_1:
    :param src_2:
    :param offset:
    :return:
    """
    max_len = min(len(src_1), len(src_2))
    # calculate the mean squared error of the two signals
    diff = src_1[:max_len] - np.roll(src_2[:max_len], offset)
    err = np.sum(diff**2) / len(diff)
    return err


def find_offset(src_1, src_2):
    """
    Find the offset with the lowest error
    :param src_1:
    :param src_2:
    :return:
    """
    offsets = tuple(range(-100, 100))
    errors = [(measure_error(src_1, src_2, offset), offset) for offset in offsets]
    
    error, offset = sorted(errors)[0]
                     
    return offset, error


def align(anchor, query, sampling_rate, algo='mel', threshold=0.65):
    """
    pass
    :param anchor:
    :param query:
    :param sampling_rate:
    :param algo: Algorithm for process onset strength
    :param threshold: Threshold for select shift offset audio time series 
    :return:
    """
    if algo == 'mel':
        onset_anchor = librosa.onset.onset_strength(anchor, sr=sampling_rate)
        onset_query = librosa.onset.onset_strength(query, sr=sampling_rate)
    elif algo == 'cqt':
        cqt_anchor = np.abs(librosa.cqt(y=anchor, sr=sampling_rate))
        cqt_query = np.abs(librosa.cqt(y=query, sr=sampling_rate))
        onset_anchor = librosa.onset.onset_strength(sr=sampling_rate,
                                                    S=librosa.amplitude_to_db(cqt_anchor, ref=np.max),
                                                    n_mels=128)
        onset_query = librosa.onset.onset_strength(sr=sampling_rate,
                                                   S=librosa.amplitude_to_db(cqt_query, ref=np.max),
                                                   n_mels=128)
    else:
        raise Exception("[ ERROR ] Please select algorithm 'mel' or 'cqt'.")
    
    process_anchor = process_onset(onset_anchor)
    process_query = process_onset(onset_query)

    offset, error = find_offset(process_anchor, process_query)
    shift_offset = offset * 512

    if abs(shift_offset / sampling_rate) < threshold:
        result = np.roll(query, shift_offset)
    else:
        result = query

    return result


def compute_alignment(anchor, query, sampling_rate, save_chunk=False, dir_save_chunk=None, chunk_name=None, 
                      threshold1=CONFIG_THRESHOLD_TERTIARY_1ST, threshold2=CONFIG_THRESHOLD_TERTIARY_2ND):
    """
    pass
    :param anchor: An array of anchor audio time series
    :param query: An array of query audio time series
    :param sampling_rate: Sampling rate of the audio time series 
    :param save_chunk:
    :param dir_save_chunk:
    :param chunk_name:
    :return:
    """
    if save_chunk:
        if dir_save_chunk is None or chunk_name is None:
            raise Exception("[ ERROR ] Please select directory to save chunk file.")

    primary_alignment = CONFIG_SEQUENCE_TIME_PAIR['primary_alignment']

    query = query[int(primary_alignment * sampling_rate):]

    # Primary alignment
    primary_query = align(anchor[int(primary_alignment * sampling_rate):],
                          query,
                          sampling_rate=sampling_rate,
                          threshold=3.0)
    
    # Secondary alignment
    secondary_query = []
    for time_pair in CONFIG_SEQUENCE_TIME_PAIR['secondary_alignment']:
        q_start = int((time_pair[0] - primary_alignment) * sampling_rate)
        q_stop = int((time_pair[1] - primary_alignment) * sampling_rate)
        tmp = align(anchor[int(time_pair[0] * sampling_rate): int(time_pair[1] * sampling_rate)], 
                    primary_query[q_start: q_stop],
                    sampling_rate=sampling_rate)
        secondary_query.append(tmp)
    
    # Tertiary alignment (1st)
    tertiary_query_1st = []
    for i, time_pair in enumerate(CONFIG_SEQUENCE_TIME_PAIR['tertiary_alignment_1st']):
        q_start = int((time_pair[0] - primary_alignment) * sampling_rate)
        q_stop = int((time_pair[1] - primary_alignment) * sampling_rate)
        tmp_anchor = anchor[int(time_pair[0] * sampling_rate): int(time_pair[1] * sampling_rate)]
        tmp_query = primary_query[q_start: q_stop]
        tmp = align(tmp_anchor, 
                    tmp_query, 
                    sampling_rate=sampling_rate)
        distance = compare_distance(tmp_anchor, tmp_query, sampling_rate, input_is_feature=False)
        if distance <= threshold1[i]:
            tertiary_query_1st.append(tmp)
        else:
            tertiary_query_1st.append(np.zeros((tmp.shape[0])))
        if save_chunk:
            save_as = os.path.join(os.path.realpath(dir_save_chunk), )
            librosa.output.write_wav(save_as, tmp, sampling_rate)

    # Tertiary alignment (2nd)
    tertiary_query_2nd = []
    query_time_factor = CONFIG_SEQUENCE_TIME_PAIR['tertiary_alignment_2nd'][0][0]
    for i, time_pair in enumerate(CONFIG_SEQUENCE_TIME_PAIR['tertiary_alignment_2nd']):
        if time_pair == CONFIG_SEQUENCE_TIME_PAIR['tertiary_alignment_2nd'][-1]:
            q_start = int((time_pair[0] - query_time_factor) * sampling_rate)
            q_stop = int((time_pair[1] - query_time_factor) * sampling_rate)
            tmp_anchor = anchor[int(time_pair[0] * sampling_rate): int(time_pair[1] * sampling_rate)]
            tmp_query = secondary_query[1][q_start: q_stop]
            tmp = align(tmp_anchor,
                        tmp_query,
                        sampling_rate=sampling_rate, 
                        algo='cqt')
            distance = compare_distance(tmp_anchor, tmp_query, sampling_rate, input_is_feature=False)
        else:
            q_start = int((time_pair[0] - query_time_factor) * sampling_rate)
            q_stop = int((time_pair[1] - query_time_factor) * sampling_rate)
            tmp_anchor = anchor[int(time_pair[0] * sampling_rate): int(time_pair[1] * sampling_rate)]
            tmp_query = secondary_query[1][q_start: q_stop]
            tmp = align(tmp_anchor, 
                        tmp_query,
                        sampling_rate=sampling_rate)
            distance = compare_distance(tmp_anchor, tmp_query, sampling_rate, input_is_feature=False)
            
        # if time_pair[0] == 39.5:
        #     print(time_pair, ' || ', distance)
        if distance <= threshold2[i]:
            tertiary_query_2nd.append(tmp)
        else:
            tertiary_query_2nd.append(np.zeros((tmp.shape[0])))
        if save_chunk:
            save_as = os.path.realpath(dir_save_chunk)
            librosa.output.write_wav(save_as, tmp, sampling_rate)

    # Concatenate Tertiary alignment (1st)
    aligned_tertiary_1st = tertiary_query_1st[0]
    for alignment in tertiary_query_1st[1:]:
        aligned_tertiary_1st = np.concatenate((aligned_tertiary_1st, alignment))

    # Concatenate Tertiary alignment (2st)
    aligned_tertiary_2nd = tertiary_query_2nd[0]
    for alignment in tertiary_query_2nd[1:]:
        aligned_tertiary_2nd = np.concatenate((aligned_tertiary_2nd, alignment))

    # Concatenate secondary alignment
    alignment_secondary = np.concatenate((aligned_tertiary_1st, aligned_tertiary_2nd))

    # Concatenate primary alignment
    dummy_alignment = np.zeros((int(primary_alignment * sampling_rate), ))
    primary_alignment = np.concatenate((dummy_alignment, alignment_secondary))

    return primary_alignment


''' ############################ DTW distance function ############################# '''


def compare_distance(anchor, query, sampling_rate, input_is_feature=False):
    """
    pass
    :param anchor:
    :param query:
    :param sampling_rate:
    :param input_is_feature:
    :return:
    """
    if not input_is_feature:
        # Extract anchor feature
        anchor = librosa.onset.onset_strength(anchor, sampling_rate)
    # for i, query in enumerate(list_query):
    if not input_is_feature:
        query = librosa.onset.onset_strength(query, sampling_rate)
    # Dynamic time warping distance
    distance, _ = fastdtw(anchor, query)
    distance = int(distance)
    
    return distance


def get_distance_matrix(anchor, query, sampling_rate, mode='ofa', logger=True, thread=None, anchor_begin=None):
    """
    pass
    :param anchor: An array of 
    :param query:
    :param sampling_rate:
    :param mode:
    :param logger:
    :param thread:
    :param anchor_begin:
    :return:
    """
    bar = None
    count = 0
    distance_matrix = []

    if logger:
        bar = progressbar.ProgressBar(maxval=100,
                                      widgets=['thread number: ', str(thread), ' ', progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()

    if mode == "ffa":
        if anchor_begin is None and thread is not None:
            raise Exception("[ ERROR ] Please add anchor_begin")
        
        max_index = anchor_begin + len(anchor) - 1
        size_of_data = int((max_index + anchor_begin) * (max_index - anchor_begin + 1) / 2)
        print("[ REPORT ] Thread number {} computes distance matrix in mode {} with data size {}.".format(thread,
                                                                                                          mode,
                                                                                                          size_of_data))
        for i, anc in enumerate(anchor):
            dist_matrix = []
            # Extract anchor's feature
            onset_anchor = librosa.onset.onset_strength(anc, sr=sampling_rate)
            if anchor_begin is not None:
                i = i + anchor_begin
            for j, qur in enumerate(query):
                if j >= i:
                    dist_matrix.append(np.inf)
                    continue
                # Extract query's feature
                onset_query = librosa.onset.onset_strength(qur, sr=sampling_rate)
                # Compare distance
                distance = compare_distance(onset_anchor, onset_query, sampling_rate, input_is_feature=True)
                dist_matrix.append(distance)
                if logger:
                    count += 1
                    percent = int((count / size_of_data) * 100)
                    bar.update(percent)

            distance_matrix.append(dist_matrix)

    elif mode == "ofa":
        size_of_data = len(query)
        print("[ REPORT ] Thread number {} computes distance matrix in mode {} with data size {}.".format(thread,
                                                                                                          mode,
                                                                                                          size_of_data))
        # Extract anchor's feature
        onset_anchor = librosa.onset.onset_strength(anchor, sr=sampling_rate)
        for i, qur in enumerate(query):
            # Extract query's feature
            onset_query = librosa.onset.onset_strength(qur, sr=sampling_rate)
            # Compare distance
            distance = compare_distance(onset_anchor, onset_query, sampling_rate, input_is_feature=True)
            distance_matrix.append(distance)
            if logger:
                count += 1
                percent = int((count / size_of_data) * 100)
                bar.update(percent)
    if logger:
        bar.finish()
    return np.asanyarray(distance_matrix)


''' ##################################### MAIN ##################################### '''


def main(args):
    t_start = time.time()

    # Read file
    anchor, list_query, list_query_name = get_audio_time_series(args.data_path, args.endswith,
                                                                args.sampling_rate, args.anchor_name)
    elapsed_time_read_file = time.time() - t_start
    if args.endswith != '.npy' and args.save_npy:
        save_as = os.path.abspath(os.path.join(os.path.join(args.data_path, 'npy'), 'anchor.npy'))
        np.save(save_as, anchor)
        for i, query in enumerate(list_query):
            file_name = list_query_name[i].split('.')[0] + '.npy'
            save_as = os.path.abspath(os.path.join(os.path.join(args.data_path, 'npy'), file_name))
            np.save(save_as, query)
        print("[ REPORT ] Save pre-process files.")

    n_data = len(list_query)
    print("[ REPORT ] Read all data file {}  files: {} s (for ffa mode, all case: {})".format(n_data,
                                                                                              elapsed_time_read_file,
                                                                                              int(n_data*(n_data-1)/2)))

    # Process dynamic time warping search
    t_start_search = time.time()

    results = []
    results_tmp = {}
    argsort_refinement = []
    distance_matrix = np.array([])

    if args.mode == 'ofa':
        n_separate = int(n_data / args.number_workers)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(args.number_workers):
                if i == args.number_workers - 1:
                    result = executor.submit(get_distance_matrix,
                                             anchor, list_query[n_separate * i:],
                                             args.sampling_rate, args.mode,
                                             True, i + 1, n_separate * i)
                else:
                    result = executor.submit(get_distance_matrix,
                                             anchor, list_query[n_separate * i: n_separate * (i + 1)],
                                             args.sampling_rate, args.mode,
                                             True, i + 1, n_separate * i)
                results_tmp[str(i)] = result
            # Refinement
            for i in range(len(results_tmp)):
                results.append(results_tmp[str(i)])
            for i, f in enumerate(concurrent.futures.as_completed(results)):
                if i == 0:
                    distance_matrix = f.result()
                else:
                    distance_matrix = np.concatenate((distance_matrix, f.result()))
        # Sort
        argsort_result = argsort(distance_matrix, 1)
        for res in argsort_result:
            if distance_matrix[res] <= args.similar_threshold:
                argsort_refinement.append(res)
    elif args.mode == 'ffa':
        n_separate = split_data(n_data, args.number_workers)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, n_s in enumerate(n_separate):
                if i == 0:
                    result = executor.submit(get_distance_matrix,
                                             list_query[:n_s], list_query,
                                             args.sampling_rate, args.mode,
                                             True, i + 1, 0)
                else:
                    result = executor.submit(get_distance_matrix,
                                             list_query[o_n_s:n_s],
                                             list_query,
                                             args.sampling_rate, args.mode,
                                             True, i + 1, o_n_s)
                results_tmp[str(i)] = result
                o_n_s = n_s
            else:
                result = executor.submit(get_distance_matrix,
                                         list_query[n_s:],
                                         list_query,
                                         args.sampling_rate, args.mode,
                                         True, i + 2, n_s)
                results_tmp[str(i + 1)] = result
            # Refinement
            for i in range(len(results_tmp)):
                results.append(results_tmp[str(i)])
            for i, f in enumerate(concurrent.futures.as_completed(results)):
                if i == 0:
                    distance_matrix = f.result()
                else:
                    distance_matrix = np.concatenate((distance_matrix, f.result()))
        # Sort
        argsort_result = argsort(distance_matrix, 2)
        for res in argsort_result:
            if distance_matrix[res[0]][res[1]] <= args.similar_threshold:
                argsort_refinement.append(res)
    else:
        raise Exception("[ ERROR ] Please select mode 'ffa' or 'ofa'")

    print("[ REPORT ] Comparison process time: {} sec.".format(time.time() - t_start_search))

    print(argsort_refinement)

    # Find unique index
    unique_index = get_unique_index(argsort_refinement)
    print("[ REPORT ] Number of Chorus: ", len(unique_index))

    tmp_list_raw_chorus_file_name = []
    for res in unique_index:
        tmp_list_raw_chorus_file_name.append(list_query_name[res])
    duplicate_raw_chorus_file(tmp_list_raw_chorus_file_name, args.data_path, args.raw_chorus_path)

    print()
    for res in unique_index:
        print(distance_matrix[res] , ' || ', list_query_name[res])
    # for i, res in enumerate(argsort_refinement):
    #     print(distance_matrix[res[0]][res[1]], ' || ', list_query_name[res[0]], ' || ', list_query_name[res[1]])
    print()




    # Apply alignment algorithm and composite the result with anchor
    bar = progressbar.ProgressBar(maxval=len(unique_index),
                                  widgets=['Align process...', ' ', progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage()]).start()
    t_start_alignment = time.time()
    result_alignment = np.zeros((100, ))
    for i, idx in enumerate(unique_index):
        res = compute_alignment(anchor, list_query[idx], sampling_rate=args.sampling_rate)
        result_alignment = composite_audio([result_alignment, res * 0.5])
        bar.update(i)
    bar.finish()
    print("[ REPORT ] Alignment process time: {} s".format(time.time() - t_start_alignment))

    # Save result as .wav
    save_as = os.path.abspath(os.path.join(args.output_path, args.output_name))
    # librosa.output.write_wav(save_as, result_alignment, sr=args.sampling_rate)
    sf.write(save_as, result_alignment, args.sampling_rate, format="wav")
    # write(save_as, args.sampling_rate, result_alignment)
    print("[ REPORT ] Save output as ", save_as)
    print("[ REPORT ] Total process time: {} s".format(time.time() - t_start))


def parse_arguments(argv):
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/production',
                        help="test")
    parser.add_argument('--endswith', type=str, default='.mp3',
                        help="test")
    parser.add_argument('--output_name', type=str, default='aligned.wav',
                        help="test")
    parser.add_argument('--anchor_name', type=str, default='anchor.mp3',
                        help="test")
    parser.add_argument('--output_path', type=str, default='out',
                        help="test")
    parser.add_argument('--number_workers', type=int, default=8,
                        help="test")
    parser.add_argument('--sampling_rate', type=int, default=16000,
                        help="test")
    parser.add_argument('--similar_threshold', type=int, default=1350,
                        help="test")
    parser.add_argument('--mode', type=str, default='ofa',
                        help="test")
    parser.add_argument('--save_npy', type=bool, default=True,
                        help="test")
    parser.add_argument('--raw_chorus_path', type=str, default='data/raw_chorus',
                        help="test")
                        
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
