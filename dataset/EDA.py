from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def find_sequence_len():
    file_name = './nli_for_simcse.csv'
    data = pd.read_csv(file_name)
    print(data)
    data_list = []
    for d in tqdm(data.iterrows()):
        da = list(d[1])
        for item in da:
            data_list.append(item)
    print(len(data_list))
    len_list = []
    for item in data_list:
        len_list.append(len(item))

    len_q = 0
    for item in len_list:
        if item <= 128:
            len_q += 1
    print(len_q/len(len_list))

    plt.hist(len_list, bins=50)
    plt.show()


# with open('./nli_for_simcse_trial.csv', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
# lines = lines[:5]
# with open('./test_dataloader.csv', 'w', encoding='utf-8') as f:
#     f.writelines(lines)
