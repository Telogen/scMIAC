import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def get_benchmark(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_recall = recall_score(true_labels, pred_labels, average='macro')
    avg_precision = precision_score(true_labels, pred_labels, average='macro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    out = {
        'accuracy': round(accuracy, 4),
        'average_recall': round(avg_recall, 4),
        'average_precision': round(avg_precision, 4),
        'macro_f1': round(macro_f1, 4)
    }
    return out



def get_each_recall(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    classes = np.unique(true_labels)
    recall_data = []

    for cls in classes:
        true_positives = np.sum((true_labels == cls) & (pred_labels == cls))
        true_count = np.sum(true_labels == cls)
        recall = true_positives / true_count if true_count != 0 else np.nan
        recall_data.append([true_positives, true_count, recall])

    # Overall recall
    overall_true_positives = np.sum(true_labels == pred_labels)
    overall_true_count = len(true_labels)
    overall_recall = overall_true_positives / overall_true_count

    recall_data.append([overall_true_positives, overall_true_count, overall_recall])
    recall_df = pd.DataFrame(recall_data, index=np.append(classes, 'all'), columns=['True Positives', 'True Count', 'Recall'])

    return recall_df



def get_each_precision(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    classes = np.unique(pred_labels)
    precision_data = []

    for cls in classes:
        true_positives = np.sum((true_labels == cls) & (pred_labels == cls))
        pred_count = np.sum(pred_labels == cls)
        precision = true_positives / pred_count if pred_count != 0 else 0
        precision_data.append([true_positives, pred_count, precision])

    # Overall precision
    overall_true_positives = np.sum(true_labels == pred_labels)
    overall_pred_count = len(pred_labels)
    overall_precision = overall_true_positives / overall_pred_count

    precision_data.append([overall_true_positives, overall_pred_count, overall_precision])
    precision_df = pd.DataFrame(precision_data, index=np.append(classes, 'all'), columns=['True Positives', 'Pred Count', 'Precision'])

    return precision_df






def get_merged_labels_dataset_ISSAACseq(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "R0 Ex-L2/3 IT": "R0",
        "R1 Ex-L2/3 IT Act": "R1",
        "R10 Ex-L6b": "R10",
        "R11 Ex-PIR Ndst4": "R11",
        "R13 In-Drd2": "R13",
        "R14 In-Hap1": "R14",
        "R15 In-Pvalb": "R15",
        "R16 In-Sst": "R16",
        "R17 In-Tac1": "R17",
        "R18 In-Vip/Lamp5": "R18",
        "R19 Astro": "R19",
        "R2 Ex-L4 IT": "R2",
        "R20 OPC": "R20",
        "R21 Oligo": "R21",
        "R22 VLMC": "R22",
        "R3 Ex-L5 IT": "R3",
        "R4 Ex-L5 NP": "R4",
        "R5 Ex-L5 NP Cxcl14": "R5",
        "R6 Ex-L5-PT": "R6",
        "R7 Ex-L6 CT": "R7",
        "R8 Ex-L6 IT Bmp3": "R8",
        "R9 Ex-L6 IT Oprk1": "R9"
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_dataset_SHAREseq(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "ahighCD34+ bulge": 'HCD34B',
        "alowCD34+ bulge": 'LCD34B',
        "Basal": 'Bas',
        "Dermal Fibroblast": 'DF',
        "Dermal Papilla": 'DP',
        "Dermal Sheath": 'DS',
        "Endothelial": 'Endo',
        "Granular": 'Gran',
        "Hair Shaft-cuticle.cortex": 'HSCC',
        "Infundibulum": 'Infu',
        "IRS": 'IRS',
        "Isthmus": 'Isth',
        "K6+ Bulge Companion Layer": 'KBCL',
        "Macrophage DC": 'MDC',
        "Medulla": 'Medu',
        "Melanocyte": 'Mela',
        "ORS": 'ORS',
        "Schwann Cell": 'SC',
        "Sebaceous Gland": 'SG',
        "Spinous": 'Spin',
        "TAC-1": 'TAC1',
        "TAC-2": 'TAC2'
    }
    return [label_map.get(label, label) for label in labels]



def get_merged_labels_Kidney(labels):
    labels = [str(label) for label in labels]
    label_map = {
        'PCT': 'PT',
        'PST': 'PT',
        'DCT1': 'DCT',
        'DCT2': 'DCT',
        'MES': 'MES_FIB',
        'FIB': 'MES_FIB'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Zhu(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'Mono', 
        "CD14 Mono": 'Mono', 
        "Monocytes": 'Mono',
        "cDC": 'DC', 
        "pDC": 'DC', 
        "DCs": 'DC',
        "CD4 Naive": 'NaiveT', 
        "CD8 Naive": 'NaiveT', 
        "Naive T cells": 'NaiveT',
        "CD4 TCM": 'CD4T', 
        "Treg": 'CD4T', 
        "CD4 TEM": 'CD4T', 
        "Activated CD4 T cells": 'CD4T',
        "CD8 TEM_2": 'CD8T', 
        "CD8 TEM_1": 'CD8T', 
        "Cytotoxic CD8 T cells": 'CD8T',
        "NK": 'ILC', 
        "NKs": 'ILC', 
        "XCL+ NKs": 'ILC',
        "Memory B": 'MemB', 
        "Memory B cells": 'MemB', 
        "Intermediate B": 'MemB',
        "Naive B": 'NaiveB', 
        "Naive B cells": 'NaiveB',
        "Plasma": 'Plasma', 
        "Cycling Plasma": 'Plasma',
        "Cycling T cells": 'CycT',
        "Megakaryocytes": 'Mega',
        "Stem cells": 'HSPC'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Wilk(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "CD16 Monocyte": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "CD14 Monocyte": 'CD14Mono',
        "cDC": 'cDC', "DC": 'cDC',
        "NK": 'ILC',
        "CD4 Naive": 'NaiveT', "CD8 Naive": 'NaiveT', "CD4n T": 'NaiveT',
        "CD4 TCM": 'CD4T', "Treg": 'CD4T', "CD4 TEM": 'CD4T', "CD4m T": 'CD4T', "CD4 T": 'CD4T',
        "CD8m T": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T', "MAIT": 'CD8T', "CD8eff T": 'CD8T',
        "gdT": 'gdT', "gd T": 'gdT',
        "Intermediate B": 'B', "Memory B": 'B', "Naive B": 'B', "B": 'B',
        "Plasmablast": 'Plasma', "Plasma": 'Plasma',
        "SC & Eosinophil": 'HSPC', "HSPC": 'HSPC',
        "Granulocyte": 'Granu'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Stephenson(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "CD16.mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "CD14.mono": 'CD14Mono',
        "cDC": 'cDC', "DC": 'cDC',
        "CD4 Naive": 'CD4NT', "CD4.Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT', "CD8.Naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T', "CD4.CM": 'CD4T', "CD4.IL22": 'CD4T',
        "CD4.Th": 'CD4T', "CD4.EM": 'CD4T', "CD4.Tfh": 'CD4T',
        "CD8.TE": 'CD8T', "CD8.EM": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "Intermediate B": 'MemB', "Memory B": 'MemB', "B_non-switched_memory": 'MemB',
        "B_switched_memory": 'MemB', "B_exhausted": 'MemB',
        "B_naive": 'NaiveB', "Naive B": 'NaiveB', "B_immature": 'NaiveB',
        "HSC": 'HSPC', "HSPC": 'HSPC',
        "NK": 'ILC', "ILC": 'ILC',
        "Lymph.prolif": 'Prolif',
        "Platelets": 'Platelet'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Hao(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono',
        "CD4 Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT',
        "CD4 CTL": 'CD4CTL',
        "CD8 TEM": 'CD8T', "CD8 TCM": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "CD4 TEM": 'CD4T', "CD4 TCM": 'CD4T', "CD4 CTL": 'CD4T',
        "Intermediate B": 'InterB', "B intermediate": 'InterB',
        "Memory B": 'MemB', "B memory": 'MemB',
        "Naive B": 'NaiveB', "B naive": 'NaiveB',
        "Plasmablast": 'Plasma', "Plasma": 'Plasma',
        "NK": 'ILC', "ILC": 'ILC',
        "pDC": 'pDC', "ASDC": 'pDC',
        "Proliferating": 'Prolif'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Monaco(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "NC_mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "C_mono": 'CD14Mono',
        "I_mono": 'InterMono',
        "NK": 'ILC',
        "cDC": 'cDC', "mDC": 'cDC',
        "CD4 Naive": 'CD4NT', "CD4_naive": 'CD4NT',
        "CD8 Naive": 'CD8NT', "CD8_naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T', "CD4_TE": 'CD4T', "TFH": 'CD4T',
        "Th1": 'CD4T', "Th1.Th17": 'CD4T', "Th17": 'CD4T', "Th2": 'CD4T', "Th1/Th17": 'CD4T',
        "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T', "CD8_CM": 'CD8T', "CD8_EM": 'CD8T', "CD8_TE": 'CD8T',
        "gdT": 'gdT', "VD2-": 'gdT', "VD2+": 'gdT', "VD2_gdT": 'gdT', "nVD2_gdT": 'gdT',
        "Intermediate B": 'MemB', "Memory B": 'MemB', "B_NSM": 'MemB', "B_Ex": 'MemB', "B_SM": 'MemB',
        "Naive B": 'NaiveB', "B_naive": 'NaiveB',
        "Plasmablasts": 'Plasma', "Plasma": 'Plasma',
        "Progenitor": 'HSPC', "HSPC": 'HSPC'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_10XMultiome(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD14 Mono": 'CD14Mono',
        "CD16 Mono": 'CD16Mono',
        "CD4 Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T',
        "NK": 'ILC',
        "Naive B": 'NaiveB',
        "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "Memory B": 'MemB',
        "Intermediate B": 'InterB'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_MouseKid(labels):
    labels = [str(label) for label in labels]
    label_map = {
        'Stroma 1': 'Stroma', 'Stroma 2': 'Stroma',
        'Early PT': 'PST', 'PST': 'PST'
    }
    return [label_map.get(label, label) for label in labels]


def get_merged_labels_histone(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "Astrocytes": "Astro",
        "mOL": "Oligo",
        "Endothelial": "VLMC",
        "Mural": "VLMC",
        "Neurons_1": "Neuron",
        "Neurons_2": "Neuron",
        "Neurons_3": "Neuron"
    }
    return [label_map.get(label, label) for label in labels]


def get_merged_labels_pancreas(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "activated_stellate": "mesenchymal",
        "quiescent_stellate": "mesenchymal"
    }
    return [label_map.get(label, label) for label in labels]




def GET_GML(dataset):
    if dataset == 'ISSAAC-seq' or dataset.lower() == 'issaac':
        return get_merged_labels_dataset_ISSAACseq
    elif dataset == 'SHARE-seq' or dataset.lower() == 'share':
        return get_merged_labels_dataset_SHAREseq
    elif dataset == 'Kidney':
        return get_merged_labels_Kidney
    elif dataset.lower() == 'zhu':
        return get_merged_labels_Zhu
    elif dataset.lower() == 'wilk':
        return get_merged_labels_Wilk
    elif dataset == 'Stephenson':
        return get_merged_labels_Stephenson
    elif dataset == 'Hao':
        return get_merged_labels_Hao
    elif dataset == 'Monaco':
        return get_merged_labels_Monaco
    # elif dataset == '10X-Multiome':
    #     return get_merged_labels_10XMultiome
    elif dataset == 'MouseKid':
        return get_merged_labels_MouseKid
    elif dataset == 'histone':
        return get_merged_labels_histone
    elif dataset == 'pancreas':
        return get_merged_labels_pancreas
    else:
        return lambda x: x






