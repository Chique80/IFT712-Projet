import csv
import numpy as np

species_mapping =           ['Acer_Opalus', 
                            'Pterocarya_Stenoptera', 
                            'Quercus_Hartwissiana', 
                            'Tilia_Tomentosa', 
                            'Quercus_Variabilis', 
                            'Magnolia_Salicifolia', 
                            'Quercus_Canariensis', 
                            'Quercus_Rubra', 
                            'Quercus_Brantii', 
                            'Salix_Fragilis', 
                            'Zelkova_Serrata', 
                            'Betula_Austrosinensis', 
                            'Quercus_Pontica', 
                            'Quercus_Afares', 
                            'Quercus_Coccifera', 
                            'Fagus_Sylvatica', 
                            'Phildelphus', 
                            'Acer_Palmatum', 
                            'Quercus_Pubescens', 
                            'Populus_Adenopoda', 
                            'Quercus_Trojana', 
                            'Alnus_Sieboldiana', 
                            'Quercus_Ilex', 
                            'Arundinaria_Simonii', 
                            'Acer_Platanoids', 
                            'Quercus_Phillyraeoides', 
                            'Cornus_Chinensis', 
                            'Liriodendron_Tulipifera', 
                            'Cytisus_Battandieri', 
                            'Rhododendron_x_Russellianum', 
                            'Alnus_Rubra', 
                            'Eucalyptus_Glaucescens', 
                            'Cercis_Siliquastrum', 
                            'Cotinus_Coggygria', 
                            'Celtis_Koraiensis', 
                            'Quercus_Crassifolia', 
                            'Quercus_Kewensis', 
                            'Cornus_Controversa', 
                            'Quercus_Pyrenaica', 
                            'Callicarpa_Bodinieri', 
                            'Quercus_Alnifolia', 
                            'Acer_Saccharinum', 
                            'Prunus_X_Shmittii', 
                            'Prunus_Avium', 
                            'Quercus_Greggii', 
                            'Quercus_Suber', 
                            'Quercus_Dolicholepis', 
                            'Ilex_Cornuta', 
                            'Tilia_Oliveri', 
                            'Quercus_Semecarpifolia', 
                            'Quercus_Texana', 
                            'Ginkgo_Biloba', 
                            'Liquidambar_Styraciflua', 
                            'Quercus_Phellos', 
                            'Quercus_Palustris', 
                            'Alnus_Maximowiczii', 
                            'Quercus_Agrifolia', 
                            'Acer_Pictum',
                            'Acer_Rufinerve', 
                            'Lithocarpus_Cleistocarpus', 
                            'Viburnum_x_Rhytidophylloides', 
                            'Ilex_Aquifolium', 
                            'Acer_Circinatum', 
                            'Quercus_Coccinea', 
                            'Quercus_Cerris', 
                            'Quercus_Chrysolepis', 
                            'Eucalyptus_Neglecta', 
                            'Tilia_Platyphyllos', 
                            'Alnus_Cordata', 
                            'Populus_Nigra', 
                            'Acer_Capillipes', 
                            'Magnolia_Heptapeta', 
                            'Acer_Mono', 
                            'Cornus_Macrophylla', 
                            'Crataegus_Monogyna', 
                            'Quercus_x_Turneri', 
                            'Quercus_Castaneifolia', 
                            'Lithocarpus_Edulis', 
                            'Populus_Grandidentata', 
                            'Acer_Rubrum', 
                            'Quercus_Imbricaria', 
                            'Eucalyptus_Urnigera', 
                            'Quercus_Crassipes', 
                            'Viburnum_Tinus', 
                            'Morus_Nigra', 
                            'Quercus_Vulcanica', 
                            'Alnus_Viridis', 
                            'Betula_Pendula', 
                            'Olea_Europaea', 
                            'Quercus_Ellipsoidalis', 
                            'Quercus_x_Hispanica', 
                            'Quercus_Shumardii', 
                            'Quercus_Rhysophylla', 
                            'Castanea_Sativa', 
                            'Ulmus_Bergmanniana', 
                            'Quercus_Nigra', 
                            'Salix_Intergra', 
                            'Quercus_Infectoria_sub', 
                            'Sorbus_Aria']

def read_data_from_csv():
    data = []
    with open('data\\train.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] != 'id':
                species = species_mapping.index(row[1])
                row[1] = species
                data.append(row)

    return np.array(data, dtype=np.float32)
pass

def read_dataset(label_has_onehot = False):
    data = read_data_from_csv()
    x = data[:,2:]

    if label_has_onehot:
        y = []
        for i in range(0,len(data)):
            species = data[i,1]
            onehot = np.zeros(len(species_mapping), dtype=np.int32)
            onehot[int(species)] = 1
            y.append(onehot)

        y = np.array(y, dtype=np.int32)
    else:
        y = np.array(data[:,1], dtype=np.int32)

    return x, y
pass
