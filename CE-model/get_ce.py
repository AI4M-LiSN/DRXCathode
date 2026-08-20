from ase.db import connect
from ase.io import read, write
from ase.db import connect
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.build import bulk, sort

def get_db(file_db,file_energy,address):
    with open(file_energy,'r') as f_energy: #get the 序号与能量
        lines=f_energy.readlines()
    name=[]
    energy=[]
    mix_energy=[]
    for line in lines:
        word=line.split()
        name.append(word[0])    # get the structrue ID
        energy.append(float(word[2].strip()))    # get the energy per f.u.
        #mix_energy.append(float(word[3].strip()))  # get the mix_energy per f.u.

    for index in range(len(name)):
        file=address+name[index]+'.vasp'
        structure = read(file, format='vasp')    
        # 将结构和能量写入ASE数据库
        with connect(file_db) as db:
            db.write(structure,str_energy=energy[index])


file_db='str_mix.db'
file_energy_0='str_energy/P0L0_energy'
address_0='str_energy/P0L0_'

file_energy_1='str_energy/P0L1_energy'
address_1='str_energy/P0L1_'

file_energy_2='str_energy/P0L2_energy'
address_2='str_energy/P0L2_'

file_energy_3='str_energy/P0L3_energy'
address_3='str_energy/P0L3_'

file_energy_4='str_energy/P1L1_energy'
address_4='str_energy/P1L1_'

file_energy_5='str_energy/P1L2_energy'
address_5='str_energy/P1L2_'

file_energy_6='str_energy/P1L3_energy'
address_6='str_energy/P1L3_'

file_energy_7='str_energy/P2L3_energy'
address_7='str_energy/P2L3_'

file_energy_8='str_energy/P2L4_energy'
address_8='str_energy/P2L4_'

file_energy_9='str_energy/P2L5_energy'
address_9='str_energy/P2L5_'

get_db(file_db,file_energy_0,address_0)
get_db(file_db,file_energy_1,address_1)
get_db(file_db,file_energy_2,address_2)
get_db(file_db,file_energy_3,address_3)
get_db(file_db,file_energy_4,address_4)
get_db(file_db,file_energy_5,address_5)
get_db(file_db,file_energy_6,address_6)
get_db(file_db,file_energy_7,address_7)
get_db(file_db,file_energy_8,address_8)
get_db(file_db,file_energy_9,address_9)


num=0
with connect(file_db) as db:
    # 获取数据库中所有条目
    all_entries = db.select()
    # 遍历每个条目并打印结构和能量

    for entry in all_entries:         
        num=num+1
        structure = entry.toatoms()        
        print("Structure:", structure)
        x_energy=entry.str_energy
        print("Energy:", x_energy)
        #if(num==1):
        #    primitive_structure = structure
#write(filename=filename,images=sort(structure),direct=True)
print(num)
# get the pri strucutre to construct cluster 
primitive_structure=read('str_energy/LMO_pri.vasp',format='vasp')
print(primitive_structure)
# get the ce pairs
cs = ClusterSpace(structure=primitive_structure,
                  cutoffs=[7, 4, 4],
                  chemical_symbols=[['Li', 'P'],['Li', 'P'], ['Na','Mn','Ni'],['Na','Mn','Ni'],['Na','Mn','Ni'],['Na','Mn','Ni'],['O'],['O'],['O'],['O'],['O'],['O'],['O'],['O']],
                  symprec=0.01,
                  position_tolerance=0.02,
                  )
print(cs)
#get the structure and energy from db

sc = StructureContainer(cluster_space=cs)
for row in db.select():
    sc.add_structure(structure=row.toatoms(),
                     properties={'str_energy': row.str_energy})
print(sc) 

#fitting

opt = CrossValidationEstimator(fit_data=sc.get_fit_data(key='str_energy'), fit_method='ardr')
opt.validate()
opt.train()
print(opt)

#get .ce contain the CEI

ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters, metadata=opt.summary)
print(ce)
ce.write('mixing_energy.ce')


opt2 = CrossValidationEstimator(fit_data=sc.get_fit_data(key='str_energy'), fit_method='least-squares')
opt2.validate()
opt2.train()
print(opt2)

#get .ce contain the CEI

ce2 = ClusterExpansion(cluster_space=cs, parameters=opt2.parameters, metadata=opt2.summary)
print(ce2)
ce2.write('ls_mixing_energy.ce')

