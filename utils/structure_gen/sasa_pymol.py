import pymol   # https://pymol.org/conda/  # conda install -c schrodinger pymol  # apt-get update && apt-get install libgl1
import os
from tqdm import tqdm
import pickle
import sys
sys.path.append('/home/gli/project/DVAE/Diffsurface')

data_base = '/home/gli/project/Data/data_surface'

path_to_fasta = f'{data_base}/processed/'
path_to_input_structures = f'{data_base}/all_structures/chothia/'
path_to_output_structures = f'{data_base}/all_structures/chothia_single_chain/'
sasa_path = f'{data_base}/processed/SASA/'
_entry_cache_path = os.path.join(f'{data_base}/processed/entry')
os.makedirs(path_to_output_structures, exist_ok=True)

with open(_entry_cache_path, 'rb') as f:
    sabdab_entries = pickle.load(f)

chain_list = []
os.makedirs(sasa_path, exist_ok=True)
for entry in sabdab_entries:
    pdb_path = os.path.join(path_to_input_structures, '{}.pdb'.format(entry['pdbcode']))  # different entries can use the same PDB
    if not os.path.exists(pdb_path):
        continue
    for antigen_chain_i in entry['ag_chains']:
        chain_list.append(entry['pdbcode'] + '_' + antigen_chain_i)


chain_list = list(set(chain_list))
for chain_i in tqdm(chain_list):
    pdb_i, chain_i = chain_i.split('_')
    if not os.path.exists(path_to_output_structures + pdb_i + '_' + chain_i + '.pdb'):
        pymol.cmd.load(path_to_input_structures + pdb_i + '.pdb')
        pymol.cmd.select('selected_chain', 'chain ' + chain_i)
        pymol.cmd.save(path_to_output_structures + pdb_i + '_' + chain_i + '.pdb', 'selected_chain')
        pymol.cmd.delete('all')

chain_list = os.listdir(f'{data_base}/all_structures/chothia_single_chain/')
for pdb_i in tqdm(chain_list):
    pdb_i = pdb_i.split('.')[0]
    if not os.path.exists(sasa_path + pdb_i):
        try:
            os.system(f'python /home/gli/project/DVAE/Diffsurface/utils/structure_gen/sasa_mdtraj.py   /home/gli/project/Data/data_surface/all_structures/chothia_single_chain/' + pdb_i + '.pdb   /home/gli/project/Data/data_surface/processed/SASA/' + pdb_i)
        except:
            print(f'{pdb_i} cannot be processed...')
            continue
