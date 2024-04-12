import os
import sys
sys.path.append('/home/gli/project/DVAE/Diffsurface')
import logging
import pandas as pd
from Bio.PDB import PDBExceptions
import pickle
import joblib
from tqdm import tqdm
import lmdb
import torch
from torch.utils.data import Dataset
from utils.protein.points import ProteinPairData
from utils.convert_pdb2npy import load_seperate_structure
from utils.geometry import atoms_to_points

Tensor, tensor = torch.LongTensor, torch.FloatTensor


class SAbDabDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    def __init__(self, processed_dir='./data/processed', chothia_dir='../all_structures/chothia', split='train', relax_struct=False, pred_struct=False, transform=None, reset=False,
                 use_plm=False, surface=None, test_dir=None, test_list=None, multiEpitope_csv=None):
        super().__init__()
        self.relax_struct = relax_struct
        self.pred_struct = pred_struct
        self.processed_dir = processed_dir
        self.test_dir = test_dir
        self.test_list = test_list
        self.multiEpitope_csv = multiEpitope_csv
        self.chothia_dir = chothia_dir
        self._entry_cache_path = os.path.join(self.processed_dir, 'entry')
        self._split_path = os.path.join(self.processed_dir, 'sequence_only_split')
        self._surface_cache_path = os.path.join(processed_dir, 'surface.lmdb')
        self._structure_cache_path = os.path.join(self.processed_dir, 'structures.lmdb')
        self._structure_relax_cache_path = os.path.join(self.processed_dir, 'structures_relax.lmdb')
        self._structure_esmfold_cache_path = os.path.join(self.processed_dir, 'structures_esmfold.lmdb')
        os.makedirs(processed_dir, exist_ok=True)

        # entries
        self._load_sabdab_entries(reset)

        self.db_conn, self.db_ids = None, None
        self.surface = surface
        if surface is not None:
            # surface
            self._load_surface()  # Load atomic coordinates
        else:
            # structures
            self._load_structures()
            if relax_struct or pred_struct:
                self._load_structures(relax_struct=relax_struct, pred_struct=pred_struct)  # filter entries based on existing structures
                assert not pred_struct or not pred_struct, 'You can only claim relaxed or predicted structure.'
        self.sabdab_entries = list(filter(lambda e: e['id'] in self.db_ids, self.sabdab_entries))

        # splits
        self._load_split(split)

        # plm features
        self.use_plm = use_plm
        if self.use_plm:
            self._load_plm_feature()
        self.transform = transform

    def _load_sabdab_entries(self, reset):
        self.sabdab_entries = None
        if not os.path.exists(self._entry_cache_path) or reset:
            raise ValueError('Please run preprocess.py first to generate entries.')
        with open(self._entry_cache_path, 'rb') as f:
            self.sabdab_entries = pickle.load(f)

    def _load_structures(self, relax_struct=False, pred_struct=False):
        path = self._structure_cache_path
        if relax_struct:
            path = self._structure_relax_cache_path
        elif pred_struct:
            path = self._structure_esmfold_cache_path
        with open(path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)

    def _load_split(self, split):
        self.ids_in_split = None
        assert split in ('train', 'val', 'test')
        if self.test_list is not None:
            self.ids_in_split = [entry['id'] for entry in self.sabdab_entries if entry['id'] in self.test_list]

        elif self.multiEpitope_csv is not None:
            entry_ids = []
            df = pd.read_csv(self.multiEpitope_csv)
            for idx, row in df[df['antibody'].str.len() > 1].iterrows():
                H, L = row["antibody"]
                entry_ids.append(f'{row["pdb_code"]}_{H}_{L}_{row["antigen"]}')
            self.ids_in_split = [entry['id'] for entry in self.sabdab_entries if entry['id'] in entry_ids]
        elif self.test_dir is None:
            with open(self._split_path, 'rb') as f:
                val_test_split = pickle.load(f)

            val_test_split['val'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] in val_test_split['val']]
            val_test_split['test'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] in val_test_split['test']]
            val_test_split['train'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] not in val_test_split['val'] + val_test_split['test']]
            self.ids_in_split = val_test_split[split]
        else:
            test_id = [i[:4] for i in os.listdir(self.test_dir)]
            val_split = [entry['id'] for entry in self.sabdab_entries if entry['id'][:4] in test_id]
            train_split = [entry['id'] for entry in self.sabdab_entries if entry['id'][:4] not in test_id]
            self.ids_in_split = train_split if split == 'train' else val_split

    def _load_plm_feature(self):
        plm_feature_path = os.path.join(self.processed_dir, 'esm2_embeddings.pt')
        self.plm_feature = torch.load(plm_feature_path)
        self.ids_in_split = [i for i in self.ids_in_split if i in self.plm_feature.keys()]

    def preprocess_surface(self, entry):
        parsed = {'id': entry['id'], 'P': None, }
        pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))  # different entries can use the same PDB
        try:
            if len(entry['ag_chains']) == 0 or entry['H_chain'] is None or entry['L_chain'] is None:  # both light and heavy chains exist
                raise ValueError(f'Missing antigen, H-chain or L-chain.')
            proteins = load_seperate_structure(pdb_path, return_map=False, ligand=entry['H_chain'] + entry['L_chain'], receptor=entry['ag_chains'])
            l, r = proteins['ligand'], proteins['receptor']

            atomxyz_ligand, atomtypes_ligand, resxyz_ligand, restypes_ligand = tensor(l["atom_xyz"]), tensor(l["atom_types"]), tensor(l["res_xyz"]), Tensor(l["res_types"])
            batch_atoms_ligand = torch.zeros(len(atomxyz_ligand)).long().to(atomxyz_ligand.device)
            pts_ligand, norms_ligand, _ = atoms_to_points(atomxyz_ligand, batch_atoms_ligand, atomtypes=atomtypes_ligand, resolution=self.surface.resolution,
                                                          sup_sampling=self.surface.sup_sampling, distance=self.surface.distance)
            atomxyz_receptor, atomtypes_receptor, resxyz_receptor, restypes_receptor = tensor(r["atom_xyz"]), tensor(r["atom_types"]), tensor(r["res_xyz"]), Tensor(r["res_types"])
            batch_atoms_receptor = torch.zeros(len(atomxyz_receptor)).long().to(atomxyz_receptor.device)
            pts_receptor, norms_receptor, _ = atoms_to_points(atomxyz_receptor, batch_atoms_receptor, atomtypes=atomtypes_receptor, resolution=self.surface.resolution,
                                                              sup_sampling=self.surface.sup_sampling, distance=self.surface.distance)
            intf_ligand = (torch.cdist(pts_ligand, pts_receptor) < self.surface.intf_cutoff).sum(dim=1) > 0
            intf_receptor = (torch.cdist(pts_receptor, pts_ligand) < self.surface.intf_cutoff).sum(dim=1) > 0

            parsed['P'] = ProteinPairData(xyz_ligand=pts_ligand, normals_ligand=norms_ligand, atomxyz_ligand=atomxyz_ligand, atomtypes_ligand=atomtypes_ligand,
                                          resxyz_ligand=resxyz_ligand, restypes_ligand=restypes_ligand, xyz_receptor=pts_receptor, normals_receptor=norms_receptor,
                                          atomxyz_receptor=atomxyz_receptor, atomtypes_receptor=atomtypes_receptor, resxyz_receptor=resxyz_receptor,
                                          restypes_receptor=restypes_receptor, intf_ligand=intf_ligand, intf_receptor=intf_receptor)
        except (PDBExceptions.PDBConstructionException, Exception, KeyError, ValueError, FileNotFoundError, EOFError) as e:
            logging.warning('[{}] {}: {}'.format(entry['id'], e.__class__.__name__, str(e)))
            return None
        return parsed

    def _load_surface(self):
        self.surf_structures = {}
        if not os.path.exists(self._surface_cache_path):
            n_jobs = max(joblib.cpu_count() // 2, 1)
            data_list = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.preprocess_surface)(e, ) for e in tqdm(self.sabdab_entries, dynamic_ncols=True, desc='Preprocess'))

            ids = []
            db_conn = lmdb.open(self._surface_cache_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False, )
            with db_conn.begin(write=True, buffers=True) as txn:
                for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                    if data is not None:
                        ids.append(data['id'])
                        txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
            with open(self._surface_cache_path + '-ids', 'wb') as f:
                pickle.dump(ids, f)
            print(f'Loading {len(ids)} complex surfaces successfully with Biopython. ')

        with open(self._surface_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)

    def get_structure(self, id, relax_struct=False, pred_struct=False):
        if self.db_conn is None:
            if self.surface:
                lmdb_path = self._surface_cache_path
            elif relax_struct:
                lmdb_path = self._structure_relax_cache_path
            elif pred_struct:
                lmdb_path = self._structure_esmfold_cache_path
            else:
                lmdb_path = self._structure_cache_path

            self.db_conn = lmdb.open(lmdb_path, map_size=self.MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.surface:
            data['P']['id'] = id
            data['P']['intf_pred'] = torch.empty_like(data['P']['intf_receptor'])
        if self.relax_struct or self.pred_struct:
            data_relax_or_pred = self.get_structure(id, relax_struct=self.relax_struct, pred_struct=self.pred_struct)
            data['antigen']['pos_heavyatom'] = data_relax_or_pred['antigen']['pos_heavyatom'].clone()  # use relax_struct positions, no dihedral are used
        if self.use_plm:
            data['antigen']['plm_feature'] = self.plm_feature[id]
        if not self.surface and self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='/home/gli/project/Data/data_surface/processed')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(processed_dir=args.processed_dir, split=args.split, reset=args.reset, use_plm=True)
    print(dataset[0])
    print(len(dataset), len(dataset.clusters))
