import os, time
if not os.path.isdir("params"):
  # get code
  print("installing ColabDesign")
  os.system("(mkdir params; apt-get install aria2 -qq; \
  aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar; \
  aria2c -q -x 16 https://files.ipd.uw.edu/krypton/af2bind_params.zip; \
  tar -xf alphafold_params_2021-07-14.tar -C params; unzip af2bind_params.zip; touch params/done.txt )&")

  os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")

  # download params
  if not os.path.isfile("params/done.txt"):
    print("downloading params")
    while not os.path.isfile("params/done.txt"):
      time.sleep(5)

import os
import argparse
from colabdesign import mk_afdesign_model
# from IPython.display import HTML
from google.colab import files
import numpy as np
from colabdesign.af.alphafold.common import residue_constants
import pandas as pd
import jax, pickle
import jax.numpy as jnp
# from google.colab import data_table
import sys
import time

start_time = time.time()

def get_pdb(pdb_code=""):
    if pdb_code is None or pdb_code == "":
        upload_dict = files.upload()
        pdb_string = upload_dict[list(upload_dict.keys())[0]]
        with open("tmp.pdb","wb") as out: out.write(pdb_string)
        return "tmp.pdb"
    elif os.path.isfile(pdb_code):
        return pdb_code
    # elif len(pdb_code) == 4:
    #     os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    #     return f"{pdb_code}.pdb"
    # else:
    #     os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb")
    #     return f"AF-{pdb_code}-F1-model_v4.pdb"


def af2bind(inputs, outputs, params, aux):
    opt = inputs["opt"]["af2bind"]
    def bypass_relu(x):
        x_relu = jax.nn.relu(x)
        x = jax.nn.leaky_relu(x)
        return jax.lax.stop_gradient(x_relu - x) + x
    xs = []
    for p in params["af2bind"]:
        if "mlp" in p:
            x = outputs["representations"]["pair"][:-20,-20:]
            x = x.reshape(x.shape[0],-1)
            x = (x - p["scale"]["mean"])/p["scale"]["std"]
            p = p["mlp"]
            for k in  range(5):
                x = x @ p["weights"][k] + p["bias"][k]
                if k < 4:
                    x = jnp.where(opt["bypass_relu"],
                                    bypass_relu(x),
                                    jax.nn.relu(x))
            x = x[:,0]
        else:
            d = outputs["distogram"]["logits"][:-20,-20:]
            # 20 bin = 8 angstroms
            d0 = jax.nn.logsumexp(d[...,:20],-1)
            # todo: check if excluding last bin makes sense
            d1 = jax.nn.logsumexp(d[...,20:-1],-1)
            x = (d0 - d1).max(-1)
        xs.append(x)
    x = jnp.stack(xs,-1)
    aux["af2bind"] = jax.nn.sigmoid(x)
    loss = x[:,opt["type"]]
    loss = (loss * opt["site"]).sum() / (opt["site"].sum() + 1e-8)
    return {"af2bind":loss}

def make_af_model(pdb_filename): #, target_chain):

    if "af_model" not in dir():
        af_model = mk_afdesign_model(protocol="binder",
                                    debug=True,
                                    loss_callback=af2bind,
                                    use_bfloat16=False)
        af_model.opt["weights"]["af2bind"] = 1.0
        af_model.opt["af2bind"] = {"type":0,
                                    "site":np.full(1,False),
                                    "bypass_relu":False}
        af2bind_params = []
        for m in ["ligand_model","peptide_model"]:
            with open(f"{m}.pkl",'rb') as handle:
                af2bind_params.append(pickle.load(handle))
        af_model._params["af2bind"] = af2bind_params + [{}]

    # af_model.prep_inputs(pdb_filename=pdb_filename, chain=target_chain, binder_len=20)
    # Try setting up without target_chain input
    af_model.prep_inputs(pdb_filename=pdb_filename, binder_len=20)
    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.set_opt(weights=0)
    af_model.set_opt("af2bind",site=np.full(af_model._target_len,False))
    af_model.set_weights(af2bind=1.0)
    af_model.predict(verbose=False)
    preds = af_model.aux["af2bind"].copy()

    return af_model, preds

if __name__ == "__main__":

    debug = False
    parser = argparse.ArgumentParser(description = '''Predict binding pockets from protein PDB using AF2BIND.''')

    parser.add_argument('--target_pdb', nargs=1, type= str, default=sys.stdin, help = 'Path to folded structure with protein.')
    parser.add_argument('--rescale_by_max_conf', action=argparse.BooleanOptionalAction, help = 'Flag to rescale confidence options')
    parser.add_argument('--thresh', nargs=1, type=float, default=0.85, help = 'Confidence threshold for selecting pocket residues.')
    parser.add_argument('--model_type', nargs=1, type= str, default="ligand", help = 'Type of model required -- ligand, peptide, or dgram.')
    parser.add_argument('--out_file', nargs=1, type= str, default=sys.stdin, help = 'Path to save pocket residue indices.')
    
    if debug is False:
        args = parser.parse_args()
        target_pdb = args.target_pdb[0]
        rescale_by_max_conf = args.rescale_by_max_conf
        thresh = args.thresh 
        out_file = args.out_file[0]
        model_type = args.model_type
        top_n = 15 # redundant
    else:
        target_pdb = "6w70" #@param {type:"string"}
        target_chain = "A" #@param {type:"string"}
        top_n = 15 # redundant
        model_type = 'ligand' #@param ["ligand", "peptide", "dgram"]
        rescale_by_max_conf = True
        thresh = 0.85 # Threshold below which to choose values
        out_file = ''

    ## Get PDB filename
    pdb_filename = get_pdb(target_pdb)

    aa_order = {v:k for k,v in residue_constants.restype_order.items()}
    af_model, preds = make_af_model(pdb_filename=pdb_filename) #, target_chain=target_chain)

    labels = ["chain","resi","resn","ligand","peptide","dgram"]
    data = []
    for i in range(af_model._target_len):
        c = af_model._pdb["idx"]["chain"][i]
        r = af_model._pdb["idx"]["residue"][i]
        a = aa_order.get(af_model._pdb["batch"]["aatype"][i],"X")
        ps = [round(float(p),3) for p in preds[i]]
        data.append([c,r,a]+ps)

    df = pd.DataFrame(data, columns=labels)
    df.to_csv('results.csv')

    model_m = 0

    # data_table.enable_dataframe_formatter()
    df_sorted = df.sort_values(model_type,ascending=False, ignore_index=True).rename_axis('rank').reset_index()
    # display(data_table.DataTable(df_sorted, min_width=100, num_rows_per_page=15, include_index=False))

    model_m = {"ligand":0, "peptide":1, "dgram":2}[model_type]
    top_n_idx = preds[:,model_m].argsort()[::-1][:15]
    # pymol_cmd="select ch"+str(target_chain)+","
    pymol_cmd="select ch,"
    for n,i in enumerate(top_n_idx):
        p = preds[i]
        c = af_model._pdb["idx"]["chain"][i]
        r = af_model._pdb["idx"]["residue"][i]
        pymol_cmd += f" resi {r}"
        if n < top_n-1:
            pymol_cmd += " +"

    print("\n🧪Pymol Selection Cmd:")
    print(pymol_cmd)

    ## Rescale with max confidence value
    if rescale_by_max_conf:
        preds_adj = preds[:,model_m].copy() / preds[:,model_m].max()
    else:
        preds_adj = preds[:,model_m].copy()

    ## Make arrays of indices above threshold
    ind_above_thresh = np.where(preds_adj >= thresh)[0]

    ## Save file with residue numbers
    np.save(out_file, ind_above_thresh)

    print(f'Time taken for calculation is {time.time()-start_time:.1f} seconds')