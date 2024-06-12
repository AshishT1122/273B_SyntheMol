import os
import glob
from argparse import Namespace
from inference import main
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/diffdock", response_class=JSONResponse)
async def diffdock(request: Request):
    request = await request.json()
    smiles_str = request['smiles']
    print(f"Request received for Smile string {smiles_str}")
    
    args = Namespace(
        config=None,
        protein_ligand_csv=None,
        complex_name=smiles_str,
        protein_path='GLP1.pdb',
        protein_sequence=None,
        ligand_description=smiles_str,
        loglevel='WARNING',
        out_dir='C:/Users/ashis/Downloads/SyntheMol/SyntheMol/diffdock_output',
        save_visualisation=False,
        samples_per_complex=10,  # same in both
        model_dir='./workdir/v1.1/score_model',
        ckpt='best_ema_inference_epoch_model.pt',  # same in both
        confidence_model_dir='./workdir/v1.1/confidence_model',
        confidence_ckpt='best_model_epoch75.pt',
        batch_size=10,
        no_final_step_noise=True,  # same in both
        inference_steps=20,  # same in both
        actual_steps=19,
        old_score_model=False,  # same in both
        old_confidence_model=True,
        initial_noise_std_proportion=1.4601642460337794,
        choose_residue=False,
        temp_sampling_tr=1.170050527854316,
        temp_psi_tr=0.727287304570729,
        temp_sigma_data_tr=0.9299802531572672,
        temp_sampling_rot=2.06391612594481,
        temp_psi_rot=0.9022615585677628,
        temp_sigma_data_rot=0.7464326999906034,
        temp_sampling_tor=7.044261621607846,
        temp_psi_tor=0.5946212391366862,
        temp_sigma_data_tor=0.6943254174849822,
        gnina_minimize=False,
        gnina_path='gnina',
        gnina_log_file='gnina_log.txt',
        gnina_full_dock=False,
        gnina_autobox_add=4.0,
        gnina_poses_to_optimize=1,
        different_schedules=False,
        inf_sched_alpha=1,
        inf_sched_beta=1,
        limit_failures=5,
        no_model=False,
        no_random=False,
        no_random_pocket=False,
        ode=False,
        old_filtering_model=True,
        resample_rdkit=False,
        sigma_schedule='expbeta'
    )


    main(args)
    
    return {
        "status": "success"
    }