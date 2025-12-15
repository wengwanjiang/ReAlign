import os, pickle, random
# os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import  numpy  as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime
# from torch.functional import F

import swanlab
from mld.config import parse_args_SPM
from mld.data.get_data import get_dataset
from collections import OrderedDict

from .models.spm import SPM, process_T5_outputs
from .models.utils import lengths_to_mask
# from mld.models.modeltype.mld import MLD
from mld.utils.utils import set_seed

from .eval_tmr import calculate_retrieval_metrics, calculate_retrieval_metrics_small_batches
from diffusers.optimization import get_scheduler

os.makedirs('./logs', exist_ok=True)
fptr = open('./logs/h3d_tmr.log', 'a')#, buffering=0)
def eval(test_loader, reward_model, epoch=0, mode='M1T0', writer=None, device='cuda:0'):
    test_result = [[], [], []]
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
        assert len(text) <= 32, "Please following the settings in Small Batch protocol defined in TMR (ICCV23)"
        tiemstep = torch.zeros(len(text),).long().to(device)
        with torch.no_grad():
            t_len, token_emb, cls_token = process_T5_outputs(text, reward_model.clip)
        feats_ref = feats_ref.float().to(device)
        m_latent = reward_model.encode_motion(feats_ref, m_len, timestep=tiemstep if mode[3] =='1' else None)[0].squeeze().detach().cpu().numpy()
        t_latent = reward_model.encode_text(token_emb, t_len, timestep=tiemstep if mode[1] =='1' else None)[0].squeeze().detach().cpu().numpy()
        for j in range(len(batch['text'])):
            test_result[0].append(batch['text'][j])
            test_result[1].append(t_latent[j])
            test_result[2].append(m_latent[j])
        
        del tiemstep, feats_ref, t_len, token_emb, cls_token, m_latent, t_latent
        torch.cuda.empty_cache()
            
    random.seed(42)
    shuffle_index = [i for i in range(len(test_result[2]))]
    random.shuffle(shuffle_index)
    
    test_result[0] = [test_result[0][i] for i in shuffle_index]
    test_result[1] = [test_result[1][i] for i in shuffle_index]
    test_result[2] = [test_result[2][i] for i in shuffle_index]

    print('==================T2M Retrieval Results====================')
    t2m_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=epoch, fptr=fptr)
    calculate_retrieval_metrics(test_result, epoch=epoch, fptr=fptr)
    temp = test_result[2]
    test_result[2] = test_result[1]
    test_result[1] = temp
    print('==================M2T Retrieval Results====================')
    m2t_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=epoch, fptr=fptr)
    calculate_retrieval_metrics(test_result, epoch=epoch, fptr=fptr)
    fptr.flush()
    
    if writer is not None:
        if t2m_metrics:
            for key, val in t2m_metrics.items():
                writer.log({f"Val/T2M_{key}": val}, step=epoch)
        if m2t_metrics:
            for key, val in m2t_metrics.items():
                writer.log({f"Val/M2T_{key}": val}, step=epoch)

def main():
    cfg = parse_args_SPM()
    set_seed(cfg.SEED_VALUE)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = get_dataset(cfg)    
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    text_src = cfg.DATASET.TEXT_SRC
    print(f'Using text source: {text_src}')
    if cfg.DATASET.NAME == 'humanml3d':
        ds_name, nfeats = 'H3D', 263
    elif cfg.DATASET.NAME == 'kit':
        ds_name, nfeats = 'KIT', 251
    
    # Convert config to dict for swanlab logging
    config_dict = dict(cfg)
    writer = swanlab.init(
        project="ReAlign-SPM",
        experiment_name=f"SPM_{ds_name}_{text_src}",
        config=config_dict
    )
    
    model = SPM(t5_path=cfg.t5_path, temp=cfg.CLTemp, thr=cfg.CLThr, nfeats=nfeats)
    model.train()
    model = model.to(device)
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon)
    cfg.TRAIN.max_train_steps = cfg.TRAIN.max_train_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_train_steps)
    
    
    # =================================== #
        
    step_aware, maxT, thr = cfg.step_aware, cfg.maxT, cfg.NoiseThr
    epochs = cfg.TRAIN.EPOCHS
    save_freq = cfg.TRAIN.save_freq
    # 定义概率分布
    probs = torch.zeros(maxT, device=device)
    probs[:maxT//10] = 0.8 / maxT  # 0到1000的概率为0.8
    probs[maxT//10:] = 0.2/ (maxT - 101)  # 401到maxT-1的概率为0.2
    
    # =================================== #
    
    
    print(f'================StepAware:{cfg.step_aware}, maxT: {cfg.maxT}, NoiseThr: {cfg.NoiseThr} ================\n')
    
    global_step = 0

    for epoch in range(epochs):
        eval(test_loader, model, epoch, writer=writer, device=device)
        torch.cuda.empty_cache()
        
        epoch_loss = 0.0
        total_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}, Avg Loss {total_loss:.4f}', total=len(train_loader)):
            feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
            feats_ref = feats_ref.float().to(device)
            # print(text_src, text)

            timestep = torch.multinomial(probs, num_samples=feats_ref.shape[0], replacement=True).long()
            #timestep = torch.randint(0, maxT, (1,), device='cuda:0').long()   
            if random.random() > thr:
                with torch.no_grad():
                    # m_len_mask = lengths_to_mask(m_len, device=feats_ref.device)
                    # z, _ = vae.encode(feats_ref, m_len_mask)
                    noise = torch.randn_like(feats_ref)  # shape=[bs, 1, 256]
                    noised_z = scheduler.add_noise(original_samples=feats_ref.clone(), noise=noise, timesteps=timestep)
                    # feats_ref= vae.decode(noised_z, m_len_mask)
                    feats_ref = noised_z
            tfs_loss = model(text=text, motion_feature=feats_ref, m_len=m_len, timestep=timestep, mode=step_aware)
            loss_item = tfs_loss.item()
            total_loss += loss_item
            epoch_loss += loss_item
            # ft_loss = torch.tensor(0).cuda()
            tfs_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            del tfs_loss, feats_ref, timestep
            torch.cuda.empty_cache()
            
            writer.log({"Train/loss": loss_item, "Train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
            global_step += 1
            
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_epoch_loss = total_loss / len(train_loader)
        print(f'Time = {time_str}, Epoch = {epoch}, Avg loss = {avg_epoch_loss:.4f}\n')
        print(f'Time = {time_str}, Epoch = {epoch}, Avg loss = {avg_epoch_loss:.4f}\n', file=fptr)
        fptr.flush()
        
        writer.log({"Epoch/avg_loss": avg_epoch_loss}, step=epoch)
        
        # total_loss = 0
        if epoch % save_freq == 0:
            new_state = OrderedDict()
            for k, v in model.state_dict().items():
                if 'clip' not in k:
                    new_state[k]=v
            
            # Create checkpoint directory if not exists
            checkpoint_dir = cfg.SPM_CHECKPOINT_DIR
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            prefix = f'{text_src}_'
            thr_s = int(thr * 100 // 1)
            if thr > 1:
                prefix += f'SPM_{ds_name}_Thr{thr_s}_Temp{maxT}_SA{step_aware}_E{epoch}.pth'
            else:
                prefix += f'SPM_{ds_name}_Thr{thr_s}_Temp{maxT}_SA{step_aware}_E{epoch}.pth'
            torch.save({'state_dict': new_state}, os.path.join(checkpoint_dir, prefix))
        


if __name__ == "__main__":
    # print(123123)
    main()
