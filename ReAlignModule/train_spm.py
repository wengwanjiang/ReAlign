import os, pickle, random
# os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import  numpy  as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime
# from torch.functional import F

from mld.config import parse_args_SPM
from mld.data.get_data import get_dataset
from collections import OrderedDict

from .models.spm import SPM, process_T5_outputs
from .models.utils import lengths_to_mask
# from mld.models.modeltype.mld import MLD
from mld.utils.utils import set_seed

from .eval_tmr import calculate_retrieval_metrics, calculate_retrieval_metrics_small_batches
from diffusers.optimization import get_scheduler

def eval(test_loader, reward_model, epoch=0, mode='M1T0'):
    test_result = [[], [], []]
    
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
        assert len(text) <= 32, "Please following the settings in Small Batch protocol defined in TMR (ICCV23)"
        tiemstep = torch.zeros(len(text),).long().cuda()
        with torch.no_grad():
            t_len, token_emb, cls_token = process_T5_outputs(text, reward_model.clip)
        feats_ref = feats_ref.float().cuda()
        m_latent = reward_model.encode_motion(feats_ref, m_len, timestep=tiemstep if mode[3] =='1' else None)[0].squeeze().detach().cpu().numpy()
        t_latent = reward_model.encode_text(token_emb, t_len, timestep=tiemstep if mode[1] =='1' else None)[0].squeeze().detach().cpu().numpy()
        for j in range(len(batch['text'])):
            test_result[0].append(batch['text'][j])
            test_result[1].append(t_latent[j])
            test_result[2].append(m_latent[j])
            
    random.seed(42)
    shuffle_index = [i for i in range(len(test_result[2]))]
    random.shuffle(shuffle_index)
    
    test_result[0] = [test_result[0][i] for i in shuffle_index]
    test_result[1] = [test_result[1][i] for i in shuffle_index]
    test_result[2] = [test_result[2][i] for i in shuffle_index]

    print('==================T2M Retrieval Results====================')
    calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
    calculate_retrieval_metrics(test_result, epoch=epoch)
    temp = test_result[2]
    test_result[2] = test_result[1]
    test_result[1] = temp
    print('==================M2T Retrieval Results====================')
    calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
    calculate_retrieval_metrics(test_result, epoch=epoch)


def main():
    cfg = parse_args_SPM()
    set_seed(cfg.SEED_VALUE)
    dataset = get_dataset(cfg)    
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    
    if cfg.DATASET.NAME == 'humanml3d':
        ds_name, nfeats = 'H3D', 263
    elif cfg.DATASET.NAME == 'kit':
        ds_name, nfeats = 'KIT', 251
    model = SPM(t5_path=cfg.t5_path, temp=cfg.CLTemp, thr=cfg.CLThr, nfeats=nfeats)
    model.train()
    model = model.cuda()
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
    total_loss = 0
        
    step_aware, maxT, thr = cfg.step_aware, cfg.maxT, cfg.NoiseThr
    epochs = cfg.TRAIN.EPOCHS
    save_freq = cfg.TRAIN.save_freq
    # 定义概率分布
    probs = torch.zeros(maxT, device='cuda:0')
    probs[:maxT//10] = 0.8 / maxT  # 0到1000的概率为0.8
    probs[maxT//10:] = 0.2/ (maxT - 101)  # 401到maxT-1的概率为0.2
    
    # =================================== #
    
    
    print(f'================StepAware:{cfg.step_aware}, maxT: {cfg.maxT}, NoiseThr: {cfg.NoiseThr} ================\n')
    
    

    for epoch in range(epochs):
        eval(test_loader, model, epoch)
        for i, batch in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}, Avg Loss {total_loss:.4f}', total=len(train_loader)):
            feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
            feats_ref = feats_ref.float().cuda()
            

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
            total_loss += tfs_loss
            # ft_loss = torch.tensor(0).cuda()
            tfs_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Time = {time_str}, Epoch = {epoch}, Avg loss = {total_loss/len(train_loader)}\n')
        total_loss = 0
        if epoch % save_freq == 0:
            new_state = OrderedDict()
            for k, v in model.state_dict().items():
                if 'clip' not in k:
                    new_state[k]=v
            
            prefix = ''
            thr_s = int(thr * 100 // 1)
            if thr > 1:
                prefix += f'SPM_{ds_name}_Thr{thr_s}_Temp{maxT}_SA{step_aware}_E{epoch}.pth'
            else:
                prefix += f'SPM_{ds_name}_Thr{thr_s}_Temp{maxT}_SA{step_aware}_E{epoch}.pth'
            torch.save({'state_dict': new_state}, f'/data/wwj/ckpt/T5_SPM/{prefix}')
        


if __name__ == "__main__":
    print(123123)
    main()
