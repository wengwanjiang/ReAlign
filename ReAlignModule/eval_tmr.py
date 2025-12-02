import os
import sys
import logging
import datetime
import os.path as osp

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from torch.functional import F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import faiss
import numpy as np
def calculate_retrieval_metrics_small_batches(test_result, batch_size=32, epoch=0):
    text_list, text_latents, motion_latents = test_result
    text_latents = np.array(text_latents).astype('float32')
    motion_latents = np.array(motion_latents).astype('float32')
    total_samples = len(text_latents)
    r1_sum, r5_sum, r10_sum = 0.0, 0.0, 0.0
    r2_sum ,r3_sum = 0, 0
    num_batches = 0
    batch_size = 32
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        # 构造当前批次的数据
        batch_text_latents = text_latents[start_idx:end_idx]
        batch_motion_latents = motion_latents[start_idx:end_idx]
        
        # 如果剩余样本不足，随机采样补全
        if len(batch_text_latents) < batch_size:
            remaining = batch_size - len(batch_text_latents)
            random_indices = np.random.choice(total_samples, remaining, replace=False)
            batch_text_latents = np.concatenate([batch_text_latents, text_latents[random_indices]])
            batch_motion_latents = np.concatenate([batch_motion_latents, motion_latents[random_indices]])
        
        # 调用原始函数计算当前批次的指标
        batch_result = calculate_retrieval_metrics(
            [text_list, batch_text_latents, batch_motion_latents],
            verbose=False,
            epoch=epoch
        )
        
        # 累加指标
        r1_sum += batch_result['R1']
        r2_sum += batch_result['R2']
        r3_sum += batch_result['R3']
        r5_sum += batch_result['R5']
        r10_sum += batch_result['R10']
        num_batches += 1

    # 计算平均指标
    avg_r1 = r1_sum / num_batches
    avg_r2 = r2_sum / num_batches
    avg_r3 = r3_sum / num_batches
    avg_r5 = r5_sum / num_batches
    avg_r10 = r10_sum / num_batches

    # 结果格式化
    results = {
        'R1': round(avg_r1, 3),
        'R2': round(avg_r2, 3),
        'R3': round(avg_r3, 3),
        'R5': round(avg_r5, 3),
        'R10': round(avg_r10, 3)
    }

    # 控制台输出
    total = len(text_latents)
    print(f"BS32 | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs")


    return results
def calculate_retrieval_metrics(train_result, verbose=True, epoch=0):

    text_list, text_latents, motion_latents = train_result
    text_latents = np.array(text_latents).astype('float32')
    motion_latents = np.array(motion_latents).astype('float32')

    # 数据校验
    assert len(text_latents) == len(motion_latents), "潜在向量数量不匹配"
    
    # 归一化处理（余弦相似度）
    faiss.normalize_L2(text_latents)
    faiss.normalize_L2(motion_latents)

    # 创建FAISS索引
    dimension = text_latents.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    

    index.add(motion_latents)  # 构建动作库

    # 执行检索（批量查询）
    k = 10
    _, indices = index.search(text_latents, k)  # indices形状为(N, k)

    # 向量化计算指标
    total = len(text_latents)
    target_ids = np.arange(total).reshape(-1, 1)  # 每个文本对应的正确目标ID
    
    # 计算各层准确率
    r1 = np.mean(np.any(indices[:, :1] == target_ids, axis=1))
    r2 = np.mean(np.any(indices[:, :2] == target_ids, axis=1))
    r3 = np.mean(np.any(indices[:, :3] == target_ids, axis=1))
    r5 = np.mean(np.any(indices[:, :5] == target_ids, axis=1))
    r10 = np.mean(np.any(indices[:, :10] == target_ids, axis=1))

    # 结果格式化
    results = {
        'R1': round(r1 * 100, 3),
        'R2': round(r2 * 100, 3),
        'R3': round(r3 * 100, 3),
        'R5': round(r5 * 100, 3),
        'R10': round(r10 * 100, 3)
    }

    # 控制台输出
    if verbose:
        print(f"FULL | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs")

    return results

# 使用示例
# metrics = calculate_retrieval_metrics(train_result, use_gpu=True)
import numpy as np

def compute_retrieval_metrics(sim_matrix, rounding=2):
    """
    sim_matrix: [N, N] 的相似度矩阵，其中 N 为 batch size，
                假定每个查询的正确匹配在对角线上
    返回：一个字典，包含 R@1, R@2, R@3, R@5, R@10 和 MedR
    """
    # 将相似度转换为距离（注意：越大的相似度对应越小的距离）
    dists = -sim_matrix  # 形状 [N, N]
    
    # 对每一行进行双重 argsort，得到每个查询中各候选的排名（0-indexed）
    ranks = np.argsort(np.argsort(dists, axis=1), axis=1)
    # ground truth 的排名为对角线上元素的排名
    gt_ranks = np.diag(ranks)  # 形状 [N]
    
    metrics = {}
    for k in [1, 2, 3, 5, 10]:
        # 计算排名小于 k 的查询比例，即 R@k
        metrics[f"R@{k}"] = 100 * np.mean(gt_ranks < k)
    
    # 中位排名：注意这里 rank 是从 0 开始的，因此加 1
    metrics["MedR"] = np.median(gt_ranks) + 1
    
    # 可选：四舍五入指标值
    for key in metrics:
        metrics[key] = round(metrics[key], rounding)
    
    return metrics

def add_noise(feats_ref):
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False
    )
    probs1 = torch.zeros(1000, device='cuda:0')
    probs1[:201] = 1 / 201  # 0到500的概率为1
    probs1[201:] = 0/ (1000 - 201)  # 201到maxT-1的概率为0
    noise = torch.randn_like(feats_ref)  # shape=[bs, 1, 256]
    timestep = torch.multinomial(probs1, num_samples=feats_ref.shape[0], replacement=True).long()
    noised_z = scheduler.add_noise(original_samples=feats_ref.clone(), noise=noise, timesteps=timestep)
    return feats_ref, timestep
    # return noised_z
def main():
    
    cfg = parse_args()
    set_seed(cfg.SEED_VALUE)
    dataset = get_dataset(cfg)
    # train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    model = MLD(cfg, dataset)
    state_dict = torch.load('checkpoints/mld_humanml/mld_humanml_v1.ckpt', map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()
    from GradGuidance.spm import SPM, process_T5_outputs
    name = 'TFS_KIT_Pured_T1000_M1T0_E47.pth'
    name = 'TFS_Pured_T1000_M0T0_E200.pth'
    reward_model = SPM(nfeats=263, clip_path='deps/sentence-t5-large', ckpt_path=f'/data/wwj/ckpt/T5_SPM/{name}', lambda_m2m=0, lambda_t2m=350).cuda()
    for epoch in range(0, 550, 10000):
        #FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E0_1e-1.pth
        #ckpt_path = f'/data/wwj/ckpt/T5_SPM/FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E{epoch}_1e-1.pth'
        # ckpt_path = f'/data/wwj/ckpt/T5_SPM/FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E{epoch}_1e-2.pth'
        # ckpt_path = f'/data/wwj/ckpt/T5_SPM/FT-[TFS_KIT_Pured_T1000_M1T0_E100]-_Pured_T1000_M1T0_E11.pth'
        ckpt_path = f'/data/wwj/ckpt/T5_SPM/FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E14_1e-2.pth'
        ckpt_path = f'/data/wwj/ckpt/T5_SPM/TFS_Pured_T1000_M0T0_E200.pth'
        spm_ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
        reward_model.load_state_dict(spm_ckpt, strict=False)
        a = 0
        for para in reward_model.parameters():
            a += para.sum()
        print(a)
        #print('Now Model State_Dict:', fuck(model))
        #train_result = [[], [], []]
        #for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        #    feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
        #    with torch.no_grad():
        #        t_len, token_emb, cls_token = process_T5_outputs(text, reward_model.clip)
        #    feats_ref = feats_ref.float().cuda()
        #    m_latent = reward_model.encode_motion(feats_ref, m_len)[0].squeeze().detach().cpu().numpy()
        #    t_latent = reward_model.encode_text(token_emb, t_len)[0].squeeze().detach().cpu().numpy()
        #    for j in range(len(batch['text'])):
        #        train_result[0].append(batch['text'][j])
        #        train_result[1].append(t_latent[j])
        #        train_result[2].append(m_latent[j])
        #calculate_retrieval_metrics_small_batches(train_result,epoch=epoch)

        test_result = [[], [], []]
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
            # feats_ref = add_noise(feats_ref)
            mask = batch['mask']

            with torch.no_grad():
                feats_ref = model(batch, add_noise)

            
            
            with torch.no_grad():
                t_len, token_emb, cls_token = process_T5_outputs(text, reward_model.clip)
            feats_ref = feats_ref.float().cuda()
            m_latent = reward_model.encode_motion(feats_ref, m_len)[0].squeeze()
            t_latent = reward_model.encode_text(token_emb, t_len)[0].squeeze()
            m_latent = F.normalize(m_latent, dim=-1).detach().cpu().numpy()  # 形状 [batch_size, d]
            t_latent = F.normalize(t_latent, dim=-1).detach().cpu().numpy()  # 形状 [batch_size, d]
            for j in range(len(batch['text'])):
                test_result[0].append(batch['text'][j])
                test_result[1].append(t_latent[j])
                test_result[2].append(m_latent[j])
            

        shuffle_index = np.load(f'./scripts/{len(test_result[0])}test_shuffle_index.npy')
        np.random.shuffle(shuffle_index)
        test_result[0] = [test_result[0][i] for i in shuffle_index]
        test_result[1] = [test_result[1][i] for i in shuffle_index]
        test_result[2] = [test_result[2][i] for i in shuffle_index]
        print(f'\n===={ckpt_path}==============T2M Results====================')
        # t_latents = np.array(test_result[1]).astype('float32')
        # m_latents = np.array(test_result[2]).astype('float32')
        # sim_matrix = m_latents @ t_latents.T
        # metrics = compute_retrieval_metrics(sim_matrix)
        # print(metrics)
        # metrics = compute_retrieval_metrics(sim_matrix.T)
        # print(metrics)
        # continue
        # exit(0)
        calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
        calculate_retrieval_metrics(test_result, epoch=epoch)
        temp = test_result[2]
        test_result[2] = test_result[1]
        test_result[1] = temp
        print(f'===={ckpt_path}==============M2T Results====================')
        calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
        calculate_retrieval_metrics(test_result, epoch=epoch)
        print(f'============================================================\n')
        
if __name__ == "__main__":
    main()
