######## Data paths, need to replace them based on your directory
train_path=data/mad_data/train_v1.jsonl
eval_path=data/mad_data/val.jsonl
eval_split_name=val

######## Setup video/textual feature path, need to replace them based on your directory
motion_feat_dir=/s1_md0/leiji/v-zhijian/MAD/MAD_data/CLIP_frames_features_5fps
appearance_feat_dir=/s1_md0/leiji/v-zhijian/MAD/MAD_data/CLIP_frames_features_5fps
text_feat_dir=/s1_md0/leiji/v-zhijian/mad_data_for_cone/offline_lmdb/clip_clip_text_features

# Feature dimension
v_motion_feat_dim=512
v_appear_feat_dim=512
t_feat_dim=512

#### training
n_epoch=30
lr_drop=25
device_id=$1
num_queries=$2
max_v_l=$3
bsz=32
eval_bsz=16
clip_length=0.2 ##  video feature are extracted by 5 FPS, thus a clip duration is 1/5 = 0.2 second
max_q_l=25
num_workers=8


######## Hyper-parameter
dset_name=mad
seed=2020
results_root=cone_results
adapter_module=$4
max_es_cnt=10
eval_epoch_interval=1
topk_window=30
start_epoch_for_adapter=10
adapter_loss_coef=0.2
exp_id=exp_multi_${max_v_l}_${num_queries}_${adapter_module}_${clip_length}

CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python cone/train.py \
--seed ${seed} \
--clip_length ${clip_length}  \
--max_es_cnt ${max_es_cnt} \
--topk_window ${topk_window} \
--eval_epoch_interval ${eval_epoch_interval} \
--start_epoch_for_adapter ${start_epoch_for_adapter} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--dset_name ${dset_name} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--t_feat_dir ${text_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--v_appear_feat_dim ${v_appear_feat_dim} \
--v_motion_feat_dim ${v_motion_feat_dim} \
--bsz ${bsz} \
--eval_bsz ${eval_bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--num_queries ${num_queries} \
--num_workers ${num_workers} \
--adapter_module ${adapter_module} \
--adapter_loss_coef ${adapter_loss_coef} \
${@:5}
