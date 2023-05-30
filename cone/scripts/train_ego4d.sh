######## Data paths, need to replace them based on your directory
train_path=data/ego4d_data/train_v1.jsonl
eval_path=data/ego4d_data/val.jsonl
eval_split_name=val

######## Setup video/textual feature path, need to replace them based on your directory
motion_feat_dir=/s1_md0/leiji/v-zhijian/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_video_feature_1.87fps
appearance_feat_dir=/s1_md0/leiji/v-zhijian/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_video_feature_1.87fps
text_feat_dir=/s1_md0/leiji/v-zhijian/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_egovlp_text_features

########  Feature dimension
v_motion_feat_dim=256
v_appear_feat_dim=256
t_feat_dim=768


######## training
device_id=$1
num_queries=$2
max_v_l=$3
bsz=32
clip_length=0.535 ## we extract video feature every 1.87 FPS, thus a clip duration is 1/1.87 = 0.535 second
max_q_l=20
num_workers=4

######## Hyper-parameter
n_epoch=150
lr_drop=120
max_es_cnt=10
eval_epoch_interval=3
start_epoch_for_adapter=30
topk_window=20
dset_name=ego4d
results_root=cone_results
adapter_module=$4
exp_id=exp_${max_v_l}_${num_queries}_${adapter_module}_${clip_length}

CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python cone/train.py \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--max_es_cnt ${max_es_cnt} \
--eval_epoch_interval ${eval_epoch_interval} \
--start_epoch_for_adapter ${start_epoch_for_adapter} \
--clip_length ${clip_length}  \
--topk_window ${topk_window} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--dset_name ${dset_name} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--v_motion_feat_dim ${v_motion_feat_dim} \
--t_feat_dir ${text_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--v_appear_feat_dim ${v_appear_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--num_queries ${num_queries} \
--num_workers ${num_workers} \
--adapter_module ${adapter_module} \
--start_epoch -1 \
${@:5}
