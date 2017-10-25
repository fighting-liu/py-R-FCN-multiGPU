from __future__ import print_function
import os
import shutil
import stat
from google.protobuf import text_format

import _init_paths
from caffe.proto import caffe_pb2
from cfgs.arg_parser import config

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

## We assume you are running the script at the FRCNN-ROOT.
dev_root = os.getcwd()
assert dev_root.split('/')[-1] == 'py-R-FCN-multiGPU', dev_root.split('/')[-1]

###### general parameters 
dataset = 'shampoo1017'
data_type = 'coco'  #either 'voc' or 'coco'
task_type = 'rfcn' #either 'frcnn' or 'rfcn'
NET = 'ResNet101'  #ether 'VGG16' or 'ResNet101' or 'ResNet50by2' or 'ResNet77by2'
num_classes = 3 + 1 # plus 1 for background
# GPU_ID = '0,1'   ## int(0) or str('0,1,2,3')
GPU_ID = 0

finetune_from_voc_frcnn_vgg16_model = False #default, we fintune from vgg16 model


###### solver part parameters 
solver_param_base_lr = 0.001
solver_param_stepsize = 240000 
solver_param_iter_size = 2
##10 * num_imgaes(for base lr) + 4 * num_images(for base_lr*0.1)
exp_ITERS = 340000

## add ft_flag to classify caffelmodel type
ft_flag = '_ft' if finetune_from_voc_frcnn_vgg16_model else ''
solver_param_snapshot_prefix = '{}_{}_{}_{}{}'.format(NET, dataset, data_type, task_type, ft_flag)  


###### Making directorys if necessary
assert data_type=='voc' or data_type=='coco'
assert task_type=='frcnn' or task_type=='rfcn'
task_type_name = 'faster_rcnn_end2end' if task_type=='frcnn' else 'rfcn_end2end' 
logs_dir = 'experiments/{}/{}/logs'.format(dataset, NET+ft_flag)
scripts_dir = 'experiments/{}/{}/scripts'.format(dataset, NET+ft_flag)
cfgs_dir = 'experiments/{}/{}/cfgs'.format(dataset, NET+ft_flag)
model_dir = 'models/{}/{}/{}'.format(dataset, NET+ft_flag, task_type_name)
make_if_not_exist(model_dir)
make_if_not_exist(logs_dir)
make_if_not_exist(scripts_dir)
make_if_not_exist(cfgs_dir)


###### Model part: create new prototxt with respect to new dataset 
train_prototxt_name = 'train.prototxt' if task_type=='frcnn' else 'train_agnostic.prototxt'
test_prototxt_name = 'test.prototxt' if task_type=='frcnn' else 'test_agnostic.prototxt'     
train_origin = 'models/{}/{}/{}/{}'.format('base', NET, task_type_name, train_prototxt_name)
test_origin = 'models/{}/{}/{}/{}'.format('base', NET, task_type_name, test_prototxt_name)
assert os.path.exists(train_origin) and os.path.exists(test_origin), (train_origin, test_origin)

new_train_prototxt = os.path.join(model_dir, train_prototxt_name)
new_test_prototxt =  os.path.join(model_dir, test_prototxt_name)
if finetune_from_voc_frcnn_vgg16_model:
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(train_origin).read(), net_params)
    for l in net_params.layer:
        if l.name == 'input-data' or l.name == 'roi-data':
            l.python_param.param_str = "'num_classes': {}".format(str(num_classes))
        elif l.name == 'cls_score':
            l.name = 'cls_score_ft'
            l.inner_product_param.num_output = num_classes
        elif l.name == 'bbox_pred':
            l.name = 'bbox_pred_ft'
            l.inner_product_param.num_output = num_classes*4
    with open(new_train_prototxt, 'w') as f:
        f.write(str(net_params)) 

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(test_origin).read(), net_params)
    for l in  net_params.layer:
        if l.name == 'cls_score':
            l.name = 'cls_score_ft'
            l.inner_product_param.num_output = num_classes
        elif l.name == 'bbox_pred':
            l.name = 'bbox_pred_ft'
            l.inner_product_param.num_output = num_classes*4          
    with open(new_test_prototxt, 'w') as f:
        f.write(str(net_params))     
elif task_type=='frcnn':  
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(train_origin).read(), net_params)
    for l in net_params.layer:
        if l.name == 'input-data' or l.name == 'roi-data':
            l.python_param.param_str = "'num_classes': {}".format(str(num_classes))
        elif l.name == 'cls_score':
            l.inner_product_param.num_output = num_classes
        elif l.name == 'bbox_pred':
            l.inner_product_param.num_output = num_classes*4
    with open(new_train_prototxt, 'w') as f:
        f.write(str(net_params)) 

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(test_origin).read(), net_params)
    for l in  net_params.layer:
        if l.name == 'cls_score':
            l.inner_product_param.num_output = num_classes
        elif l.name == 'bbox_pred':
            l.inner_product_param.num_output = num_classes*4          
    with open(new_test_prototxt, 'w') as f:
        f.write(str(net_params)) 
elif task_type == 'rfcn':
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(train_origin).read(), net_params)
    for l in net_params.layer:
        if l.name == 'input-data':
            l.python_param.param_str = "'num_classes': {}".format(str(num_classes))
        elif l.name == 'rfcn_cls':
            l.convolution_param.num_output = num_classes*49 ##49=7^2
        elif l.name == 'rfcn_bbox':
            l.convolution_param.num_output = 4*2*49 ##(bbox_cor*bg/non-bg*score_map_size)
        elif l.name == 'psroipooled_cls_rois':
            l.psroi_pooling_param.output_dim = num_classes       
    with open(new_train_prototxt, 'w') as f:
        f.write(str(net_params)) 

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(test_origin).read(), net_params)
    for l in  net_params.layer:
        if l.name == 'rfcn_cls':
            l.convolution_param.num_output = num_classes*49 ##49=7^2
        elif l.name == 'rfcn_bbox':
            l.convolution_param.num_output = 4*2*49 ##(bbox_cor*bg/non-bg*score_map_size)
        elif l.name == 'psroipooled_cls_rois':
            l.psroi_pooling_param.output_dim = num_classes  
        # elif l.name == 'cls_prob_reshape':
        #     # print(l.reshape_param.shape.dim)
        #     # raise
        #     # print(caffe_pb2.BlobShape([-1, 21]))
        #     l.reshape_param.shape.dim = -1 
        #     l.reshape_param.shape.dim = 21          
    with open(new_test_prototxt, 'w') as f:
        f.write(str(net_params))     
else:
    raise             


###### Solver part
solver_file = os.path.join(model_dir, 'solver.prototxt')
solver_param = {
    'train_net': new_train_prototxt,
    'base_lr': solver_param_base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'stepsize': solver_param_stepsize,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': solver_param_iter_size,  
    'snapshot': 0,
    'snapshot_prefix': solver_param_snapshot_prefix,
    'display': 20,
    'average_loss': 100,
    'type': "SGD",
    }
solver = caffe_pb2.SolverParameter(**solver_param)
with open(solver_file, 'w') as f:
    print(solver, file=f)
  

###### Experiment part: 
## copy cfgs director to target file
base_cfg_dir = 'experiments/base/cfgs'
assert os.path.exists(base_cfg_dir)
yml_name = 'faster_rcnn_end2end.yml' if task_type=='frcnn' else 'rfcn_end2end.yml'
shutil.copy(os.path.join(base_cfg_dir, yml_name), cfgs_dir)
# shutil.copytree(base_cfg_dir, cfgs_dir, False)
TRAIN_LMDB = "{}_2007_trainval_{}".format(data_type, dataset)
TEST_LMDB = "{}_2007_test_{}".format(data_type, dataset)

if finetune_from_voc_frcnn_vgg16_model:
    weights = 'data/faster_rcnn_models/voc_vgg16_faster_rcnn_final.caffemodel'
else:
    weights = 'data/imagenet_models/{}.v2.caffemodel'.format(NET)    

## generating training scripts
train_sh_name = 'faster_rcnn_end2end.sh' if task_type=='frcnn' else 'rfcn_end2end.sh'

script_type = 'train_net.py' if type(GPU_ID) is int else 'train_net_multi_gpu.py'
with open(os.path.join(scripts_dir, train_sh_name), 'w') as f:
    f.write('#!/bin/bash\n')
    log_str = '''LOG="experiments/{}/{}/logs/{}_{}_{}.txt.`date +'%Y-%m-%d_%H-%M-%S'`" '''.format(dataset, NET+ft_flag, task_type_name, data_type, NET)
    f.write('set -x\n')
    f.write('set -e\n')
    f.write('export PYTHONUNBUFFERED="True"\n')
    f.write(log_str+'\n')
    f.write('exec &> >(tee -a "$LOG") \n')
    f.write('echo Logging output to {} \n'.format('"$LOG"'))

    f.write('time ./tools/{} --gpu {} \\\n'.format(script_type, GPU_ID))
    f.write('--solver {} \\\n'.format(solver_file))
    f.write('--weights {} \\\n'.format(weights))
    # f.write('--weights data/imagenet_models/{}.v2.caffemodel \\\n'.format(NET))
    f.write('--imdb {} \\\n'.format(TRAIN_LMDB))
    f.write('--iters {} \\\n'.format(exp_ITERS))
    f.write('--cfg {} \n'.format(os.path.join(cfgs_dir, yml_name)))
os.chmod(os.path.join(scripts_dir, train_sh_name), stat.S_IRWXU)

options = config(section_name='{}_test'.format(dataset), conf_file='lib/cfgs/detection.cfg')
annFile = options['input_json_file']
resFile = options['ouput_json_file']
class_names = ','.join(eval(options['classes']))

# ## generating training scripts
# train_sh_name = 'faster_rcnn_end2end.sh' if task_type=='frcnn' else 'rfcn_end2end.sh'
# with open(os.path.join(scripts_dir, train_sh_name), 'w') as f:
#     f.write('#!/bin/bash\n')
#     log_str = '''LOG="experiments/{}/{}/logs/{}_{}_{}.txt.`date +'%Y-%m-%d_%H-%M-%S'`" '''.format(dataset, NET+ft_flag, task_type_name, data_type, NET)
#     f.write('set -x\n')
#     f.write('set -e\n')
#     f.write('export PYTHONUNBUFFERED="True"\n')
#     f.write(log_str+'\n')
#     f.write('exec &> >(tee -a "$LOG") \n')
#     f.write('echo Logging output to {} \n'.format('"$LOG"'))

#     f.write('time ./tools/train_net.py --gpu {} \\\n'.format(GPU_ID))
#     f.write('--solver {} \\\n'.format(solver_file))
#     f.write('--weights {} \\\n'.format(weights))
#     # f.write('--weights data/imagenet_models/{}.v2.caffemodel \\\n'.format(NET))
#     f.write('--imdb {} \\\n'.format(TRAIN_LMDB))
#     f.write('--iters {} \\\n'.format(exp_ITERS))
#     f.write('--cfg {} \n'.format(os.path.join(cfgs_dir, yml_name)))
# os.chmod(os.path.join(scripts_dir, train_sh_name), stat.S_IRWXU)

# options = config(section_name='{}_test'.format(dataset), conf_file='lib/cfgs/detection.cfg')
# annFile = options['input_json_file']
# resFile = options['ouput_json_file']
# class_names = ','.join(eval(options['classes']))



# ############new_test_prototxt duplicate entrys
# ## generating testing scripts
# test_sh_name = 'frcnn_test.sh' if task_type=='frcnn' else 'rfcn_test.sh'
# with open(os.path.join(scripts_dir, test_sh_name), 'w') as f:
#     f.write('time ./tools/test_net.py --gpu {} \\\n'.format(GPU_ID))
#     f.write('--def models/{}/{}/{}/{} \\\n'.format(dataset, NET+ft_flag, task_type_name, test_prototxt_name))
#     f.write('--net output/{}/{}_2007_trainval_{}/{}_iter_{}.caffemodel \\\n'.format(task_type_name, data_type, dataset,\
#          solver_param_snapshot_prefix, exp_ITERS))
#     f.write('--imdb {} \\\n'.format(TEST_LMDB))
#     f.write('--cfg {} \n'.format(os.path.join(cfgs_dir, yml_name)))    
#     f.write('\n')
#     f.write('\n')
#     f.write('\n')
#     f.write('python /data/home/liuhuawei/detection/dev-py-faster-rcnn/lib/coco-master/PythonAPI/evaldemo.py \\\n')
#     f.write('--annFile {} \\\n'.format(annFile))
#     f.write('--resFile {} \\\n'.format(resFile))
#     f.write('--class_names {}'.format(class_names))
# os.chmod(os.path.join(scripts_dir, test_sh_name), stat.S_IRWXU)    














