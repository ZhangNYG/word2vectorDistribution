## master: 192.168.1.117
## 一台
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0
### --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0

### 参数服务器：
### python distribute.py --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=0
### 计算节点0
### python distribute.py --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=0
### 计算节点1
### python distribute.py --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=1

## 两台电脑
### 参数服务
### --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=0
### 计算节点0
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=0
### 计算节点1
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=1

## 两台电脑 linux
### 192.168.1.160
### 192.168.1.161
### 参数服务
### --job_name=ps --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=0
### 计算节点0
### --job_name=worker --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=0
### 计算节点1
### --job_name=worker --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=1

### tensorflow运行完之后ctrl+z停止之后彻底停止
### ps -ef|grep python|grep -v grep|cut -c 9-15|xargs kill -9


/home/cdh/workForWord2verctor/formalWord2vector


### 参数服务器：10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=ps --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=0


### 计算节点0: 10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=0


### 计算节点1: 10.1.0.42
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=1


### 计算节点2: 10.1.0.43
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=2


### 计算节点3: 10.1.0.218
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=3


### 计算节点4: 10.1.0.45
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=4



#####################################################
##增加计算节点个数

### 参数服务器：10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=ps --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=0

### 计算节点0: 10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=0

### 计算节点1: 10.1.0.42
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=1

### 计算节点2: 10.1.0.42
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=2


### 计算节点3: 10.1.0.43
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=3


### 计算节点4: 10.1.0.43
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=4

### 计算节点5: 10.1.0.218
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=5

### 计算节点6: 10.1.0.218
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=6

### 计算节点7: 10.1.0.45
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=7

### 计算节点8: 10.1.0.45
python LinuxDidtributionWord2vector.py \
--job_name=worker --ps_hosts=10.1.0.41:11111 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.42:22223,\
10.1.0.43:22224,\
10.1.0.43:22225,\
10.1.0.218:22226,\
10.1.0.218:22227,\
10.1.0.45:22228,\
10.1.0.45:22229 --task_id=8


####################################################
增加参数服务器
3个参数服务器

### 参数服务器：10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=ps \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=0

### 参数服务器：10.1.0.42
python LinuxDidtributionWord2vector.py \
--job_name=ps \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=1

### 参数服务器：10.1.0.43
python LinuxDidtributionWord2vector.py \
--job_name=ps \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=2


### 计算节点0: 10.1.0.41
python LinuxDidtributionWord2vector.py \
--job_name=worker \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=0

### 计算节点1: 10.1.0.42
python LinuxDidtributionWord2vector.py \
--job_name=worker \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=1

### 计算节点2: 10.1.0.43
python LinuxDidtributionWord2vector.py \
--job_name=worker \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=2


### 计算节点3: 10.1.0.218
python LinuxDidtributionWord2vector.py \
--job_name=worker \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=3


### 计算节点4: 10.1.0.45
python LinuxDidtributionWord2vector.py \
--job_name=worker \
--ps_hosts=\
10.1.0.41:11111,\
10.1.0.42:11112,\
10.1.0.43:11113 \
--worker_hosts=\
10.1.0.41:22221,\
10.1.0.42:22222,\
10.1.0.43:22223,\
10.1.0.218:22224,\
10.1.0.45:22225 --task_id=4

cp /home/cdh/anaconda3/envs/tensorflow/x86_64-conda_cos6-linux-gnu/sysroot/lib/libstdc++.so.6.0.24 /usr/lib64/
cd /usr/lib64/
cp libstdc++.so.6 libstdc++.so.6.bak
rm libstdc++.so.6
ln -s libstdc++.so.6.0.24 libstdc++.so.6


cd workForWord2verctor/formalWord2vector/
source activate tensorflow


