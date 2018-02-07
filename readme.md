## master: 192.168.1.117
### 一台
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0
### --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0

### 参数服务器：
### python distribute.py --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=0
### 计算节点0
### python distribute.py --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=0
### 计算节点1
### python distribute.py --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.117:33333 --task_id=1

### 两台电脑
### 参数服务
### --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=0
### 计算节点0
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=0
### 计算节点1
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222,192.168.1.121:33333 --task_id=1

### 两台电脑 linux
### 192.168.1.160
### 192.168.1.161
### 参数服务
### --job_name=ps --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=0
### 计算节点0
### --job_name=worker --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=0
### 计算节点1
### --job_name=worker --ps_hosts=192.168.1.160:11111 --worker_hosts=192.168.1.160:22222,192.168.1.161:33333 --task_id=1